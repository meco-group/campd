from __future__ import annotations
from typing import Any, Dict, Optional, Union

import os
from pydantic import BaseModel, ConfigDict, Field, model_validator, validate_call
from torch.utils.data import DataLoader

from campd.data.trajectory_dataset import TrajectoryDataset, TrajectoryDatasetCfg
from campd.models.diffusion.model import ContextTrajectoryDiffusionModel, ContextTrajectoryDiffusionModelCfg
from campd.training.base import TrainerCfg, Trainer
from campd.experiments.base import BaseExperiment, ExperimentCfg
from campd.experiments.registry import EXPERIMENTS
from accelerate import Accelerator


class AccelerateLauncherCfg(BaseModel):
    """Configuration for HuggingFace Accelerate launcher.

    Args:
        num_processes (int): Number of processes to use for training.
        args (dict): Arguments to pass to the Accelerator.
    """
    num_processes: int = 1
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    args: dict = Field(default_factory=dict)


class TrainExperimentCfg(ExperimentCfg):
    """
    Configuration for a complete training experiment.

    Args:
        dataset_dir (str): Directory containing the dataset.
        train_file (str): Path to the training dataset.
        val_file (Optional[str]): Path to the validation dataset.
        val_set_size (float): Fraction of the training dataset to use as validation.
        dataset (TrajectoryDatasetCfg): Configuration for the dataset.
        dataloader (Dict[str, Union[int, str, float]]): Configuration for the dataloader.
        model (ContextTrajectoryDiffusionModelCfg): Configuration for the model.
        pretrained_model_path (Optional[str]): Path to a pretrained model. If provided, the model will be loaded from this path.
        trainer (TrainerCfg): Configuration for the trainer.
        results_dir (str): Directory to save the results.
        accelerate (AccelerateLauncherCfg): Configuration for HuggingFace Accelerate.

    If val_file is not provided, the dataset will be split into train and validation sets
    using the val_set_size.
    """
    dataset_dir: str
    train_file: str
    val_file: Optional[str] = None
    val_set_size: Optional[float] = None
    pretrained_model_path: Optional[str] = None
    dataset: TrajectoryDatasetCfg
    dataloader: Dict[str, Union[bool, int, str, float]] = Field(
        default_factory=lambda: {"batch_size": 32, "num_workers": 4})
    model: Optional[ContextTrajectoryDiffusionModelCfg] = None
    trainer: TrainerCfg
    accelerate: Optional[AccelerateLauncherCfg] = None

    @model_validator(mode="after")
    def check_val_source(self) -> "TrainExperimentCfg":
        if self.val_file is None and self.val_set_size is None:
            raise ValueError(
                "Either val_file or val_set_size must be specified.")
        return self

    @model_validator(mode="after")
    def check_model_source(self) -> "TrainExperimentCfg":
        if self.__class__ == TrainExperimentCfg:
            if self.model is None and self.pretrained_model_path is None:
                raise ValueError(
                    "Either model or pretrained_model_path must be specified.")

        if self.pretrained_model_path is not None:
            if self.model is not None:
                print('Warning: a model config is provided but a pretrained_model_path is also provided. The model config will be ignored.')

        return self

    @model_validator(mode="before")
    @classmethod
    def set_dataset_hdf5_file(cls, values: dict[str, Any]) -> dict[str, Any]:
        dataset = values.get("dataset")
        train_file = values.get("train_file")

        if dataset is not None:
            if isinstance(dataset, dict):
                if dataset.get("hdf5_file") is None:
                    dataset["hdf5_file"] = train_file
            elif hasattr(dataset, "hdf5_file"):
                if getattr(dataset, "hdf5_file") is None:
                    setattr(dataset, "hdf5_file", train_file)

        return values


@EXPERIMENTS.register("train")
class TrainExperiment(BaseExperiment):
    """
    Base class for running experiments.
    Orchestrates data loading, model initialization, and training.
    """
    CfgClass = TrainExperimentCfg

    @validate_call
    def __init__(self, cfg: TrainExperimentCfg):
        super().__init__(cfg)
        self.cfg: TrainExperimentCfg = cfg

        self.dataset: Optional[TrajectoryDataset] = None
        self.dataset_val: Optional[TrajectoryDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.model: Optional[ContextTrajectoryDiffusionModel] = None
        self.trainer: Optional[Trainer] = None
        self.accelerator: Optional[Accelerator] = None

    @property
    def _is_main_process(self) -> bool:
        """True on rank-0 (or when not using distributed)."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process

    def _print(self, *args, **kwargs):
        """Print only on the main process."""
        if self._is_main_process:
            print(*args, **kwargs)

    def init_dataset(self):
        """Initialize the dataset."""
        self._print("Initializing dataset...")
        train_cfg = self.cfg.dataset.model_copy()
        train_cfg.hdf5_file = os.path.basename(self.cfg.train_file)
        self.dataset = TrajectoryDataset(train_cfg)

        if self.cfg.val_file is not None:
            val_cfg = self.cfg.dataset.model_copy()
            val_cfg.hdf5_file = os.path.basename(self.cfg.val_file)
            self.dataset_val = TrajectoryDataset(val_cfg)

    def get_dataloaders(self):
        """Initialize dataloaders."""
        if self.dataset is None:
            raise RuntimeError(
                "Dataset not initialized. Call init_dataset() first.")

        self._print("Initializing dataloaders...")
        if self.cfg.val_file is not None:
            if self.dataset_val is None:
                raise RuntimeError(
                    "Validation dataset not initialized but val_file provided.")

            self.train_loader = self.dataset.get_dataloader(
                **self.cfg.dataloader
            )

            self.val_loader = self.dataset_val.get_dataloader(
                **self.cfg.dataloader
            )
        else:
            train_indices, val_indices = self.dataset.random_split(
                self.cfg.val_set_size, save=os.path.join(
                    self.cfg.results_dir, "train_val_split.pt"))

            self.train_loader = self.dataset.get_dataloader(
                indices=train_indices,
                **self.cfg.dataloader
            )

            self.val_loader = self.dataset.get_dataloader(
                indices=val_indices,
                **self.cfg.dataloader
            )

        self._print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

    def init_model(self):
        """Initialize the model."""
        self._print("Initializing model...")

        if self.cfg.pretrained_model_path is not None:
            self._print(
                f"Loading pretrained model from {self.cfg.pretrained_model_path}...")
            self.model = ContextTrajectoryDiffusionModel.from_pretrained(
                self.cfg.pretrained_model_path,
                device=self.cfg.dataset.tensor_args.device,
                freeze_params=False,
                model_iter="last"
            )

            if self.cfg.model is None:
                self.cfg.model = self.model.config

            # Overwrite dataset normalizer with pretrained model normalizer
            assert self.dataset is not None, "Dataset not initialized. Call init_dataset() first."
            norm_cfg = self.model.normalizer.export_config()
            self.dataset.set_normalizer(norm_cfg)
            if self.dataset_val is not None:
                self.dataset_val.set_normalizer(norm_cfg)
        else:
            self.cfg.model.normalizer = self.dataset.normalizer.export_config()
            self.model = ContextTrajectoryDiffusionModel(
                self.cfg.model
            )

        # Save model config
        self.model.save_config(self.cfg.results_dir)

    def init_trainer(self):
        """Initialize the trainer."""
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call init_model() first.")

        self._print("Initializing trainer...")
        self.trainer = Trainer(
            self.cfg.trainer, model=self.model, accelerator=self.accelerator)

    def train(self):
        """Run the training loop."""
        if self.trainer is None or self.train_loader is None:
            raise RuntimeError("Trainer or dataloaders not initialized.")

        self._print("Starting training...")
        self.trainer.fit(self.train_loader, self.val_loader)

    def _execute_training(self):
        """Training logic executed inside the notebook_launcher process."""
        if self.cfg.accelerate:
            args = self.cfg.accelerate.model_dump()["args"]
            self.accelerator = Accelerator(
                mixed_precision=self.cfg.accelerate.mixed_precision,
                gradient_accumulation_steps=self.cfg.accelerate.gradient_accumulation_steps,
                **args)
            print(f"Accelerator initialized: device={self.accelerator.device}")

            # Ensure dataset and model use the correct device assigned by Accelerator
            self.cfg.dataset.tensor_args.device = self.accelerator.device
            self.cfg.trainer.tensor_args.device = self.accelerator.device
            self.cfg.device = str(self.accelerator.device)
            if self.cfg.model and self.cfg.model.normalizer:
                # Ensure normalizer also respects the device
                pass  # Normalizer usually moves with model.to()

        self.init_dataset()
        self.init_model()
        self.get_dataloaders()
        self.init_trainer()
        self.train()

    @staticmethod
    def _distributed_entrypoint(cfg_dict: Dict[str, Any]):
        """Static entrypoint for distributed training to avoid pickling issues.

        Reconstructs the experiment from the configuration dictionary and runs training.
        """
        # Reconstruct config from JSON-dict (handles Spec serialization)
        cfg = TrainExperimentCfg.model_validate(cfg_dict)

        # Instantiate and run experiment
        exp = TrainExperiment(cfg)
        exp._execute_training()

    def run(self):
        """Run the full experiment pipeline."""
        if self.cfg.accelerate is None:
            # No accelerate config — run directly
            self._execute_training()
        else:
            from torch.distributed.launcher.api import LaunchConfig, elastic_launch
            import os

            # Use spawn to avoid "Cannot re-initialize CUDA" errors.
            # Convert config to JSON dict to avoid pickling errors with Spec objects.
            cfg_dict = self.cfg.model_dump(mode='json')

            config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=self.cfg.accelerate.num_processes,
                start_method="spawn",
                run_id="campd_run",
                rdzv_backend="c10d",
                rdzv_endpoint="localhost:29500",
                max_restarts=0,
            )
            elastic_launch(config, self._distributed_entrypoint)(cfg_dict)
