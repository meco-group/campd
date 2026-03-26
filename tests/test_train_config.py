
from campd.experiments.train import TrainExperimentCfg
from campd.data.trajectory_dataset import TrajectoryDatasetCfg
from campd.training.base import TrainerCfg
from campd.models.diffusion.model import ContextTrajectoryDiffusionModelCfg
from campd.utils.registry import Spec
import torch

# Mocks
mock_dataset_cfg = TrajectoryDatasetCfg(
    dataset_dir="/tmp/dataset",
    hdf5_file="data.hdf5"
)

mock_model_cfg = ContextTrajectoryDiffusionModelCfg(
    state_dim=7,
    network=Spec(cls="Linear", init={}),
    normalizer={}  # Mock normalizer config
)

mock_trainer_cfg = TrainerCfg(
    results_dir="/tmp/results",
    objective=Spec(cls="MSELoss", init={}),
    tensor_args={}  # Defaults OK?
)


def test_model_only():
    print("Testing model only...")
    try:
        cfg = TrainExperimentCfg(
            cls="train",
            dataset_dir="/tmp/dataset",
            train_file="train.hdf5",
            val_file="val.hdf5",
            dataset=mock_dataset_cfg,
            dataloader={},
            model=mock_model_cfg,
            trainer=mock_trainer_cfg,
        )
        print("Success: Model only config passed.")
    except Exception as e:
        print(f"Failure: Model only config failed: {e}")
        raise e


def test_pretrained_only():
    print("Testing pretrained only...")
    try:
        cfg = TrainExperimentCfg(
            cls="train",
            dataset_dir="/tmp/dataset",
            train_file="train.hdf5",
            val_file="val.hdf5",
            dataset=mock_dataset_cfg,
            dataloader={},
            pretrained_model_path="/tmp/pretrained_model",
            trainer=mock_trainer_cfg,
        )
        print("Success: Pretrained only config passed.")
    except Exception as e:
        print(f"Failure: Pretrained only config failed: {e}")
        raise e


def test_neither():
    print("Testing neither...")
    try:
        cfg = TrainExperimentCfg(
            cls="train",
            dataset_dir="/tmp/dataset",
            train_file="train.hdf5",
            val_file="val.hdf5",
            dataset=mock_dataset_cfg,
            dataloader={},
            trainer=mock_trainer_cfg,
        )
        print("Failure: Neither config passed (should have failed).")
    except ValueError as e:
        print(f"Success: Neither config failed as expected: {e}")
    except Exception as e:
        print(f"Failure: Unexpected error for neither: {e}")
        raise e


def test_both():
    print("Testing both...")
    try:
        cfg = TrainExperimentCfg(
            cls="train",
            dataset_dir="/tmp/dataset",
            train_file="train.hdf5",
            val_file="val.hdf5",
            dataset=mock_dataset_cfg,
            dataloader={},
            model=mock_model_cfg,
            pretrained_model_path="/tmp/pretrained_model",
            trainer=mock_trainer_cfg,
        )
        # This is allowed now, maybe strictly logic would prefer one,
        # but validation says "if self.model is None AND self.pretrained is None"
        # so both is valid.
        print("Success: Both config passed.")
    except Exception as e:
        print(f"Failure: Both config failed: {e}")
        raise e


if __name__ == "__main__":
    test_model_only()
    test_pretrained_only()
    test_neither()
    test_both()
