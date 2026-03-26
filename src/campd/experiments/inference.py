from pydantic import validate_call
from campd.utils.stats import summarize_all_stats
from collections import defaultdict
import yaml
from campd.utils.torch import TimerCUDA
import os
from typing import Optional

import torch

from campd.utils.registry import Spec
from campd.data.trajectory_dataset import TrajectoryDataset, TrajectoryDatasetCfg
from campd.models.diffusion.model import ContextTrajectoryDiffusionModel, SamplingCfg
from campd.experiments.base import BaseExperiment, ExperimentCfg
from campd.experiments.validators import Validator
from campd.experiments.registry import EXPERIMENTS
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class InferenceExperimentCfg(ExperimentCfg):
    """
    Configuration for an inference experiment.
    """
    model_dir: str
    model_iter: Optional[str] = None

    dataset: TrajectoryDatasetCfg
    validator: Optional[Spec[Validator]] = None

    # Optional default sampling config to be used if not provided in run
    sampling_cfg: Optional[SamplingCfg] = None
    save_samples: bool = True


@EXPERIMENTS.register("inference")
class InferenceExperiment(BaseExperiment):
    """Loads a trained diffusion model checkpoint and runs sampling."""
    CfgClass = InferenceExperimentCfg

    @validate_call
    def __init__(self, cfg: InferenceExperimentCfg):
        super().__init__(cfg)
        self.cfg: InferenceExperimentCfg = cfg

        self._init_model()
        self._init_dataset()
        self._init_validator()

    def _init_model(self):
        print('#' * 106)
        print(f'Model -- {os.path.dirname(self.cfg.model_dir)}')
        print('#' * 106)

        self.model = ContextTrajectoryDiffusionModel.from_pretrained(
            self.cfg.model_dir,
            device=self.cfg.device,
            freeze_params=True,
            model_iter=self.cfg.model_iter
        )

    def _init_dataset(self):
        self.dataset = TrajectoryDataset(self.cfg.dataset)

    def _init_validator(self):
        if self.cfg.validator is not None:
            self.validator = self.cfg.validator.build()
        else:
            self.validator = None

    def run(self):
        """Run inference on the dataset."""
        if self.cfg.sampling_cfg is None:
            raise ValueError(
                "sampling_cfg must be provided in config for run()")

        sampling_cfg = self.cfg.sampling_cfg.model_copy()
        self.model.prepare_for_sampling(sampling_cfg)

        # warm up
        for _ in range(5):
            self.model.sample(self.dataset[0].context.to(self.cfg.device))

        all_stats = defaultdict(list)
        print('Starting inference...')
        pbar = tqdm(self.dataset)

        for idx, sample in enumerate(pbar):
            buf = sample.context.to(self.cfg.device)
            save_dir = os.path.join(
                self.cfg.results_dir, 'data', f'{idx:06d}')
            os.makedirs(save_dir, exist_ok=True)
            with TimerCUDA() as timer:
                gen = self.model.sample(buf)

            gen_time = timer.elapsed

            stats = {}
            if self.validator is not None:
                stats = self.validator.validate(gen, save_dir)

            stats.update({
                'gen_time': gen_time,
            })
            pbar.set_postfix({
                'gen_time': gen_time,
            })
            with open(os.path.join(save_dir, 'stats.yaml'), 'w') as f:
                yaml.dump(stats, f)

            if self.cfg.save_samples:
                torch.save(gen, os.path.join(save_dir, 'gen.pt'))

            for key in stats:
                all_stats[key].append(stats[key])

        all_stats = summarize_all_stats(all_stats)

        with open(os.path.join(self.cfg.results_dir, 'stats.yaml'), 'w') as f:
            yaml.dump(all_stats, f)
