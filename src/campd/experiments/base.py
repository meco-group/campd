from __future__ import annotations
from abc import ABC, abstractmethod

import yaml
import random
import os
from typing import Any, Optional
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, validate_call
from campd.utils.torch import get_torch_device
from campd.utils.config import propagate_attributes_dict, process_imports

from experiment_launcher import run_experiment, single_experiment_yaml
from .registry import EXPERIMENTS


class DependenciesCfg(BaseModel):
    base_dir: str | None = None
    modules: list[str] = []

    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(BaseModel):
    """
    Base configuration for an experiment.
    """
    cls: str
    dependencies: Optional[DependenciesCfg] = None
    seed: int = 42

    # We might want to include device here too if it's common
    device: str = "cuda"
    results_dir: str = None

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        kwargs = propagate_attributes_dict(kwargs, self.__class__)
        super().__init__(**kwargs)


class BaseExperiment(ABC):
    """
    Base class for running experiments.
    """

    @validate_call
    def __init__(self, cfg: ExperimentCfg):
        self.cfg = cfg

        # import modules so that registries are populated
        import campd.all_imports
        if cfg.dependencies is not None:
            process_imports(cfg.dependencies.modules,
                            cfg.dependencies.base_dir)

        self.device = get_torch_device(cfg.device)
        self._set_seed(self.cfg.seed)
        self._setup_results_dir()

        print("Experiment directory: ", self.cfg.results_dir)
        print("Device: ", self.device)
        print("Seed: ", self.cfg.seed)

    def _setup_results_dir(self):
        """Setup the output directory."""
        if self.cfg.results_dir is None:
            raise ValueError("Output directory not specified.")
        if not os.path.exists(self.cfg.results_dir):
            os.makedirs(self.cfg.results_dir)

    @classmethod
    def run_from_config(cls, *args, **kwargs) -> None:
        @single_experiment_yaml(input_is_config=True)
        def _run(cfg: dict) -> None:
            if 'dependencies' in cfg:
                process_imports(cfg['dependencies']['modules'],
                                cfg['dependencies']['base_dir'])
            exp_cls = EXPERIMENTS[cfg["cls"]]
            exp = exp_cls(cfg)
            exp.run()

        _run(*args, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> BaseExperiment:
        """Load experiment from a YAML configuration file."""
        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            if "experiment" in cfg_dict:
                cfg_dict = cfg_dict["experiment"]

        if hasattr(cls, 'CfgClass'):
            return cls(cls.CfgClass.model_validate(cfg_dict))

        return cls(cfg_dict)

    def _set_seed(self, seed: int):
        """Set global random seeds."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def run(self) -> None:
        """Run the experiment. Subclasses should implement this."""
        raise NotImplementedError


# needed for experiment_launcher
def experiment(*args, **kwargs):
    return BaseExperiment.run_from_config(*args, **kwargs)
