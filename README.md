# Context-Aware Motion Planning Diffusion


A modular framework for training, and running inference with context-aware diffusion models on trajectory data. It provides a registry-driven component system, YAML-based configuration, and built-in support for HuggingFace Accelerate (multi-GPU), CUDA graphs, Weights & Biases logging, and experiment sweeps.

This framework accompanies the paper:

> **Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models**
> Edward Sandra, Lander Vanroye, Dries Dirckx, Ruben Cartuyvels, Jan Swevers, Wilm Decré
> arXiv:2510.14615 — [https://arxiv.org/abs/2510.14615](https://arxiv.org/abs/2510.14615)

If you use this framework in your research, please cite:

```bibtex
@misc{sandra2025campd,
  title   = {Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models},
  author  = {Sandra, Edward and Vanroye, Lander and Dirckx, Dries and Cartuyvels, Ruben and Swevers, Jan and Decr\'{e}, Wilm},
  year    = {2025},
  eprint  = {2510.14615},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url     = {https://arxiv.org/abs/2510.14615},
}
```

---


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
  - [Registry System](#registry-system)
  - [The `Spec` Pattern](#the-spec-pattern)
  - [Configuration & Attribute Propagation](#configuration--attribute-propagation)
  - [Dependencies / Imports in YAML](#dependencies--imports-in-yaml)
- [Launching Experiments](#launching-experiments)
  - [Via `campd-run` (recommended)](#via-campd-run-recommended)
  - [Via custom Python script](#via-custom-python-script)
- [YAML Configuration Reference](#yaml-configuration-reference)
- [Extending the Framework](#extending-the-framework)
- [Built-in Components](#built-in-components)
- [Troubleshooting](#troubleshooting)

---

## Overview

CAMPD is built around a **diffusion-model pipeline for trajectory generation** conditioned on (but not limited to) environment context (e.g. obstacle geometries). The high-level flow is:

1. **Data** — Load trajectory datasets from HDF5 files (with context fields like cuboid/cylinder/sphere obstacle descriptions).
2. **Model** — A `ContextTrajectoryDiffusionModel` wrapping HuggingFace Diffusers schedulers, a reverse-diffusion denoising network (e.g. `TemporalUnet`), and an optional context encoder.
3. **Training** — A `Trainer` runs the training loop with configurable objectives, callbacks, summaries, multi-objective optimization (TorchJD), AMP, gradient clipping, and optional CUDA graph acceleration.
4. **Inference** — Load a trained checkpoint and sample trajectories, optionally validating them with domain-specific validators.

Everything is wired together through **YAML config files** and a **registry system**, so you can swap components without changing code.

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended)

### Steps

```bash
pip install git+https://github.com/meco-group/campd.git@v0.1.0
```

This installs the `campd` package and the `campd-run` CLI entry point. 

> **Note:** Some example projects (e.g. `examples/franka_curobo/`) may have additional dependencies not listed in `pyproject.toml` (e.g. `curobo`, `pinocchio`). These are installed separately; check each example's `requirements.txt`.

> **Note:** If you want to enable Weights and Biases logging, install it with `pip install wandb` and run `wandb login` to authenticate.

---

## Core Concepts

### Registry System

The framework uses a **registry pattern** to enable config-driven component selection. Each category of component has its own `Registry` instance:

| Registry           | Module                                | Purpose                           |
|:-------------------|:--------------------------------------|:----------------------------------|
| `EXPERIMENTS`      | `experiments/registry.py`             | Experiment types                  |
| `MODULES`          | `architectures/registry.py`           | Generic `nn.Module` building blocks |
| `REVERSE_NETS`     | `architectures/registry.py`           | Denoising networks                |
| `CONTEXT_NETS`     | `architectures/registry.py`           | Context encoder networks          |
| `LOSSES`           | `training/registry.py`               | Loss functions                    |
| `CALLBACKS`        | `training/registry.py`               | Training callbacks                |
| `SUMMARIES`        | `training/registry.py`               | Training summaries                |
| `OBJECTIVES`       | `training/registry.py`               | Training objectives               |
| `VALIDATORS`       | `experiments/validators.py`           | Inference validators              |

Components self-register using a decorator:

```python
from campd.training.registry import CALLBACKS

@CALLBACKS.register("MyCallback")
class MyCallback(Callback):
    ...
```

Then in the YAML config:

```yaml
callbacks:
  - cls: "MyCallback"
```

**Critical:** For a component to be available at runtime, its module must be **imported** before the registry lookup happens. This is handled by two mechanisms:

1. `campd/all_imports.py` — Bulk-imports all built-in subpackages (architectures, data, experiments, models, training), which triggers their `__init__.py` chains and populates the registries with built-in components.
2. The `dependencies` key in YAML config — Imports external/example-specific modules at startup (see [Dependencies / Imports in YAML](#dependencies--imports-in-yaml)).

### The `Spec` Pattern

`Spec` (defined in `utils/registry.py`) is a Pydantic model that describes **how to build an object** from config. It supports two modes:

#### Init Mode — Direct constructor kwargs

```yaml
optimizer:
  cls: "torch.optim.Adam"      # Full import path or registry key
  init:
    lr: 1.0e-4
    weight_decay: 0.0
```

This calls `torch.optim.Adam(lr=1e-4, weight_decay=0.0)`.

#### Config Mode — Factory method via `from_config`

```yaml
objective:
  cls: "DiffusionObjective"    # Registry key
  config:
    loss_fn:
      cls: "torch.nn.MSELoss"
      init:
        reduction: "mean"
```

This calls `DiffusionObjective.from_config(config_dict)`. The class must have a `from_config` classmethod.

#### Registry vs Import Path Resolution

- If a `registry` field is set on the `Spec`, the `cls` string is looked up in that specific registry.
- Otherwise, the `cls` string is first tried as a **registry key** (if a registry is passed to `build_from`), then as a **Python import path** (e.g. `torch.optim.Adam`).
- This means you can reference any importable class by its full dotted path, or use short registry keys for registered components.

### Configuration & Attribute Propagation

The framework uses **Pydantic models** for configuration validation. A key feature is **attribute propagation**: parent-level config values are automatically pushed down to nested child configs that share the same field name. For example:

```yaml
experiment:
  device: "cuda:0"           # Parent-level
  dataset:
    # device is NOT declared here, but if TrajectoryDatasetCfg has a
    # 'device' field, it will receive "cuda:0" from the parent.
    ...
  trainer:
    tensor_args:
      device: "cuda:0"       # Explicit — but could also be propagated
```

YAML anchors (`&name` / `*name`) can be used in config files for DRY configuration.

### Dependencies / Imports in YAML

The `dependencies` top-level key in YAML configs lists modules or directories that should be imported before the experiment runs. This is essential for **registering custom components** (e.g. example-specific summaries, validators, architectures):

```yaml
dependencies:
  - "../src"              # A directory — all .py files inside are imported
  - "my_custom_module"    # A Python module import path
  - "./my_file.py"        # A single Python file
```

Paths are resolved relative to the config file's directory.

**This is the mechanism that makes custom components available to the registries.** If you define a custom `@SUMMARIES.register("ValidationSummary")` class in `examples/franka_curobo/src/training_summary.py`, listing `"../src"` in `dependencies` ensures it's imported and registered before the config tries to reference `"ValidationSummary"`.

---

## Launching Experiments

### Via `campd-run` (recommended)

The `campd-run` CLI is installed as a console script by pip:

```bash
campd-run path/to/config.yaml
```

This:

1. Parses the YAML file.
2. Extracts `dependencies`, `experiment`, `wandb`, `launcher`, and `sweep` sections.
3. Imports built-in and user-defined dependencies to populate registries.
4. Uses [experiment-launcher](https://github.com/robot-learning-group/experiment-launcher) to manage experiment execution (seeding, output directories, optional SLURM submission).
5. Looks up the experiment class via `experiment.cls` in the `EXPERIMENTS` registry and calls its `run()` method.

**Launcher configuration** controls experiment management:

```yaml
launcher:
  exp_name: "my_experiment"
  n_seeds: 1                  # Number of seeds (repetitions)
  start_seed: 0
  base_dir: "results/"        # Output base directory
  use_timestamp: true         # Append timestamp to output dir
  resources: 
    n_exps_in_parallel: 1       # Parallel experiments
    ... (see experiment-launcher docs)
```

### Via custom Python script

For simpler use cases or debugging, you can bypass the launcher:

```python
import os
from campd.experiments import TrainExperiment

base_dir = os.path.dirname(os.path.abspath(__file__))
exp = TrainExperiment.from_yaml(os.path.join(base_dir, "configs/train.yaml"))
exp.run()
```

> **Note:** When using `from_yaml`, the `dependencies` section is **not** processed automatically. You must import your custom modules manually before calling `from_yaml` (e.g. `import my_custom_module`).
>
> The `campd-run` CLI handles this for you.

---

## YAML Configuration Reference

A full config file has up to five top-level sections:

```yaml
# 1. Dependencies — modules/directories to import for custom registrations
dependencies:
  - "../src"

# 2. WandB — Weights & Biases logging
wandb:
  mode: "online"           # "online", "offline", or "disabled"
  entity: "my-team"
  project: "my-project"
  group: "group_name"
  name: &name "run_name"
  

# 3. Launcher — experiment-launcher settings
launcher:
  exp_name: *name
  base_dir: "results/"
  n_seeds: 1
  # ... (see experiment-launcher docs)

# 4. Sweep — hyperparameter sweep (optional)
sweep:
  trainer:
    lr: [1e-4, 1e-3]      # Creates one run per value

# 5. Experiment — the actual experiment configuration
experiment:
  cls: "train"             # Registered experiment key

  # Common fields (from ExperimentCfg):
  seed: 42
  device: "cuda:0"
  # results_dir: set by launcher

  # Experiment-specific fields (e.g. TrainExperimentCfg):
  dataset_dir: "data/train/my_dataset"
  train_file: "train.hdf5"
  val_file: "val.hdf5"    # Optional
  # val_set_size: 0.1       # Optional

  dataset:
    trajectory_state: "pos"          # "pos", "pos+vel", "pos+vel+acc"
    field_config:
      trajectory_field: "solutions"  # HDF5 key for trajectory data
      q_dim: 7                       # Configuration-space dimension
      context_fields:                # Maps list of HDF5 keys -> context key
        cuboids: ["cuboid_centers", "cuboid_dims", "cuboid_quaternions"]
        # Note that the subkeys are still accessible inside the TrajectoryContext
        # object
      # also possible to use a list of HDF5 keys
      # context_fields:
      #   - "cuboid_centers"
      #   - "cuboid_dims"
      #   - "cuboid_quaternions"
    # ...

  model:
    state_dim: 7
    model_type: "epsilon"            # "epsilon", "sample", or "v_prediction"
    n_diffusion_steps: 25
    network:                         # Spec for reverse diffusion network
      cls: "TemporalUnet"
      config: { ... }
    context_network:                 # Spec for context encoder (optional)
      cls: "campd.architectures.context.encoder.ContextEncoder"
      config: { ... }

  trainer:
    max_epochs: 200
    optimizer:
      cls: "torch.optim.Adam"
      init: { lr: 1e-4 }
    objective:
      cls: "DiffusionObjective"
      config:
        loss_fn:
          cls: "torch.nn.MSELoss"
          init: { reduction: "mean" }
    callbacks:
      - cls: "PrinterCallback"
      - cls: "EMACallback"
        init: { decay: 0.995 }
      - cls: "CheckpointCallback"
        init: { save_best: true }
      - cls: "WandBCallback"
    summaries:
      - cls: "ValidationSummary"      # Custom (from dependencies)
        init: { every_n_steps: 2500 }
```

---

## Extending the Framework

The general pattern for adding a new component:

1. **Create a Python file** with your class, inheriting from the appropriate base class.
2. **Decorate** it with `@REGISTRY.register("key")` using the relevant registry.
3. **Make sure it's imported** at startup — either by placing it in a built-in subpackage (and re-exporting via `__init__.py`), or by listing its module/directory in the `dependencies` section of your YAML config.
4. **Reference it** in your YAML config via the registry key.

### Registering a New Experiment

```python
# my_experiments/custom_exp.py
from campd.experiments.base import BaseExperiment, ExperimentCfg
from campd.experiments.registry import EXPERIMENTS
from pydantic import validate_call

class MyExperimentCfg(ExperimentCfg):
    my_param: str = "default"

@EXPERIMENTS.register("my_experiment")
class MyExperiment(BaseExperiment):
    CfgClass = MyExperimentCfg

    @validate_call
    def __init__(self, cfg: MyExperimentCfg):
        super().__init__(cfg)

    def run(self):
        print(f"Running with param: {self.cfg.my_param}")
```

```yaml
dependencies:
  - "my_experiments"        # Directory containing custom_exp.py

experiment:
  cls: "my_experiment"
  my_param: "hello"
```

### Registering a New Network Architecture

```python
# my_networks/custom_net.py
import torch.nn as nn
from campd.architectures.registry import REVERSE_NETS
from campd.utils.registry import FromCfg

@REVERSE_NETS.register("MyDenoiser")
class MyDenoiser(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        # ... build layers ...

    @classmethod
    def from_config(cls, cfg):
        if isinstance(cfg, dict):
            return cls(**cfg)
        return cls(**cfg.model_dump())

    def forward(self, x, t, context=None):
        # x: [B, T, state_dim], t: [B], context: EmbeddedContext or None
        ...
```

```yaml
model:
  network:
    cls: "MyDenoiser"
    config:
      state_dim: 7
      hidden_dim: 128
```

### Registering a New Training Callback

```python
from campd.training.callbacks import Callback
from campd.training.registry import CALLBACKS

@CALLBACKS.register("LRLoggerCallback")
class LRLoggerCallback(Callback):
    def on_epoch_end(self, trainer, train_losses=None):
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Current LR: {lr}")
```

Available hooks: `on_train_start`, `on_fit_start`, `on_train_end`, `on_epoch_start`, `on_epoch_end`, `on_batch_start`, `on_batch_end`, `on_validation_start`, `on_validation_end`, `on_summary_end`.

### Registering a New Training Summary

```python
from campd.training.summary import Summary
from campd.training.registry import SUMMARIES

@SUMMARIES.register("MySummary")
class MySummary(Summary):
    def __init__(self, every_n_steps=1000):
        super().__init__(every_n_steps=every_n_steps)

    def _run(self, model, train_dataloader, val_dataloader, step):
        # Generate samples, compute metrics, return dict/figures
        return {"my_metric": 0.95}
```

### Registering a New Training Objective

```python
from campd.training.objectives.base import TrainingObjective
from campd.training.registry import OBJECTIVES

@OBJECTIVES.register("MyObjective")
class MyObjective(TrainingObjective):
    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def step(self, model, batch):
        # Return: (losses_dict, model_output, info_dict)
        loss = ...
        return {"my_loss": loss}, model_out, {}
```

### Registering a New Validator

```python
from campd.experiments.validators import Validator, VALIDATORS

@VALIDATORS.register("MyValidator")
class MyValidator(Validator):
    def validate(self, batch, output_dir):
        # Return dict of validation metrics
        return {"success_rate": 0.85}
```

---

## Built-in Components

### Experiments
| Key            | Class                      | Description                                |
|:---------------|:---------------------------|:-------------------------------------------|
| `"train"`      | `TrainExperiment`          | Full training pipeline (data → model → fit)|
| `"inference"`  | `InferenceExperiment`      | Load checkpoint & sample trajectories      |

### Callbacks
| Key                    | Description                                       |
|:-----------------------|:--------------------------------------------------|
| `"PrinterCallback"`   | Logs training start/end messages                  |
| `"EMACallback"`       | Exponential moving average of model weights       |
| `"CheckpointCallback"`| Saves checkpoints (best, last, periodic)          |
| `"WandBCallback"`     | Logs metrics/artifacts to Weights & Biases        |
| `"EarlyStoppingCallback"` | Stops training when validation loss plateaus  |

### Objectives
| Key                      | Description                                  |
|:-------------------------|:---------------------------------------------|
| `"DiffusionObjective"`   | Standard diffusion loss (ε, sample, or v)    |

### Losses
| Key            | Description      |
|:---------------|:-----------------|
| `"WeightedL1"` | Weighted L1 loss |
| `"WeightedL2"` | Weighted L2 loss |
| `"MSE"`        | `nn.MSELoss`     |
| `"L1"`         | `nn.L1Loss`      |

---

## Troubleshooting

- **CUDA graph errors**: If `use_cuda_graph: true` and you get runtime errors, ensure your PyTorch CUDA version matches the system CUDA version. Also verify that all tensor shapes remain constant across batches (CUDA graphs require fixed shapes).

- **`KeyError: Unknown 'X' in registry 'Y'`**: The component `X` is not registered. Ensure:
  1. The module defining the component is imported before the registry lookup.
  2. The module is listed in the `dependencies` section of your config.
  3. The `@REGISTRY.register("X")` decorator is present on the class.

If your issue isn't covered above, please [open a GitHub issue](../../issues) with a minimal reproducible example and the full error traceback.

## License

See [LICENSE](LICENSE) for details.