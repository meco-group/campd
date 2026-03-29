# Extending the Framework

The core philosophy of CAMPD is that you can add virtually any functionality without modifying the source framework.

Ensure you adhere to the general pattern:
1. Inherit from the requisite built-in Base class (`Callback`, `BaseExperiment`, `Summary`, etc).
2. Override the abstract implementation methods.
3. Decorate your custom class with the `@REGISTRY.register("CustomName")`.
4. Import the file utilizing the `dependencies` YAML array.

## Registering an Experiment

To implement a completely novel execution loop (e.g. reinforcement learning fine-tuning or specialized inference loops).

```python
from campd.experiments.base import BaseExperiment, ExperimentCfg
from campd.experiments.registry import EXPERIMENTS

class RLTrainingCfg(ExperimentCfg):
    actor_critic: str = "ppo"
    penalty_weight: float = 0.5

@EXPERIMENTS.register("rl_trainer")
class RLTrainer(BaseExperiment):
    CfgClass = RLTrainingCfg     # Dictates what configuration Schema to parse

    def __init__(self, cfg: RLTrainingCfg):
        super().__init__(cfg)    # Automatically seeds and builds output-dir

    def run(self):               # Abstract method you must implement
        print(f"Executing loop with {self.cfg.actor_critic}")
```

## Registering Custom Network Architectures

Whether implementing a bespoke denoiser (`REVERSE_NETS`) or a unique vision encoder (`CONTEXT_NETS`).

```python
import torch.nn as nn
from campd.architectures.registry import REVERSE_NETS
from campd.utils.registry import FromCfg

@REVERSE_NETS.register("MyDenoiser")
class MyDenoiser(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Linear(state_dim, hidden_dim)

    # Implements the FromCfg protocol. Passed exactly what is in YAML 'config'.
    @classmethod
    def from_config(cls, cfg):
        if isinstance(cfg, dict):
            return cls(**cfg)
        return cls(**cfg.model_dump())

    def forward(self, x, t, context=None):
        return self.net(x)
```

## Registering Custom Training Objectives

To utilize customized noise schedulers, implement your own step loss definitions based off of `TrainingObjective`.

```python
from campd.training.objectives.base import TrainingObjective
from campd.training.registry import OBJECTIVES

@OBJECTIVES.register("WeightedMSEObjective")
class WeightedMSEObjective(TrainingObjective):
    
    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def step(self, model, batch):
        """
        Executed every training/validation step.
        Returns: Tuple of (Dict of Losses, Output Dict, Meta Info Dict)
        """
        x, context = batch
        # custom forward + logic
        loss = ...
        return {"total_loss": loss}, {}, {}
```

## Registering Training Callbacks

```python
from campd.training.callbacks import Callback
from campd.training.registry import CALLBACKS

@CALLBACKS.register("LRLogger")
class LRLogger(Callback):
    def on_epoch_end(self, trainer, train_losses=None):
        print(trainer.optimizer.param_groups[0]['lr'])
```
**Available Callback Hooks:**
`on_train_start`, `on_fit_start`, `on_train_end`, `on_epoch_start`, `on_epoch_end`, `on_batch_start`, `on_batch_end`, `on_validation_start`, `on_validation_end`, `on_summary_end`.

## Registering Inference Validators

Used within specific `InferenceExperiments` to perform validation operations immediately following continuous sample generation (e.g. running a kinematic solver to check trajectory collisions).
```python
from campd.experiments.validators import Validator, VALIDATORS

@VALIDATORS.register("KinematicCollisionValidator")
class KinematicCollisionValidator(Validator):
    def validate(self, batch, output_dir):
        # ... logic ...
        return {"collision_free_rate": 0.95}
```
