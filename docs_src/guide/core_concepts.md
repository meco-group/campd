# Core Concepts

CAMPD is built around a flexible, modular architecture designed to separate the definition of physical components from their configuration and instantiation.

## Registry System

The framework uses a **registry pattern** to enable configuration-driven component selection. This means you do not need to alter hardcoded imports when testing different models, experiments, or datasets; you simply update a string in a YAML file.

Each category of component has its own `Registry` instance.
| Registry | Module | Purpose |
|---|---|---|
| `EXPERIMENTS` | `experiments/registry.py` | Experiment runners (Train vs Inference) |
| `MODULES` | `architectures/registry.py` | Generic `nn.Module` building blocks |
| `REVERSE_NETS` | `architectures/registry.py` | Denoising networks inside the Diffusion model |
| `CONTEXT_NETS` | `architectures/registry.py` | Context encoder networks |
| `LOSSES` | `training/registry.py` | Torch loss functions (`nn.Module`) |
| `CALLBACKS` | `training/registry.py` | Training callbacks for metrics and operations |
| `SUMMARIES` | `training/registry.py` | Generation and logging of periodic summaries |
| `OBJECTIVES` | `training/registry.py` | Implementation of diffusion loss steps |
| `VALIDATORS` | `experiments/validators.py` | Inference validation logic |

Custom components can self-register using decorators:
```python
from campd.training.registry import CALLBACKS

@CALLBACKS.register("MyCallback")
class MyCallback(Callback):
    ...
```

**How it works under the hood**: For a registry string (e.g., `"MyCallback"`) to be resolved, the module containing that class MUST be imported before the registry lookup occurs. For built-in CAMPD components, this is automatically handled by `campd/all_imports.py` when an experiment starts. For your own custom classes, you must use the YAML `dependencies` key (see Configuration sections).

## The `Spec` Pattern

A `Spec` (defined in `utils/registry.py`) is a Pydantic model that describes exactly **how to build an object** from a configuration. 

Because we decouple instantiation from configuration, a `Spec` allows you to declare dynamic pipelines using either direct constructor arguments or factory methods.

### Init Mode (Direct kwargs)
Pass arguments directly to an imported class's `__init__` constructor.
```yaml
optimizer:
  cls: "torch.optim.Adam"      # Full import path
  init:
    lr: 1.0e-4
    weight_decay: 0.0
```

### Config Mode (Factory method)
Certain complex objects (like Objectives or Modules) benefit from deeply nested dictionaries validating their properties. Using `config`, the class must implement a `@classmethod def from_config(cls, cfg):` factory to handle initialization.
```yaml
objective:
  cls: "DiffusionObjective"    # Registry key
  config:
    loss_fn:
      cls: "torch.nn.MSELoss"
      init:
        reduction: "mean"
```

The system first checks if `cls` is a known registry key. If not, it attempts to resolve it as a standard Python import path (`pkg.module.ClassName`).

## Configuration and Attribute Propagation

Configurations in CAMPD are validated using **Pydantic V2**. 
A major feature of our configuration structure is **attribute propagation**: parent-level configurations automatically push matching fields down to their nested children if the child declares that field, keeping your YAML cleanly structured without excessive repetition.

For example, if the top-level Experiment configuration dictates `device: "cuda:0"`, any child config (such as the Dataset config or Trainer config) that explicitly requests a `device` property will automatically inherit `"cuda:0"` without you needing to duplicate it in the YAML.
