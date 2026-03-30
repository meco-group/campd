# YAML Configuration Reference

CAMPD relies on heavily nested YAML configurations. A complete experiment configuration is composed of 5 primary root nodes. By convention, utilizing YAML anchors (`&name` and `*name`) is heavily encouraged to prevent duplication.

## Root Configuration Nodes

### 1. `dependencies`
A list of module strings or relative paths that must be imported prior to parsing the rest of the configuration.
```yaml
dependencies:
  - "../src"                 # Recursively imports all .py files here
  - "my_custom_module"       # An installed pip package/module
```

### 2. `wandb`
Configure your exact Weights & Biases connection.
```yaml
wandb:
  mode: "online"             # online, offline, disabled
  entity: "my-team"
  project: "campd-trials"
  name: &run_name "run_1"    # Defining an anchor
```

### 3. `launcher`
Parameters supplied directly to `experiment-launcher`. For a full list of available options, see [Launching Experiments](launching.md#experiment-launcher-configuration).
```yaml
launcher:
  exp_name: *run_name        # Retrieving the anchor
  base_dir: "results/"
  n_seeds: 1
  resources:
    n_exps_in_parallel: 2
```

### 4. `sweep` (Optional)
Generates multiple configurations by sweeping hyperparameters via Cartesian product. Creates independent directory paths for every permutation.
```yaml
sweep:
  trainer:
    lr: [1e-4, 5e-4]
    max_epochs: [100, 200]
```

### 5. `experiment`
The actual execution blueprint provided to your runner. It must declare a `cls` corresponding to the `EXPERIMENTS` registry (e.g. `"train"` or `"inference"`).

## Example: Built-in `train` Structure
Below is an abridged mapping of standard fields available when `cls: "train"` is specified.

```yaml
experiment:
  cls: "train"
  seed: 42
  device: "cuda:0"

  dataset_dir: "data/train/sphere_world"
  train_file: "train.hdf5"
  
  dataset:
    trajectory_state: "pos"          # Format of trajectory (pos, pos+vel, etc)
    field_config:
      q_dim: 7                       # The robot joint configuration space
      trajectory_field: "solutions"  
      context_fields:                # Map HDF5 keys to context dicts
        cuboids: ["centers", "dims", "quaternions"]

  model:
    state_dim: 7
    model_type: "epsilon"            # epsilon, sample, v_prediction
    n_diffusion_steps: 25
    network:                         # Define the reverse denoiser
      cls: "TemporalUnet"            # From REVERSE_NETS registry
      config: { hidden_dim: 256 }
    context_network:                 # Define the context encoder
      cls: "ContextEncoder"          # From CONTEXT_NETS registry
      config: { ... }

  trainer:
    max_epochs: 200
    optimizer:
      cls: "torch.optim.AdamW"
      init: { lr: 1e-4 }
    objective:
      cls: "DiffusionObjective"
      config: { ... }
    callbacks:
      - cls: "CheckpointCallback"
        init: { save_best: true }
```
