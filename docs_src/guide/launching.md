# Launching Experiments

Experiments in CAMPD can be launched either rapidly via a Command Line Interface (CLI) tailored for massive hyperparameter sweeps and clustering, or through your own custom python scripts.

## Via the `campd-run` CLI (Recommended)

When you install CAMPD, the `campd-run` console script is installed automatically. It is the primary way to execute declarative YAML configurations.

```bash
campd-run configs/train.yaml
```

Under the hood, `campd-run` performs several setup tasks:
1. **Parses the YAML file**, separating global launch directives from internal component configurations.
2. **Injects Dependencies**: Reads the `dependencies` key and recursively imports any `.py` modules listed. This critically executes your `@REGISTRY.register` decorators *before* the experiment config attempts to load them.
3. **Initializes the Launcher**: Delegates execution to the [experiment-launcher](https://github.com/robot-learning-group/experiment-launcher) package.
4. **Executes**: Looks up the `experiment.cls` string in the `EXPERIMENTS` registry, instantiates it, and invokes its `run()` method.

### Experiment Launcher Configuration

The `experiment-launcher` package provides robust configuration for single scripts, multi-processing clusters, or Slurm CPU/GPU queues.

```yaml
launcher:
  exp_name: "my_model_training"
  n_seeds: 3                  # Run the same experiment 3 times with different random seeds
  start_seed: 0
  base_dir: "results/"        # Output base directory
  use_timestamp: true         # Append _YYYYMMDD_HHMMSS to your experiment dir
  resources: 
    n_exps_in_parallel: 2     # How many jobs to run concurrently on this machine
```

## Via Custom Python Script

For simpler execution, debugging through your IDE, or embedding CAMPD inside another project, you can bypass the launcher CLI.

```python
import os
from campd.experiments import TrainExperiment

# Manually load custom components that aren't built-in!
import my_custom_modules.callbacks

base_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(base_dir, "configs/train.yaml")

# Load configuration and spin up the experiment runner
exp = TrainExperiment.from_yaml(yaml_path)
exp.run()
```

> **Warning:** When strictly using `from_yaml()` on an Experiment class, the `dependencies` array within your YAML is **not dynamically imported**. You must assume responsibility for explicitly `import`ing custom scripts in your python file before instantiating the classes.
