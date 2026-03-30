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

The `experiment-launcher` package provides robust configuration for single scripts, multi-processing clusters, or Slurm CPU/GPU queues. Below are the key configuration groups you can define under the `launcher:` block in your YAML files.

#### General Parameters

*   `exp_name` (**str**, *required*): The baseline name of the experiment.
*   `n_seeds` (**int**, *required*): Number of random seeds to run for each configuration.
*   `start_seed` (**int**, default: 0): The starting seed value.
*   `base_dir` (**str**, default: "./logs"): Base directory for saving experiment results.
*   `use_timestamp` (**bool**, default: true): Determines if a timestamp is appended to the experiment name. If true, the output directory name will include the date and time.
*   `compact_dirs` (**bool**, default: false): If true, creates subdirectories for parameter sweeps using only the swept value, rather than `parameter_name_value`. This keeps paths shorter.
*   `after_run_dir` (**str**, optional): Directory to copy the results into after the run has completed. Useful when running on a temporary scratch space.

#### Compute Resources (`resources`)

Supports parallelization both locally and on clusters, with CPU and memory configuration available for cluster environments.

*   `n_cores` (**int**, default: 1): Number of CPU cores requested per job.
*   `memory_per_core` (**int**, default: 2000): Memory (in MB) requested per core.
*   `n_exps_in_parallel` (**int**, default: 1): The number of independent experiments to execute concurrently on the machine or within a single SLURM job array task.
*   `manage_gpu_devices` (**bool**, default: false): If true, the launcher automatically assigns GPU devices to parallel experiments. Each experiment run receives a `device` parameter string (e.g., `'cuda:0'`).
*   `gpu_devices` (**list[int]**, optional): Explicit list of GPU IDs to use (e.g., `[0, 1, 2]`). If `manage_gpu_devices` is true but this is empty, it attempts to dynamically detect available GPUs via `torch.cuda.device_count()`.

#### SLURM Job Scheduling (`slurm`)

Configs specific to running experiments on a SLURM cluster.

*   `partition` (**str**, optional): The SLURM partition to submit the job to.
*   `gres` (**str**, optional): Generic resource scheduling (e.g., requesting GPUs like `gpu:a100:1`).
*   `constraint` (**str**, optional): Node features constraints.
*   `account` (**str**, optional): Account to charge resources to.
*   `cluster` (**str**, optional): Cluster name to submit the job to.
*   `project_name` (**str**, optional): Project identifier for SLURM tracking.
*   `begin` (**str**, optional): Start time for the job utilizing the `--begin` flag.

#### Job Duration (`duration`)

Sets the maximum runtime timeout for the SLURM job. Note: does not apply to local job execution.

*   `days` (**int**, default: 0)
*   `hours` (**int**, default: 24)
*   `minutes` (**int**, default: 0)
*   `seconds` (**int**, default: 0)

#### Environment Setup (`environment`)

Controls the conda environment and initial software modules via `module load` on a cluster.

*   `conda_env` (**str**, optional): Name of the Conda environment to activate.
*   `initial_module_load` (**list[str]**, optional): List of modules to load prior to execution (e.g., `["CUDA/12.1", "Python/3.10"]`).

#### Parameter Sweeping
Inside your YAML, you can sweep over an argument by using `sweep:` block at the root level, which `campd-run` handles by converting into `experiment-launcher` parameter sweeps (leveraging `experiment_launcher.Sweep`). This automatically generates Cartesian products of every configuration and separate directory paths.

```yaml
launcher:
  exp_name: "my_model_training"
  n_seeds: 3                  
  start_seed: 0
  base_dir: "results/"        
  use_timestamp: true         
  
  resources: 
    n_exps_in_parallel: 2
    n_cores: 4
    manage_gpu_devices: true
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
