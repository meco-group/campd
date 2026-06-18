# Getting Started

This guide walks through the full pipeline end-to-end using the Franka Panda example.

## Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU
- **Windows is not natively supported.** Use Linux, macOS, or WSL2 on Windows.

## Installation

```bash
pip install campd
```

To enable Weights & Biases logging:

```bash
pip install wandb
wandb login
```

## Step 1 — Download training and test data

Datasets are hosted as Git LFS objects in the [campd-data](https://github.com/meco-group/campd-data) repository.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/meco-group/campd-data.git /tmp/campd-data
(cd /tmp/campd-data && git lfs pull)

# Training data
mkdir -p data/train/
tar -xzf /tmp/campd-data/train_data_campd_franka_spheres.tar.gz -C data/train/
tar -xzf /tmp/campd-data/train_data_campd_mpinets.tar.gz        -C data/train/

# Test data (small held-out set for quick inference)
mkdir -p data/test/
tar -xzf /tmp/campd-data/test_data_campd_franka_spheres.tar.gz  -C data/test/
tar -xzf /tmp/campd-data/test_data_campd_mpinets.tar.gz         -C data/test/

rm -rf /tmp/campd-data
```

Verify the download:

```bash
ls data/train/franka_spheres/     # should contain train.hdf5, val.hdf5
ls data/train/mpinets_curobo/     # should contain train.hdf5, val.hdf5
ls data/test/franka_spheres/      # should contain test.hdf5
ls data/test/mpinets_curobo/      # should contain test.hdf5
```

## Step 2 — Install example dependencies

```bash
cd examples/franka
pip install -r requirements.txt
```

## Step 3 — Train a model

```bash
campd-run configs/spheres/train.yaml
```

Training logs loss every batch to the console. If `WandBCallback` is configured, full training curves and periodic summaries are available in your W&B run. Checkpoints are saved under `results/`.

## Step 4 — Run inference

Open `examples/franka/configs/spheres/inference.yaml` and set `model_dir` to the checkpoint directory from your training run:

```yaml
experiment:
  model_dir: "results/franka_spheres_train/<timestamp>/1/checkpoints"
  # dataset_dir and hdf5_file already point to the test set
```

Then run:

```bash
campd-run configs/spheres/inference.yaml
```

Generated trajectories are saved as `.pt` files alongside per-sample `stats.yaml` files. A timing and stats summary is printed to the terminal when inference finishes.

```{note}
Domain-specific evaluation (collision checks, trajectory visualizations, success rate)
requires implementing a custom `Validator` for your project.
See [Extending the Framework](extending.md) for details.
```
