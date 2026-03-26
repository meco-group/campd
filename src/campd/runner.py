

from campd.utils.config import merge_dict_b_in_a
from campd.utils.config import convert_dict_to_sweep_dict


def main():
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from campd.experiments.base import BaseExperiment
    from campd.utils.registry import Spec
    import argparse
    import sys
    import campd.all_imports
    import yaml
    from campd.utils.config import process_imports
    from experiment_launcher import Launcher

    parser = argparse.ArgumentParser(
        description="Run a campd experiment.")
    parser.add_argument("config_path", type=str,
                        help="Path to the YAML configuration file")

    args = parser.parse_args()

    # Add config directory to sys.path to allow importing local modules
    import os
    base_dir = os.path.dirname(os.path.abspath(args.config_path))
    if "config" == os.path.basename(base_dir):
        base_dir = os.path.dirname(base_dir)

    # Load config dictionary manually to handle imports and nesting
    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Process dependencies
    dependencies = cfg_dict.pop("dependencies", None)
    if dependencies is not None:
        cfg_dict["experiment"]["dependencies"] = {
            "base_dir": base_dir,
            "modules": dependencies
        }

    exp_cfg = cfg_dict.pop("experiment", None)
    wandb_cfg = cfg_dict.pop("wandb", None)

    # add sweep options
    sweep_config = cfg_dict.pop("sweep", None)
    if sweep_config is not None:
        sweep_config = convert_dict_to_sweep_dict(sweep_config)
        merge_dict_b_in_a(exp_cfg, sweep_config)

    launcher_config = cfg_dict["launcher"]
    launcher_config["exp_file"] = "campd.experiments.base"
    launcher = Launcher(launcher_config)
    launcher.add_experiment(cfg=exp_cfg, wandb=wandb_cfg)
    launcher.run(local=True)


if __name__ == "__main__":
    main()
