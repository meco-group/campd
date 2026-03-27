# either launch via campd-run <path/to/config.yaml>
# or run this script directly

RESULTS_BASE_DIR = "results/train_via_launch_file_"
CONFIG_FILE = "configs/spheres/train.yaml"

if __name__ == "__main__":
    import sys
    import os
    # add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

    # import modules so that registries are populated
    import training_summary
    import inference_validator
    from campd.experiments import TrainExperiment
    import yaml
    from datetime import datetime

    results_dir = RESULTS_BASE_DIR + datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, CONFIG_FILE), 'r') as f:
        cfg_dict = yaml.safe_load(f)["experiment"]
        # set results dir
        cfg_dict["results_dir"] = os.path.join(base_dir, results_dir)

    exp = TrainExperiment(cfg_dict)
    exp.run()
