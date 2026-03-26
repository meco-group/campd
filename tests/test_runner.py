
import pytest
from unittest.mock import MagicMock, patch
import sys
from campd.runner import main
from campd.experiments.base import BaseExperiment


def test_runner_train_experiment():
    # Mock the experiment class and its methods
    mock_experiment_cls = MagicMock()
    mock_experiment_instance = MagicMock()
    mock_experiment_cls.from_yaml.return_value = mock_experiment_instance
    mock_experiment_cls.from_config.return_value = mock_experiment_instance
    mock_experiment_cls.__name__ = "TrainExperiment"  # Necessary for getattr/mocking

    # We need to make sure issubclass works, so we set the spec/bases
    # But effectively, we can just patch 'campd.experiments.TrainExperiment'

    from campd.experiments.registry import EXPERIMENTS
    # Patch open to return a dummy yaml string
    mock_open = MagicMock()
    # Patcher for open context manager
    mock_open.return_value.__enter__.return_value = "dummy content"

    mock_yaml_load = MagicMock(return_value={"imports": [], "experiment": {
                               "cls": "TrainExperiment"}, "launcher": {"exp_name": "test", "n_seeds": 1}})

    # Patch Launcher to avoid FS ops
    with patch("campd.experiments.TrainExperiment", mock_experiment_cls), \
            patch("sys.argv", ["campd-run", "config.yaml"]), \
            patch("campd.runner.issubclass") as mock_issubclass, \
            patch("experiment_launcher.Launcher") as mock_launcher_cls, \
            patch.dict(EXPERIMENTS._map, {"TrainExperiment": mock_experiment_cls}), \
            patch("builtins.open", mock_open), \
            patch("yaml.safe_load", mock_yaml_load), \
            patch("campd.utils.config.process_imports") as mock_process_imports:

        mock_launcher_instance = MagicMock()
        mock_launcher_cls.return_value = mock_launcher_instance

        mock_issubclass.return_value = True

        main()

        # runner now manually loads yaml and calls from_config
        # assert files opened?
        # assert from_config called
        mock_launcher_instance.add_experiment.assert_called_once()
        mock_launcher_instance.run.assert_called_once_with(local=True)


def test_runner_invalid_api():
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value = "dummy"
    mock_yaml_load = MagicMock(return_value={"launcher": {
                               "exp_name": "test", "n_seeds": 1}, "experiment": {"cls": "NonExistentExperiment"}})

    with patch("sys.argv", ["campd-run", "NonExistentExperiment", "config.yaml"]), \
            patch("builtins.print") as mock_print, \
            patch("builtins.open", mock_open), \
            patch("yaml.safe_load", mock_yaml_load), \
            patch("experiment_launcher.Launcher") as mock_launcher_cls, \
            patch("campd.utils.config.process_imports"), \
            pytest.raises(SystemExit) as pytest_wrapped_e:

        main()

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2
    # Check that some error was printed
    assert pytest_wrapped_e.value.code == 2
