
import pytest
import unittest.mock as mock
import yaml
import os
import torch
from torch.utils.data import DataLoader

from campd.experiments.train import TrainExperiment, TrainExperimentCfg
from campd.utils.registry import Spec

# Mock classes to simulate dependencies


class MockDataset:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        # Add necessary attributes for random_split
        self.n_trajs = 100
        self.normalizer = mock.MagicMock()
        self.normalizer.export_config.return_value = {}

    def random_split(self, val_set_size, save=None):
        # Return dummy indices
        return list(range(90)), list(range(90, 100))

    def get_dataloader(self, indices=None, **kwargs):
        # Return a dummy dataloader
        return mock.MagicMock(spec=DataLoader, __len__=lambda x: 10)

    def build(self):
        return self


class MockModel:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def build(self):
        return self

    def save_config(self, save_dir):
        pass


class MockTrainer:
    def __init__(self, *args, model=None, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.fit_called = False
        self.train_loader = None
        self.val_loader = None

    def fit(self, train_loader, val_loader):
        self.fit_called = True
        self.train_loader = train_loader
        self.val_loader = val_loader

    def build(self, model=None):
        return MockTrainer(model=model, **self.kwargs)


@pytest.fixture
def mock_experiment_config(tmp_path):
    # Create a dummy config file
    config_data = {
        "results_dir": str(tmp_path / "results"),
        "train_file": str(tmp_path / "dataset" / "train_data.h5"),
        "dataset_dir": str(tmp_path / "dataset"),
        "dataset": {
            "dataset_dir": str(tmp_path / "dataset"),
            "dataset_dir": "subdir",
        },
        "model": {
            "state_dim": 3,
            "network": {"cls": "campd.models.MockNetwork", "init": {}},
            "normalizer": {"cls": "campd.data.normalization.DatasetNormalizer", "init": {}},
        },
        "trainer": {
            "tensor_args": {"device": "cpu", "dtype": "float32"},
            "objective": {"cls": "campd.training.objectives.MockObjective", "init": {}}
        },
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2
        },
        "val_set_size": 0.1,
        "seed": 12345,
        "cls": "dummy_cls"
    }

    config_file = tmp_path / "experiment.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return str(config_file)


def test_config_loading(mock_experiment_config):
    experiment = TrainExperiment.from_yaml(mock_experiment_config)
    assert isinstance(experiment.cfg, TrainExperimentCfg)
    assert experiment.cfg.seed == 12345
    assert experiment.cfg.val_set_size == 0.1
    assert experiment.cfg.dataloader["batch_size"] == 16


@mock.patch("campd.experiments.train.Trainer")
@mock.patch("campd.experiments.train.ContextTrajectoryDiffusionModel")
@mock.patch("campd.experiments.train.TrajectoryDataset")
@mock.patch("campd.utils.registry.import_string")
def test_experiment_run(mock_import_string, mock_dataset_cls, mock_model_cls, mock_trainer_cls, mock_experiment_config):

    # Setup mocks
    mock_dataset_cls.side_effect = MockDataset
    mock_model_cls.side_effect = MockModel
    mock_trainer_cls.side_effect = MockTrainer

    def import_side_effect(path):
        if "MockDataset" in path:
            return MockDataset
        if "MockModel" in path:
            return MockModel
        if "MockTrainer" in path:
            return MockTrainer
        raise ImportError(f"Unknown path {path}")

    mock_import_string.side_effect = import_side_effect

    experiment = TrainExperiment.from_yaml(mock_experiment_config)

    # Run the experiment
    experiment.run()

    # Checks
    assert experiment.dataset is not None
    assert isinstance(experiment.dataset, MockDataset)

    assert experiment.train_loader is not None
    assert experiment.val_loader is not None

    assert experiment.model is not None
    assert isinstance(experiment.model, MockModel)

    assert experiment.trainer is not None
    assert isinstance(experiment.trainer, MockTrainer)

    # Check if trainer.fit was called
    assert experiment.trainer.fit_called
    assert experiment.trainer.train_loader == experiment.train_loader
    assert experiment.trainer.val_loader == experiment.val_loader


def test_missing_init_order(mock_experiment_config):
    experiment = TrainExperiment.from_yaml(mock_experiment_config)

    # Calling init_trainer without model should fail
    with pytest.raises(RuntimeError, match="Model not initialized"):
        experiment.init_trainer()

    with pytest.raises(RuntimeError, match="Dataset not initialized"):
        experiment.get_dataloaders()


def test_train_with_val_file(tmp_path):
    # Create valid dummy config
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_file = dataset_dir / "train.h5"
    val_file = dataset_dir / "val.h5"

    # Create dummy files (empty is fine for this test as we mock dataset)
    train_file.touch()
    val_file.touch()

    config_data = {
        "results_dir": str(tmp_path / "results"),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "dataset_dir": str(dataset_dir),
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "hdf5_file": "train.h5"  # explicit override or base
        },
        "model": {
            "state_dim": 3,
            "network": {"cls": "campd.models.MockNetwork", "init": {}},
            "normalizer": {"cls": "campd.data.normalization.DatasetNormalizer", "init": {}},
        },
        "trainer": {
            "tensor_args": {"device": "cpu", "dtype": "float32"},
            "objective": {"cls": "campd.training.objectives.MockObjective", "init": {}}
        },
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2
        },
        "seed": 12345,
        "cls": "dummy_cls"
    }

    config_file = tmp_path / "experiment_val.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mocking
    with mock.patch("campd.experiments.train.Trainer") as mock_trainer, \
            mock.patch("campd.experiments.train.ContextTrajectoryDiffusionModel") as mock_model, \
            mock.patch("campd.experiments.train.TrajectoryDataset") as mock_dataset_cls:

        mock_dataset_instance = MockDataset()
        mock_dataset_cls.side_effect = [
            mock_dataset_instance, mock_dataset_instance]  # One for train, one for val

        experiment = TrainExperiment.from_yaml(str(config_file))

        # Init dataset
        experiment.init_dataset()

        assert experiment.dataset is not None
        assert experiment.dataset_val is not None
        assert mock_dataset_cls.call_count == 2

        # Verify call args
        call_args_list = mock_dataset_cls.call_args_list
        # First call (train)
        assert call_args_list[0][0][0].hdf5_file == "train.h5"
        # Second call (val)
        assert call_args_list[1][0][0].hdf5_file == "val.h5"

        # Get dataloaders
        experiment.get_dataloaders()
        assert experiment.train_loader is not None
        assert experiment.val_loader is not None
