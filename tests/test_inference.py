import pytest
import torch
from unittest.mock import MagicMock, patch

from campd.experiments.inference import InferenceExperiment, InferenceExperimentCfg
from campd.data.trajectory_dataset import TrajectoryDatasetCfg
from campd.data.normalization import DatasetNormalizer
from campd.data.trajectory_sample import TrajectorySample
from campd.models.diffusion.model import SamplingCfg, ContextTrajectoryDiffusionModel


@pytest.fixture
def mock_model_dir(tmp_path):
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir()
    # Create valid args.yaml so file checks pass if needed
    (model_dir / "args.yaml").touch()
    return str(model_dir)


@patch('campd.experiments.inference.TrajectoryDataset')
@patch('campd.experiments.inference.ContextTrajectoryDiffusionModel.from_pretrained')
def test_inference_experiment_init(mock_from_pretrained, mock_dataset_cls, mock_model_dir):
    # Setup mock return
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    # Init
    cfg = InferenceExperimentCfg(
        model_dir=mock_model_dir,
        results_dir=str(mock_model_dir),
        device='cpu',
        cls='inference',
        dataset=TrajectoryDatasetCfg(dataset_dir=str(
            mock_model_dir), hdf5_file="data.hdf5")
    )
    inference_experiment = InferenceExperiment(cfg=cfg)

    # Verify from_pretrained called
    # Note: validate_call on from_pretrained fills in default freeze_params=True
    # and results in positional model_dir
    mock_from_pretrained.assert_called_once_with(
        mock_model_dir,
        device='cpu',
        model_iter=None,
        freeze_params=True
    )

    assert inference_experiment.model == mock_model


@patch('campd.experiments.inference.TrajectoryDataset')
@patch('campd.experiments.inference.ContextTrajectoryDiffusionModel.from_pretrained')
def test_inference_sample(mock_from_pretrained, mock_dataset_cls, mock_model_dir):
    # Setup mock
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    # Mock sample return
    mock_model.sample.return_value = TrajectorySample(
        trajectory=torch.randn(1, 10, 3),
        context=None,
        is_normalized=True,
        is_batched=True
    )

    cfg = InferenceExperimentCfg(
        model_dir=mock_model_dir,
        results_dir=str(mock_model_dir),
        device='cpu',
        cls='inference',
        dataset=TrajectoryDatasetCfg(dataset_dir=str(
            mock_model_dir), hdf5_file="data.hdf5")
    )
    inference_experiment = InferenceExperiment(cfg=cfg)

    # Provide a minimal normalized context
    inference_experiment.context_normalized = MagicMock()

    # Run inference
    sampling_cfg = SamplingCfg(
        n_support_points=1, batch_size=1, return_chain=False)
    inference_experiment.cfg.sampling_cfg = sampling_cfg
    inference_experiment.run()

    mock_model.prepare_for_sampling.assert_called()
    mock_model.sample.assert_called()
