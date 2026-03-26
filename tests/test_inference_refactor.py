import unittest
from unittest.mock import MagicMock, patch
import torch
import os
import shutil

from campd.experiments.inference import InferenceExperiment, InferenceExperimentCfg
from campd.data.trajectory_dataset import TrajectoryDatasetCfg, TrajectoryDataset
from campd.models.diffusion.model import SamplingCfg, ContextTrajectoryDiffusionModel
from campd.experiments.validators import Validator
from campd.utils.registry import Spec
from campd.data.trajectory_sample import TrajectorySample
from campd.data.context import TrajectoryContext


class MockValidator(Validator):
    def validate(self, batch, output_dir):
        print("MockValidator.validate called")
        return {}


class TestInferenceExperimentRefactor(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tmp_test_inference_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch('campd.experiments.inference.ContextTrajectoryDiffusionModel')
    @patch('campd.experiments.inference.TrajectoryDataset')
    @patch('campd.models.diffusion.model.load_params_from_yaml')
    def test_run_inference(self, mock_load_yaml, mock_dataset_cls, mock_model_cls):
        # Mock model loading
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        # Mock dataloader
        # Create a dummy batch
        dummy_context = MagicMock(spec=TrajectoryContext)
        dummy_context.batch_size = 2
        dummy_context.is_batched = True  # Added is_batched
        dummy_context.to.return_value = dummy_context

        dummy_trajectory = torch.randn(2, 10, 7)
        dummy_batch = TrajectorySample(
            trajectory=dummy_trajectory,
            context=dummy_context,
            is_normalized=True,
            is_batched=True
        )
        dummy_batch.to = MagicMock(return_value=dummy_batch)

        # samples.slice(i) needs to return something
        def slice_side_effect(i):
            return torch.randn(10, 7)
        dummy_batch.slice = MagicMock(side_effect=slice_side_effect)

        mock_dataset.__iter__.return_value = [dummy_batch]

        # Mock inference output
        def inference_side_effect(sampling_cfg):
            # Return samples, t_gen
            return dummy_batch, 1.0

        # Mock experiment instance to override inference if needed, but we can mock model.sample or inference
        # Here we mock model.sample implicitly via inference call, or we can mock inference method if we subclass
        # But we are testing the class, so we should let it run.
        # We need to mock self.inference or model.sample.
        # Since we mocked model, model.sample is a mock.

        # The inference method calls:
        # self.model.prepare_for_sampling(sampling_cfg)
        # output = self.model.sample(context=self.context)

        # Set return value of model.sample
        mock_model.sample.return_value = dummy_batch

        # Config
        dataset_cfg = TrajectoryDatasetCfg(
            dataset_dir="dummy_dataset", hdf5_file="data.hdf5")
        # Using local class name, might need full path or register it
        validator_cfg = Spec(cls="tests.test_inference_refactor:MockValidator")

        # Register MockValidator locally if needed, or just use object
        # The validator spec build uses import_string or registry.
        # Let's mock validator config to return a pre-built object or Mock

        cfg = InferenceExperimentCfg(
            cls="inference",
            model_dir="dummy_model_dir",
            results_dir=self.output_dir,
            dataset=dataset_cfg,
            validator=None,
            sampling_cfg=SamplingCfg(n_support_points=10),
            save_samples=False
        )

        # Create experiment
        exp = InferenceExperiment(cfg)

        # Inject validator mock
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {}
        exp.validator = mock_validator

        # Run
        exp.run()

        # Verify
        self.assertEqual(mock_dataset.__iter__.call_count, 1)
        self.assertEqual(mock_model.prepare_for_sampling.call_count, 1)
        self.assertEqual(mock_model.sample.call_count, 6)
        mock_validator.validate.assert_called_with(
            dummy_batch, os.path.join(self.output_dir, 'data', '000000'))

        # Check if files saved
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, "data", "000000", "stats.yaml")))


if __name__ == '__main__':
    unittest.main()
