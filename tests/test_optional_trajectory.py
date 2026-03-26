import h5py
import numpy as np
import os
import tempfile
import pytest
from campd.data.trajectory_dataset import TrajectoryDataset, TrajectoryDatasetCfg, HDF5FieldCfg


class SimpleDataset(TrajectoryDataset):
    """Concrete implementation for testing."""

    def _update_metadata(self):
        pass


def create_dummy_hdf5(path, n_trajs=10, horizon=20, state_dim=7):
    with h5py.File(path, 'w') as f:
        # Create trajectories [N, T, D]
        trajs = np.random.randn(n_trajs, horizon, state_dim).astype(np.float32)
        f.create_dataset('trajectories', data=trajs)

        # Create some context
        boxes = np.random.randn(n_trajs, 5, 7).astype(np.float32)
        f.create_dataset('boxes', data=boxes)

        return trajs, boxes


def test_load_all_trajectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, 'data.hdf5')
        trajs_gt, boxes_gt = create_dummy_hdf5(hdf5_path)

        cfg = TrajectoryDatasetCfg(
            dataset_dir=tmpdir,
            hdf5_file='data.hdf5',
            load_only_context=False
        )

        dataset = SimpleDataset(cfg)

        assert dataset.n_support_points == 20

        # Denormalize to check values
        sample = dataset[0]
        sample_denorm = sample.denormalize(dataset.normalizer)

        assert sample_denorm.trajectory.shape == (20, 7)
        assert np.allclose(sample_denorm.trajectory.numpy(),
                           trajs_gt[0], atol=1e-5)


def test_load_only_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, 'data.hdf5')
        trajs_gt, boxes_gt = create_dummy_hdf5(hdf5_path, horizon=20)

        cfg = TrajectoryDatasetCfg(
            dataset_dir=tmpdir,
            hdf5_file='data.hdf5',
            load_only_context=True,
            field_config=HDF5FieldCfg(context_fields={'boxes': 'boxes'})
        )

        dataset = SimpleDataset(cfg)

        # Check metadata
        assert dataset.n_support_points == 2

        # Check trajectory shape (normalized)
        traj = dataset[0].trajectory
        assert traj.shape == (2, 7)

        # Check contents (start and goal) - denormalize first
        sample = dataset[0]
        sample_denorm = sample.denormalize(dataset.normalizer)
        traj_denorm = sample_denorm.trajectory

        assert np.allclose(traj_denorm[0].numpy(), trajs_gt[0, 0], atol=1e-5)
        assert np.allclose(traj_denorm[1].numpy(), trajs_gt[0, -1], atol=1e-5)

        # Check context
        assert dataset[0].context is not None

        # Start/Goal in context should match trajectory start/goal
        # Context is also normalized in the dataset, so we need to check denormalized context
        ctx_denorm = sample_denorm.context

        assert ctx_denorm.start.shape == (7,)
        assert np.allclose(ctx_denorm.start.numpy(), trajs_gt[0, 0], atol=1e-5)
        assert np.allclose(ctx_denorm.goal.numpy(), trajs_gt[0, -1], atol=1e-5)


def test_load_only_context_error_short_traj():
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, 'data.hdf5')
        # Create trajectories with only 1 point
        create_dummy_hdf5(hdf5_path, horizon=1)

        cfg = TrajectoryDatasetCfg(
            dataset_dir=tmpdir,
            hdf5_file='data.hdf5',
            load_only_context=True
        )

        history_match = "at least 2 are required"
        with pytest.raises(ValueError, match=history_match):
            SimpleDataset(cfg)


if __name__ == "__main__":
    test_load_all_trajectories()
    test_load_only_context()
    test_load_only_context_error_short_traj()
    print("All tests passed!")
