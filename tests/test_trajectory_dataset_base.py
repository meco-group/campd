
import pytest
import torch
import numpy as np
import h5py
import os
from campd.data.trajectory_dataset import TrajectoryDataset, HDF5FieldCfg, TrajectoryDatasetCfg
from campd.data.trajectory_sample import TrajectorySample

# Concrete implementation for testing


class MockDataset(TrajectoryDataset):
    def _update_metadata(self):
        self.metadata['custom_key'] = 'custom_value'


@pytest.fixture
def dataset_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def hdf5_file(dataset_dir):
    # Create temp HDF5 file in the temp directory
    filepath = os.path.join(dataset_dir, 'data.hdf5')
    with h5py.File(filepath, 'w') as f:
        # Trajectories: 10 trajs, 20 steps, 7 dim
        f.create_dataset('trajectories', data=np.random.randn(10, 20, 7))

        # Context: cuboids
        f.create_dataset('cuboid_centers', data=np.random.randn(10, 5, 3))
        f.create_dataset('cuboid_dims', data=np.random.randn(10, 5, 3))

        # Attributes
        f.attrs['attr1'] = 123

        # Extra data
        f.create_dataset('extra_info', data=np.random.randn(10, 1))

    return filepath


def test_dataset_initialization(dataset_dir, hdf5_file):
    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset = MockDataset(config=cfg)

    assert len(dataset) == 10
    assert dataset.n_support_points == 20
    assert dataset.q_dim == 7
    assert dataset.traj_dim == 7
    assert dataset.metadata['attr1'] == 123
    assert dataset.metadata['custom_key'] == 'custom_value'

    # Check extra data
    assert 'extra_info' in dataset.extra_data
    assert len(dataset.extra_data['extra_info']) == 10


def test_dataset_context_loading(dataset_dir, hdf5_file):
    # Config to concatenate centers and dims into 'cuboids'
    field_config = HDF5FieldCfg(
        context_fields={
            'cuboids': ['cuboid_centers', 'cuboid_dims']
        }
    )

    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        field_config=field_config,
    )
    dataset = MockDataset(config=cfg)

    # Concatenated dimension: 3+3=6
    assert 'cuboids' in dataset.context.keys()
    assert dataset.context['cuboids'].shape == (10, 5, 6)

    # Check components
    assert dataset.context.components['cuboid_centers'] == ('cuboids', 0, 3)
    assert dataset.context.components['cuboid_dims'] == ('cuboids', 3, 6)


def test_dataset_getter(dataset_dir, hdf5_file):
    # We need to make sure we don't try to load 'normalizer.pkl' if we re-use directory,
    # or rely on load_all_trajectories=True to create new one.
    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset = MockDataset(config=cfg)

    sample = dataset[0]

    # Trajectory should be normalized (mock normalization usually roughly [-1, 1], but data is randn)
    # Just check shapes and types
    assert isinstance(sample.trajectory, torch.Tensor)
    assert sample.trajectory.shape == (20, 7)
    assert sample.is_normalized

    # Let's re-init with explicit config for robustness of this test
    field_cfg = HDF5FieldCfg(context_fields={'cuboids': ['cuboid_centers']})
    cfg2 = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        field_config=field_cfg,
    )
    dataset = MockDataset(config=cfg2)
    sample = dataset[0]
    assert sample.context is not None
    assert 'cuboids' in sample.context.keys()
    assert sample.context['cuboids'].shape == (5, 3)  # ONLY centers


def test_trajectory_state_modes(dataset_dir, hdf5_file):
    # Test pos+vel (requires 2x dim in file, currently file has 7 dim, q_dim will be detected as 7)
    # If file has 7 dims, and we ask for pos+vel, it should fail if q_dim is 7.
    # We need a file with 14 dims to test pos+vel if q_dim=7.

    with h5py.File(hdf5_file, 'r+') as f:
        del f['trajectories']
        f.create_dataset(
            'trajectories', data=np.random.randn(10, 20, 14))  # 14 dims

    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        trajectory_state='pos+vel',
        field_config=HDF5FieldCfg(q_dim=7),
    )
    dataset = MockDataset(config=cfg)

    assert dataset.traj_dim == 14
    assert dataset.has_velocity
    assert not dataset.has_acceleration


def test_random_split(dataset_dir, hdf5_file):
    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset = MockDataset(config=cfg)

    train, val = dataset.random_split(val_set_size=0.2)

    # 10 trajectories total. 0.2 * 10 = 2 val, 8 train.
    assert len(train) == 8
    assert len(val) == 2


def test_normalization_consistency(dataset_dir, hdf5_file):
    # Train run: load all

    cfg_train = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset_train = MockDataset(config=cfg_train)

    # Eval run: load_all=False, use saved normalizer (simulated by using same logic or manual transfer)
    # Since we removed save_normalizer, we just want to ensure consistent normalization logic
    # or manual save/load if that were supported outside.
    # For now, let's just check that two datasets initialized same way produce same normalizer

    cfg_eval = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset_eval = MockDataset(config=cfg_eval)

    # Check limits match
    k = 'traj'
    m1 = dataset_train.normalizer.normalizers[k].mins
    m2 = dataset_eval.normalizer.normalizers[k].mins
    assert torch.allclose(m1, m2)


def test_get_dataloader(dataset_dir, hdf5_file):
    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
    )
    dataset = MockDataset(config=cfg)

    # 1. Test random_split (verify it returns generic indices that work)
    train_indices, val_indices = dataset.random_split(val_set_size=0.2)
    assert len(train_indices) == 8
    assert len(val_indices) == 2

    # 2. Test get_dataloader with indices (Train set)
    train_loader = dataset.get_dataloader(
        indices=train_indices,
        batch_size=4,
        shuffle=True
    )
    assert len(train_loader) == 2  # 8 samples / 4 batch_size = 2 batches

    # Verify iteration produces TrajectorySample objects
    for batch in train_loader:
        assert isinstance(batch, TrajectorySample)
        assert batch.trajectory.shape[0] == 4
        assert batch.is_normalized

    # 3. Test get_dataloader without indices (Full set)
    full_loader = dataset.get_dataloader(batch_size=2)
    assert len(full_loader) == 5  # 10 samples / 2 batch_size = 5 batches

    assert isinstance(batch, TrajectorySample)


def test_field_config_save_load(dataset_dir, hdf5_file):
    # 1. Initialize dataset with specific field config
    custom_field_cfg = HDF5FieldCfg(trajectory_field='custom_traj_field')

    # We need to ensure the hdf5 file actually has this field for the dataset to load successfully
    with h5py.File(hdf5_file, 'r+') as f:
        if 'custom_traj_field' not in f:
            f['custom_traj_field'] = f['trajectories'][:]  # Copy data

    cfg = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        field_config=custom_field_cfg
    )
    dataset = MockDataset(config=cfg)

    # 2. Save field config
    dataset.save_field_config()

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    assert os.path.exists(yaml_path)

    # 3. Initialize new dataset WITHOUT field_config
    # It should autoload from the yaml file
    cfg_autoload = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        # field_config is None by default now
    )
    dataset_autoload = MockDataset(config=cfg_autoload)

    # Check if loaded config matches
    assert dataset_autoload.field_config.trajectory_field == 'custom_traj_field'

    # 4. Initialize dataset WITH override field_config
    # Explicit config should take precedence over yaml
    override_cfg = HDF5FieldCfg(
        trajectory_field='trajectories')  # Back to default
    cfg_override = TrajectoryDatasetCfg(
        dataset_dir=dataset_dir,
        hdf5_file=os.path.basename(hdf5_file),
        field_config=override_cfg
    )
    dataset_override = MockDataset(config=cfg_override)

    assert dataset_override.field_config.trajectory_field == 'trajectories'
