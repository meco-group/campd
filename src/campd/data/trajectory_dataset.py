from torch.utils.data import DataLoader
import abc
import os
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import yaml
import torch
import h5py
from torch.utils.data import Dataset

from campd.data.normalization import DatasetNormalizer, NormalizationCfg
from campd.data.context import TrajectoryContext
from campd.data.trajectory_sample import TrajectorySample
from campd.utils.torch import to_torch, TensorArgs


from pydantic import BaseModel, PrivateAttr, Field, field_validator, ConfigDict, validate_call


class HDF5FieldCfg(BaseModel):
    """
    Configuration for HDF5 field names.

    Allows customization of which HDF5 fields to load for trajectories and context.
    Context fields can be specified as single field names or lists of fields to concatenate.

    Args:
        trajectory_field: HDF5 field name for trajectories (default: 'trajectories')
        q_dim: Configuration space dimension (e.g., 7 for Panda robot)
        context_fields: Dictionary mapping context type names to HDF5 field name(s)
                       Can be a single string or list of strings to concatenate
                       Example::

                           {
                               'cuboids': ['cuboid_centers', 'cuboid_dims', 'cuboid_quats'],
                               'spheres': 'sphere_data'
                           }

                       If None, will auto-detect common field names

    Example::

        # Use default field names
        config = HDF5FieldCfg()

        # Custom field names with concatenation
        config = HDF5FieldCfg(
            trajectory_field='solutions',
            context_fields={
                'cuboids': ['cuboid_centers', 'cuboid_dims', 'cuboid_quats'],
                'spheres': ['sphere_centers', 'sphere_radii']
            }
        )
    """
    trajectory_field: str = 'trajectories'
    q_dim: Optional[int] = None
    context_fields: Optional[Dict[str, Any]] = None  # Can be str or List[str]

    # Common context field patterns to auto-detect (with concatenation support)
    _common_context_patterns: Dict[str, List[List[str]]] = PrivateAttr(default_factory=lambda: {
        'cuboids': [
            ['cuboid_centers', 'cuboid_dims', 'cuboid_quats'],
            ['cuboid_centers', 'cuboid_dims', 'cuboid_quaternions'],
            ['cuboids'],
            ['boxes']
        ],
        'spheres': [
            ['sphere_centers', 'sphere_radii'],
            ['spheres'],
            ['sphere_data']
        ],
        'cylinders': [
            ['cylinder_centers', 'cylinder_radii',
                'cylinder_heights', 'cylinder_quats'],
            ['cylinder_centers', 'cylinder_radii',
                'cylinder_heights', 'cylinder_quaternions'],
            ['cylinders'],
            ['cylinder_data']
        ],
        'obstacles': [['obstacle_data']]
    })

    def get_context_fields(self, available_fields: List[str]) -> Dict[str, List[str]]:
        """
        Get context field mappings, auto-detecting if not specified.

        Args:
            available_fields: List of available HDF5 field names

        Returns:
            Dictionary mapping context type to list of HDF5 field names to concatenate
        """
        if self.context_fields is not None:
            # Normalize to list format
            normalized = {}
            for context_type, fields in self.context_fields.items():
                if isinstance(fields, str):
                    normalized[context_type] = [fields]
                else:
                    normalized[context_type] = fields
            return normalized

        # Auto-detect context fields
        detected = {}
        for context_type, pattern_groups in self._common_context_patterns.items():
            for pattern_group in pattern_groups:
                # Check if all fields in this pattern group are available
                if all(field in available_fields for field in pattern_group):
                    detected[context_type] = pattern_group
                    break

        return detected


class TrajectoryDatasetCfg(BaseModel):
    """
    Configuration for TrajectoryDataset.

    Args:
        dataset_dir: Path to dataset directory containing HDF5 file(s)
        hdf5_file: Name of HDF5 file (default: 'data.hdf5')
        field_config: HDF5FieldCfg for custom field names
        trajectory_state: Trajectory state to use ('pos', 'pos+vel', or 'pos+vel+acc')
        normalization_config: Optional NormalizationCfg with explicit limits
        save_normalizer: Whether to save normalizer state to disk
        tensor_args: Dictionary with 'device' and 'dtype' for tensors
        storage_device: Device to store loaded data on (e.g., 'cpu' or 'cuda')
    """
    dataset_dir: str
    hdf5_file: str = 'data.hdf5'
    field_config: Optional[HDF5FieldCfg] = None
    trajectory_state: str = 'pos'
    normalization_config: Optional[NormalizationCfg] = None
    tensor_args: TensorArgs = Field(default_factory=TensorArgs)
    storage_device: str = 'cpu'
    load_only_context: bool = False

    @field_validator('trajectory_state')
    @classmethod
    def validate_trajectory_state(cls, v):
        valid_states = ['pos', 'pos+vel', 'pos+vel+acc']
        if v not in valid_states:
            raise ValueError(
                f"trajectory_state must be one of {valid_states}, got '{v}'"
            )
        return v


class TrajectoryDataset(Dataset, abc.ABC):
    """
    Abstract base class for trajectory datasets loaded from HDF5 files.

    This class provides a standardized interface for loading trajectory and context data
    from HDF5 files with configurable field names and normalization limits.

    HDF5 File Structure:
        Expected structure (with default field names):
        - 'trajectories': [n_trajectories, n_support_points, state_dim]
        - 'boxes': [n_trajectories, n_boxes, box_dim] (optional)
        - 'spheres': [n_trajectories, n_spheres, sphere_dim] (optional)
        - 'cylinders': [n_trajectories, n_cylinders, cylinder_dim] (optional)
        - ... other context types

    Args:
        dataset_dir: Path to dataset directory containing HDF5 file(s)
        hdf5_file: Name of HDF5 file (default: 'data.hdf5')
        field_config: HDF5FieldCfg for custom field names
        trajectory_state: Trajectory state to use ('pos', 'pos+vel', or 'pos+vel+acc')
        normalization_config: Optional NormalizationCfg with explicit limits
        load_all_trajectories: If True, load all data at once
        tensor_args: Dictionary with 'device' and 'dtype' for tensors

    Example::

        # Basic usage with defaults
        dataset = MyDataset(dataset_dir='path/to/data')

        # Custom field names
        field_config = HDF5FieldCfg(
            trajectory_field='solutions',
            context_fields={'boxes': 'cuboid_data'}
        )
        dataset = MyDataset(
            dataset_dir='path/to/data',
            field_config=field_config
        )

        # With custom normalization limits
        norm_config = NormalizationCfg(field_limits={
            'traj': {'mins': [...], 'maxs': [...]},
            'context_boxes': {'mins': [...], 'maxs': [...]}
        })
        dataset = MyDataset(
            dataset_dir='path/to/data',
            normalization_config=norm_config
        )
    """

    @validate_call
    def __init__(self,
                 config: TrajectoryDatasetCfg):
        """
        Args:
            config: TrajectoryDatasetCfg object.
        """
        super().__init__()
        self.cfg = config

        # Expose config fields as attributes for internal usage
        self.dataset_dir = self.cfg.dataset_dir
        self.hdf5_file = self.cfg.hdf5_file
        self.hdf5_path = os.path.join(self.dataset_dir, self.hdf5_file)

        # Field config resolution:
        # 1. Use config-provided field_config if available
        # 2. Try to load from yaml file next to hdf5 file
        # 3. Use default HDF5FieldCfg
        if self.cfg.field_config is not None:
            self.field_config = self.cfg.field_config
        else:
            # Try load from yaml
            yaml_path = self._get_field_config_path()
            if os.path.exists(yaml_path):
                print(f"Loading field config from {yaml_path}")
                try:
                    with open(yaml_path, 'r') as f:
                        config_dict = yaml.safe_load(f)
                    self.field_config = HDF5FieldCfg(**config_dict)
                except Exception as e:
                    print(
                        f"[TrajectoryDataset] Warning: Failed to load field config from {yaml_path}: {e}")
                    self.field_config = HDF5FieldCfg()
            else:
                self.field_config = HDF5FieldCfg()

        self.trajectory_state = self.cfg.trajectory_state  # Validated by Pydantic
        self.tensor_args = self.cfg.tensor_args
        self.storage_device = self.cfg.storage_device
        self.normalization_config = self.cfg.normalization_config
        self.load_only_context = self.cfg.load_only_context

        # Field keys for data organization
        self.field_key_traj = 'traj'
        self.field_key_context = 'context'

        # Storage for loaded data
        self.fields = {}
        self.metadata = {}
        self.extra_data = {}  # Store non-context HDF5 fields
        self.context_input_dims = {}
        self.context_normalized: Optional[TrajectoryContext] = None

        self._load_data_from_hdf5()
        self._update_metadata()
        self._setup_normalization()
        self._normalize_all_fields()

    def _update_metadata(self):
        """
        Update metadata with additional information.

        Override this method to add metadata.
        The base metadata (n_trajs, n_support_points, traj_dim, HDF5 attributes)
        is already populated from the HDF5 file.
        """
        pass

    # =========================================================================
    # HDF5 Loading Methods
    # =========================================================================

    def _load_data_from_hdf5(self):
        """Load trajectories and context data from HDF5 file."""
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as f:
            # Read metadata from HDF5 file
            self.metadata = {}

            # Read HDF5 file attributes
            for attr_name in f.attrs:
                self.metadata[attr_name] = f.attrs[attr_name]

            # Load trajectories
            trajs = self._load_trajectories_from_hdf5(f)

            # Ensure trajectories are [n_trajs, horizon, state_dim]
            if trajs.ndim != 3:
                raise ValueError(
                    f"Trajectories must be 3D [n_trajs, horizon, state_dim], got shape {trajs.shape}"
                )

            self.metadata['n_trajs'] = trajs.shape[0]
            self.metadata['n_support_points'] = trajs.shape[1]
            self.metadata['traj_dim'] = trajs.shape[2]

            # Determine q_dim (configuration dimension) from the raw trajectory data
            raw_traj_dim = trajs.shape[2]
            if self.field_config.q_dim is not None:
                self.q_dim = self.field_config.q_dim
                # Validate that traj_dim is a multiple of q_dim (1x, 2x, or 3x for pos, vel, acc)
                if raw_traj_dim % self.q_dim != 0 or raw_traj_dim // self.q_dim > 3:
                    raise ValueError(
                        f"Trajectory dimension {raw_traj_dim} must be 1x, 2x, or 3x the configuration dimension {self.q_dim}. "
                        f"Got {raw_traj_dim // self.q_dim if raw_traj_dim % self.q_dim == 0 else 'invalid'} times q_dim."
                    )
            else:
                self.q_dim = raw_traj_dim

            # Determine what's available in the raw trajectory data
            raw_traj_multiplier = raw_traj_dim // self.q_dim

            # Store trajectories based on trajectory_state
            if self.trajectory_state == 'pos':
                # Position only
                self.fields[self.field_key_traj] = trajs[..., :self.q_dim]
            elif self.trajectory_state == 'pos+vel':
                # Position + velocity
                if raw_traj_multiplier < 2:
                    raise ValueError(
                        f"trajectory_state='pos+vel' requested but data only contains position. "
                        f"Trajectory dimension is {raw_traj_dim}, q_dim is {self.q_dim}"
                    )
                self.fields[self.field_key_traj] = trajs[..., :2*self.q_dim]
            elif self.trajectory_state == 'pos+vel+acc':
                # Position + velocity + acceleration
                if raw_traj_multiplier < 3:
                    raise ValueError(
                        f"trajectory_state='pos+vel+acc' requested but data does not contain acceleration. "
                        f"Trajectory dimension is {raw_traj_dim}, q_dim is {self.q_dim}"
                    )
                self.fields[self.field_key_traj] = trajs[..., :3*self.q_dim]

            # Set dataset dimensions based on what we're actually storing
            b, h, d = self.fields[self.field_key_traj].shape
            self.n_trajs = b
            self.n_support_points = h
            # Actual trajectory dimension we're using (can be q_dim, 2*q_dim, or 3*q_dim)
            self.traj_dim = d

            # Determine what's included in the stored trajectory
            self.traj_multiplier = d // self.q_dim
            if self.traj_multiplier == 1:
                self.has_velocity = False
                self.has_acceleration = False
            elif self.traj_multiplier == 2:
                self.has_velocity = True
                self.has_acceleration = False
            elif self.traj_multiplier == 3:
                self.has_velocity = True
                self.has_acceleration = True

            self.trajectory_dim = (self.n_support_points, self.traj_dim)

            # Load context data and extra metadata
            self._load_context_from_hdf5(f)
            self._load_extra_metadata_from_hdf5(f)

    def _load_trajectories_from_hdf5(self, hdf5_file: h5py.File) -> torch.Tensor:
        """Load trajectory data from HDF5 file."""
        traj_field = self.field_config.trajectory_field

        if traj_field not in hdf5_file:
            if self.load_only_context:
                print(f"[TrajectoryDataset] Warning: Trajectory field '{traj_field}' not found in HDF5 file. "
                      f"Using placeholder zeros since load_only_context is True.")
                # Infer number of samples N from any dataset in the HDF5 file
                N = 0
                for k, v in hdf5_file.items():
                    if isinstance(v, h5py.Dataset):
                        N = v.shape[0]
                        break

                # Use configured q_dim or default to 7
                q_dim = self.field_config.q_dim if self.field_config.q_dim is not None else 7
                trajs = np.zeros((N, 2, q_dim), dtype=np.float32)
            else:
                raise ValueError(
                    f"Trajectory field '{traj_field}' not found in HDF5 file. "
                    f"Available fields: {list(hdf5_file.keys())}"
                )
        else:
            if self.load_only_context:
                # Load only the start and goal states
                traj_dataset = hdf5_file[traj_field]
                n_support_points = traj_dataset.shape[1]

                if n_support_points < 2:
                    raise ValueError(
                        f"Dataset has {n_support_points} support points, but at least 2 are required for context loading."
                    )

                # Slice to get start (index 0) and goal (index -1)
                # HDF5 slicing handles this efficiently
                start = np.array(traj_dataset[:, 0, :])
                goal = np.array(traj_dataset[:, -1, :])

                # Stack to create [N, 2, D]
                trajs = np.stack([start, goal], axis=1)
            else:
                # Load full trajectory
                trajs = np.array(hdf5_file[traj_field])

        # Store on CPU to avoid using GPU memory
        return to_torch(trajs, device=self.storage_device, dtype=self.tensor_args.dtype)

    def _load_context_from_hdf5(self, hdf5_file: h5py.File):
        """Load context data from HDF5 file."""
        # Get context field mappings (auto-detect if not specified)
        available_fields = list(hdf5_file.keys())
        context_field_map = self.field_config.get_context_fields(
            available_fields)

        if not context_field_map:
            # No context data found
            return

        # Temporary storage for components map
        self._temp_components = {}

        # Load each context type
        for context_type, hdf5_fields in context_field_map.items():
            # hdf5_fields is now a list of field names to concatenate
            field_data_list = []
            current_offset = 0

            for hdf5_field in hdf5_fields:
                if hdf5_field not in hdf5_file:
                    raise ValueError(
                        f"Field '{hdf5_field}' not found in HDF5 file")

                # Load this field's data
                data = np.array(hdf5_file[hdf5_field])
                field_data_list.append(data)

                # Track component mapping if we are merging multiple fields
                dim = data.shape[-1]
                self._temp_components[hdf5_field] = (
                    context_type, current_offset, current_offset + dim)
                current_offset += dim

            if not field_data_list:
                # No data loaded for this context type
                continue

            # Concatenate along the last dimension if multiple fields
            if len(field_data_list) == 1:
                context_data = field_data_list[0]
            else:
                context_data = np.concatenate(field_data_list, axis=-1)

            # Store on CPU to avoid GPU memory issues
            context_tensor = to_torch(
                context_data, device=self.storage_device, dtype=self.tensor_args.dtype)

            # Add to temporary dict for creating TrajectoryContext later
            if not hasattr(self, '_temp_context_data'):
                self._temp_context_data = {}
            self._temp_context_data[context_type] = context_tensor

            # ALSO add to self.fields so it gets picked up by normalizer
            # Use specific key format: 'context_{type}'
            field_key = f"{self.field_key_context}_{context_type}"
            self.fields[field_key] = context_tensor

            # Store dimension
            self.context_input_dims[context_type] = context_tensor.shape[-1]

        # After loading all context types, create the full context object if we have data
        if hasattr(self, '_temp_context_data') and self._temp_context_data:
            # IMPORTANT: The tensors in _temp_context_data are the SAME objects as in self.fields
            trajs = self.fields[self.field_key_traj]
            start = trajs[:, 0, :]
            goal = trajs[:, -1, :]

            self.context = TrajectoryContext(
                self._temp_context_data,
                components=self._temp_components,
                start=start,
                goal=goal,
                is_batched=True
            )
            del self._temp_context_data
            del self._temp_components

    def _load_extra_metadata_from_hdf5(self, hdf5_file: h5py.File):
        """Load extra HDF5 fields that are not trajectories or context."""
        # Get all HDF5 field names
        all_fields = set(hdf5_file.keys())

        # Get trajectory field
        trajectory_field = {self.field_config.trajectory_field}

        # Get context fields (flattened list)
        context_field_map = self.field_config.get_context_fields(
            list(all_fields))
        context_fields = set()
        for field_list in context_field_map.values():
            context_fields.update(field_list)

        # Find extra fields (not trajectory or context)
        extra_fields = all_fields - trajectory_field - context_fields

        # Load extra fields as metadata (not normalized)
        for field_name in extra_fields:
            try:
                data = np.array(hdf5_file[field_name])
                # Store on CPU to avoid GPU memory issues
                self.extra_data[field_name] = to_torch(
                    data, device=self.storage_device, dtype=self.tensor_args.dtype)
            except Exception as e:
                print(
                    f"Warning: Could not load extra field '{field_name}': {e}")

    def _get_field_config_path(self) -> str:
        """Get path for field config yaml file."""
        base_name = os.path.splitext(self.hdf5_file)[0]
        return os.path.join(self.dataset_dir, f"{base_name}.yaml")

    def save_field_config(self, save_path: Optional[str] = None):
        """
        Save current field configuration to a YAML file.

        Args:
            save_path: Path to save the YAML file. If None, saves to [dataset_name].yaml
                       next to the HDF5 file.
        """
        if save_path is None:
            save_path = self._get_field_config_path()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        with open(save_path, 'w') as f:
            # simple dump of the model
            yaml.dump(self.field_config.model_dump(),
                      f, default_flow_style=False)

        print(f"Saved field config to {save_path}")

    # =========================================================================
    # Concrete Methods - Provided by base class
    # =========================================================================

    def _setup_normalization(self):
        """Setup normalization with optional custom limits."""

        # Create new normalizer from data (fields always loaded)
        self.normalizer = DatasetNormalizer(
            self.fields,
            normalization_config=self.normalization_config
        )

        # Export config for later use
        self.normalization_config = self.normalizer.export_config()

    def _normalize_all_fields(self):
        """Normalize all data fields."""
        # Get all field keys that need normalization
        normalizer_keys = [
            k for k in self.fields.keys() if not k.endswith('_normalized')]

        for key in normalizer_keys:
            self.fields[f'{key}_normalized'] = self.normalizer(
                self.fields[key], key)

        # Also normalize the full context object if it exists
        if hasattr(self, 'context') and self.context is not None:
            # Instead of re-normalizing (which duplicates data/computation),
            # we construct context_normalized from the ALREADY normalized fields in self.fields

            normalized_data_map = {}
            for context_type in self.context.keys():
                # Key in fields is 'context_{type}_normalized'
                field_key_norm = f"{self.field_key_context}_{context_type}_normalized"

                if field_key_norm in self.fields:
                    # Get the normalized tensor (shared reference)
                    norm_tensor = self.fields[field_key_norm]
                    # Get mask from original context
                    mask = self.context.get_mask(context_type)

                    normalized_data_map[context_type] = {
                        'data': norm_tensor,
                        'mask': mask
                    }

            if normalized_data_map:
                self.context_normalized = TrajectoryContext(
                    normalized_data_map,
                    components=self.context.components,
                    start=self.fields[f'{self.field_key_traj}_normalized'][:, 0, :],
                    goal=self.fields[f'{self.field_key_traj}_normalized'][:, -1, :],
                    is_normalized=True,
                    is_batched=True
                )

    # =========================================================================
    # Public Interface
    # =========================================================================
    def set_normalizer(self, normalizer_config: NormalizationCfg):
        """Set the normalizer for the dataset."""
        self.normalization_config = normalizer_config
        self._setup_normalization()
        self._normalize_all_fields()

    def __len__(self) -> int:
        """Return number of trajectories in dataset."""
        return self.n_trajs

    def __repr__(self) -> str:
        """String representation of dataset."""
        msg = f'{self.__class__.__name__}\n' \
            f'n_trajs: {self.n_trajs}\n' \
            f'n_support_points: {self.n_support_points}\n' \
            f'trajectory_dim: {self.trajectory_dim}\n' \
            f'context_types: {list(self.context_input_dims.keys())}\n'
        return msg

    def __getitem__(self, index: int) -> TrajectorySample:
        """
        Get a single trajectory sample.

        Args:
            index: Index of trajectory to retrieve

        Returns:
            TrajectorySample object
        """
        field_traj_normalized = f'{self.field_key_traj}_normalized'

        # Get data from CPU storage
        traj_normalized = self.fields[field_traj_normalized][index]

        # Get context
        context_normalized = None

        # Add context if available
        if self.context_normalized is not None:
            # Slice the full context to get data for this specific trajectory
            context_normalized = self.context_normalized.slice(index)

        metadata = {}
        for key in self.extra_data:
            metadata[key] = self.extra_data[key][index]

        return TrajectorySample(
            trajectory=traj_normalized,
            context=context_normalized,
            metadata=metadata,
            is_normalized=True
        )

    def random_split(self, val_set_size: float = 0.05, save: Optional[str] = None) -> Tuple[List[int], List[int]]:
        """
        Randomly split dataset into train and validation sets. Returns indices of train and validation sets.

        Args:
            val_set_size: Fraction of data to use for validation

        Returns:
            Tuple of (train_indices, val_indices)
        """
        n_val = int(np.floor(val_set_size * self.n_trajs))
        n_train = self.n_trajs - n_val
        train_indices = np.random.choice(self.n_trajs, n_train, replace=False)
        val_indices = np.setdiff1d(np.arange(self.n_trajs), train_indices)

        if save is not None:
            torch.save((train_indices, val_indices), save)

        return train_indices, val_indices

    def get_dataloader(self,
                       indices: Optional[List[int]] = None,
                       **dataloader_kwargs: Dict[str, Any]) -> DataLoader[TrajectorySample]:
        """
        Get dataloader for dataset.

        Args:
            **dataloader_kwargs: Keyword arguments to pass to DataLoader constructor

        Returns:
            DataLoader object
        """
        # remove collate_fn
        if 'collate_fn' in dataloader_kwargs:
            del dataloader_kwargs['collate_fn']
        dataloader_kwargs['collate_fn'] = TrajectorySample.collate

        if indices is None:
            return DataLoader[TrajectorySample](self, **dataloader_kwargs)

        return DataLoader[TrajectorySample](torch.utils.data.Subset(self, indices), **dataloader_kwargs)
