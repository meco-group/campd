from __future__ import annotations
import einops
import torch
from typing import Dict
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
import pydantic_numpy.typing as pnd
from .registry import NORMALIZATION_REGISTRY
from .base import Normalizer

# -----------------------------------------------------------------------------#
# --------------------------- normalization config ----------------------------#
# -----------------------------------------------------------------------------#


class FieldLimit(BaseModel):
    mins: pnd.Np1DArray
    maxs: pnd.Np1DArray


class NormalizationCfg(BaseModel):
    """
    Configuration for normalization limits.

    Allows specifying custom min/max limits for specific fields.
    If a field is not specified, limits will be computed from data.

    Example::

        config = NormalizationCfg(
            field_limits={
                'context_boxes': {
                    'mins': np.array([-2, -2, 0, 0, 0.1, 0.1, -1, -1, -1, -1]),
                    'maxs': np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
                },
                'traj': {
                    'mins': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                    'maxs': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
                }
            }
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=False)
    field_limits: Dict[str, FieldLimit] = Field(default_factory=dict)
    field_means: Dict[str, pnd.Np1DArray] = Field(default_factory=dict)
    field_stds: Dict[str, pnd.Np1DArray] = Field(default_factory=dict)
    normalizer_class: str = 'LimitsNormalizer'

    def is_empty(self) -> bool:
        return not self.field_limits and not self.field_means and not self.field_stds

    def get_limits(self, field_name: str) -> FieldLimit:
        """Get normalization limits for a specific field."""
        assert self.has_limits(
            field_name), f'Field {field_name} not found in normalization config'
        return self.field_limits[field_name]

    def has_limits(self, field_name: str) -> bool:
        """Check if limits are specified for a field."""
        return field_name in self.field_limits

    def set_limits(self, field_name: str, mins: np.ndarray, maxs: np.ndarray):
        """Set normalization limits for a field."""
        self.field_limits[field_name] = FieldLimit(mins=mins, maxs=maxs)

    def set_means(self, field_name: str, means: np.ndarray):
        """Set means for a field."""
        self.field_means[field_name] = means

    def set_stds(self, field_name: str, stds: np.ndarray):
        """Set stds for a field."""
        self.field_stds[field_name] = stds

    def get_means(self, field_name: str) -> np.ndarray:
        """Get means for a field."""
        assert self.has_means(
            field_name), f'Field {field_name} not found in normalization config'
        return self.field_means[field_name]

    def get_stds(self, field_name: str) -> np.ndarray:
        """Get stds for a field."""
        assert self.has_stds(
            field_name), f'Field {field_name} not found in normalization config'
        return self.field_stds[field_name]

    def has_means(self, field_name: str) -> bool:
        """Check if means are specified for a field."""
        return field_name in self.field_means

    def has_stds(self, field_name: str) -> bool:
        """Check if stds are specified for a field."""
        return field_name in self.field_stds

# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#


class DatasetNormalizer:

    def __init__(self,
                 dataset: Dict[str, torch.Tensor] | None = None,
                 normalization_config: NormalizationCfg | dict | None = None):
        """
        Initialize dataset normalizer.

        Args:
            dataset: Dictionary of data fields to normalize, or None if loading from config
            normalization_config: Optional configuration with explicit limits per field
        """
        if normalization_config is None:
            normalization_config = NormalizationCfg()
        else:
            normalization_config = NormalizationCfg.model_validate(
                normalization_config)
        self.normalization_config = normalization_config

        # Determine normalizer class
        norm_cls_name = self.normalization_config.normalizer_class
        self.normalizer = NORMALIZATION_REGISTRY[norm_cls_name]

        if dataset is not None:
            dataset = flatten(dataset)

            self.normalizers = {}
            for key, val in dataset.items():
                # Check if explicit limits are provided for this field
                # Check if explicit limits/means are provided for this field
                if self.normalization_config.has_limits(key) or self.normalization_config.has_means(key):
                    kwargs = {'X': None}

                    if self.normalization_config.has_limits(key):
                        limits = self.normalization_config.get_limits(key)
                        kwargs['mins'] = self._to_torch(limits.mins)
                        kwargs['maxs'] = self._to_torch(limits.maxs)

                    if self.normalization_config.has_means(key):
                        kwargs['means'] = self._to_torch(
                            self.normalization_config.get_means(key))

                    if self.normalization_config.has_stds(key):
                        kwargs['stds'] = self._to_torch(
                            self.normalization_config.get_stds(key))

                    self.normalizers[key] = self.normalizer(**kwargs)
                else:
                    # Compute limits from data
                    self.normalizers[key] = self.normalizer(val)
        else:
            # Initialize from config limits if available
            self.normalizers = {}
            if self.normalization_config.field_limits or self.normalization_config.field_means:
                # Get all unique keys from both limits and means/stds
                all_keys = set(self.normalization_config.field_limits.keys()) | \
                    set(self.normalization_config.field_means.keys())

                for key in all_keys:
                    kwargs = {'X': None}

                    if self.normalization_config.has_limits(key):
                        limits = self.normalization_config.get_limits(key)
                        kwargs['mins'] = self._to_torch(limits.mins)
                        kwargs['maxs'] = self._to_torch(limits.maxs)

                    if self.normalization_config.has_means(key):
                        kwargs['means'] = self._to_torch(
                            self.normalization_config.get_means(key))

                    if self.normalization_config.has_stds(key):
                        kwargs['stds'] = self._to_torch(
                            self.normalization_config.get_stds(key))

                    self.normalizers[key] = self.normalizer(**kwargs)

    @classmethod
    def from_config(cls, cfg: NormalizationCfg | dict | None) -> "DatasetNormalizer":
        """Build a DatasetNormalizer from a saved config.
        """
        return cls(dataset=None, normalization_config=cfg)

    def _to_torch(self, arr):
        """Convert array to torch tensor if needed."""
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        elif isinstance(arr, torch.Tensor):
            return arr.float()
        else:
            return torch.tensor(arr).float()

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def normalize_delta(self, x, key):
        """Normalize a delta (difference) using only the scale factor."""
        return self.normalizers[key].normalize_delta(x)

    def get_field_normalizers(self):
        return self.normalizers

    def export_config(self):
        config = NormalizationCfg(
            normalizer_class=self.normalization_config.normalizer_class)
        for key, normalizer in self.normalizers.items():
            if hasattr(normalizer, 'mins') and hasattr(normalizer, 'maxs'):
                config.set_limits(
                    key,
                    normalizer.mins.detach().cpu().numpy(),
                    normalizer.maxs.detach().cpu().numpy()
                )

            if hasattr(normalizer, 'means'):
                config.set_means(
                    key,
                    normalizer.means.detach().cpu().numpy()
                )

            if hasattr(normalizer, 'stds'):
                config.set_stds(
                    key,
                    normalizer.stds.detach().cpu().numpy()
                )
        return config

    def to(self, device: torch.device):
        for normalizer in self.normalizers.values():
            if hasattr(normalizer, 'mins') and isinstance(normalizer.mins, torch.Tensor):
                normalizer.mins = normalizer.mins.to(device)
            if hasattr(normalizer, 'maxs') and isinstance(normalizer.maxs, torch.Tensor):
                normalizer.maxs = normalizer.maxs.to(device)
            if hasattr(normalizer, 'means') and isinstance(normalizer.means, torch.Tensor):
                normalizer.means = normalizer.means.to(device)
            if hasattr(normalizer, 'stds') and isinstance(normalizer.stds, torch.Tensor):
                normalizer.stds = normalizer.stds.to(device)
        return self

    @staticmethod
    def save(normalization_config: NormalizationCfg, path: str):
        with open(path, 'w') as f:
            yaml.dump(normalization_config.model_dump(), f)

    @staticmethod
    def load(path: str) -> DatasetNormalizer:
        with open(path, 'r') as f:
            normalization_config = NormalizationCfg.model_validate(
                yaml.load(f, Loader=yaml.UnsafeLoader))
        return DatasetNormalizer(normalization_config=normalization_config)

# def flatten(dataset):
#     '''
#         flattens dataset of { key: [ batch x length x dim ] }
#             to { key : [ (batch * length) x dim ]}
#     '''
#     flattened = {}
#     for key, xs in dataset.items():
#         flattened[key] = einops.rearrange(xs, 'b h d -> (b h) d')
#     return flattened


def flatten(dataset: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
        flattens dataset of { key: [ ... x dim ] }
            to { key : [ (...) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        xs_new = xs
        if xs.ndim == 2:
            # environments (e d)
            pass
        elif xs.ndim == 3:
            # trajectories in fixed environments
            xs_new = einops.rearrange(xs, 'b h d -> (b h) d')
        elif xs.ndim == 4:
            # trajectories in variable environments
            xs_new = einops.rearrange(xs, 'e b h d -> (e b h) d')
        else:
            raise NotImplementedError
        flattened[key] = xs_new
    return flattened


@NORMALIZATION_REGISTRY.register('Identity')
class Identity(Normalizer):
    def __init__(self, *args, **kwargs):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x

    def normalize_delta(self, delta):
        return delta


@NORMALIZATION_REGISTRY.register('GaussianNormalizer')
class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, means=None, stds=None,  **kwargs):
        # If X is None, Normalizer.__init__ expects mins/maxs.
        # GaussianNormalizer doesn't use them, so we pass dummies if not provided.
        if 'X' not in kwargs and (not args or args[0] is None):
            if 'mins' not in kwargs:
                kwargs['mins'] = torch.tensor([])
            if 'maxs' not in kwargs:
                kwargs['maxs'] = torch.tensor([])

        super().__init__(*args, **kwargs)
        if self.X is None:
            self.means = means
            self.stds = stds
        else:
            self.means = self.X.mean(dim=0)
            self.stds = self.X.std(dim=0)
            self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {torch.round(self.means, decimals=2)}\n    '''
            f'''stds: {torch.round(self.z * self.stds, decimals=2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

    def normalize_delta(self, delta):
        return delta / self.stds


@NORMALIZATION_REGISTRY.register('LimitsNormalizer')
class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        # [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        # [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        x = x.clamp(-1, 1)

        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

    def normalize_delta(self, delta):
        return 2 * delta / (self.maxs - self.mins)


@NORMALIZATION_REGISTRY.register('SafeLimitsNormalizer')
class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                      f'''max = min = {self.maxs[i]}'''
                      )
                self.mins -= eps
                self.maxs += eps


@NORMALIZATION_REGISTRY.register('FixedLimitsNormalizer')
class FixedLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but with fixed limits not derived from the data
    '''

    def __init__(self, *args, min=-1, max=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mins = torch.ones_like(self.mins) * min
        self.maxs = torch.ones_like(self.maxs) * max
