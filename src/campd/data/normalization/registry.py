from campd.utils.registry import Registry
from .base import Normalizer

NORMALIZATION_REGISTRY = Registry[Normalizer]("normalization")
