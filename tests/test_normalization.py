
import pytest
import torch
import numpy as np
import os
import tempfile
from campd.data.normalization import (
    NormalizationCfg,
    DatasetNormalizer,
    Identity,
    GaussianNormalizer,
    LimitsNormalizer,
    SafeLimitsNormalizer,
    FixedLimitsNormalizer
)

# Test NormalizationCfg


def test_normalization_config():
    config = NormalizationCfg()
    assert config.field_limits == {}
    assert config.normalizer_class == 'LimitsNormalizer'

    config.set_limits('test', [0, 0], [1, 1])
    assert config.has_limits('test')
    assert (config.get_limits('test').mins == np.array([0, 0])).all()

# Test Identity Normalizer


def test_identity_normalizer():
    norm = Identity()
    x = torch.randn(5, 5)
    assert torch.allclose(norm.normalize(x), x)
    assert torch.allclose(norm.unnormalize(x), x)

# Test Gaussian Normalizer


def test_gaussian_normalizer():
    # Test from data (deterministic)
    data = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    norm = GaussianNormalizer(X=data)
    normalized = norm.normalize(data)

    # Mean of [0, 2] is 1. Std of [0, 2] is sqrt(2) ~= 1.414
    # (0-1)/1.414 = -0.707
    # (2-1)/1.414 = 0.707
    # Mean should be 0. Std should be 1.

    # Check mean ~0, std ~1
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(normalized.std(dim=0), torch.ones(2), atol=1e-5)

    # Check reconstruction
    reconstructed = norm.unnormalize(normalized)
    assert torch.allclose(reconstructed, data, atol=1e-5)

    # Test with explicit means/stds
    norm2 = GaussianNormalizer(means=torch.zeros(2), stds=torch.ones(2))
    x = torch.tensor([[1.0, -1.0]])
    assert torch.allclose(norm2.normalize(x), x)

# Test Limits Normalizer


def test_limits_normalizer():
    # Test with explicit limits
    mins = torch.tensor([0.0])
    maxs = torch.tensor([10.0])
    norm = LimitsNormalizer(mins=mins, maxs=maxs)

    x = torch.tensor([0.0, 5.0, 10.0])
    normalized = norm.normalize(x.unsqueeze(1))

    # Should map to [-1, 0, 1]
    expected = torch.tensor([-1.0, 0.0, 1.0]).unsqueeze(1)
    assert torch.allclose(normalized, expected)

    # Check reconstruction
    reconstructed = norm.unnormalize(normalized)
    assert torch.allclose(reconstructed, x.unsqueeze(1))

# Test Safe Limits Normalizer


def test_safe_limits_normalizer():
    # Constant data
    data = torch.ones(10, 1) * 5.0
    # Capture stdout to avoid cluttering test output? Or just let it print.
    norm = SafeLimitsNormalizer(X=data, eps=1.0)

    # Mins should be 4.0, Maxs should be 6.0 due to eps
    assert norm.mins[0] == 4.0
    assert norm.maxs[0] == 6.0

    normalized = norm.normalize(data)
    # 5.0 in [4.0, 6.0] -> 0.0 in [-1, 1] (midpoint)
    assert torch.allclose(normalized, torch.zeros_like(normalized))

# Test Fixed Limits Normalizer


def test_fixed_limits_normalizer():
    norm = FixedLimitsNormalizer(X=torch.randn(10, 2), min=-2, max=2)
    assert torch.all(norm.mins == -2)
    assert torch.all(norm.maxs == 2)

# Test DatasetNormalizer


def test_dataset_normalizer():
    dataset = {
        'field1': torch.rand(10, 2),
        'field2': torch.rand(10, 3)
    }

    # Initialize from data
    normalizer = DatasetNormalizer(dataset=dataset)

    # Test normalize
    norm_f1 = normalizer.normalize(dataset['field1'], 'field1')
    assert norm_f1.min() >= -1.0
    assert norm_f1.max() <= 1.0

    # Test unnormalize
    denorm_f1 = normalizer.unnormalize(norm_f1, 'field1')
    assert torch.allclose(denorm_f1, dataset['field1'], atol=1e-5)

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        config = normalizer.export_config()
        DatasetNormalizer.save(config, tmp.name)
        tmp.close()

        loaded_normalizer = DatasetNormalizer.load(tmp.name)

        # Check if loaded normalizer has same limits
        f1_limits = normalizer.normalizers['field1']
        f1_limits_loaded = loaded_normalizer.normalizers['field1']

        assert torch.allclose(f1_limits.mins, f1_limits_loaded.mins)
        assert torch.allclose(f1_limits.maxs, f1_limits_loaded.maxs)

        os.unlink(tmp.name)
