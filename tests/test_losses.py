import pytest
import torch
import torch.nn as nn
from campd.training.losses import WeightedL1, WeightedL2
from campd.training.registry import LOSSES


class TestWeightedL1:
    def test_forward_no_weights(self):
        """Test WeightedL1 loss without weights."""
        loss_fn = WeightedL1()
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        targ = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # [1, 2, 2]

        loss, info = loss_fn(pred, targ)

        # Expected: mean of absolute differences = mean([1, 1, 1, 1]) = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))
        assert info == {}

    def test_forward_with_weights(self):
        """Test WeightedL1 loss with custom weights."""
        weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        loss_fn = WeightedL1(weights=weights)
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        targ = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])

        loss, info = loss_fn(pred, targ)

        # Expected: mean of weighted absolute differences
        # abs_diff = [1, 1, 1, 1]
        # weighted = [1*1, 1*2, 1*3, 1*4] = [1, 2, 3, 4]
        # mean = 10/4 = 2.5
        assert torch.isclose(loss, torch.tensor(2.5))
        assert info == {}

    def test_shape_preservation(self):
        """Test that loss handles different batch sizes."""
        loss_fn = WeightedL1()
        pred = torch.randn(4, 10, 5)  # [batch, horizon, dim]
        targ = torch.randn(4, 10, 5)

        loss, info = loss_fn(pred, targ)

        assert loss.dim() == 0  # Scalar output
        assert loss.item() >= 0  # Loss should be non-negative


class TestWeightedL2:
    def test_forward_no_weights(self):
        """Test WeightedL2 loss without weights."""
        loss_fn = WeightedL2()
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        targ = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])

        loss, info = loss_fn(pred, targ)

        # Expected: mean of squared differences = mean([1, 1, 1, 1]) = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))
        assert info == {}

    def test_forward_with_weights(self):
        """Test WeightedL2 loss with custom weights."""
        weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        loss_fn = WeightedL2(weights=weights)
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        targ = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])

        loss, info = loss_fn(pred, targ)

        # Expected: mean of weighted squared differences
        # sq_diff = [1, 1, 1, 1]
        # weighted = [1*1, 1*2, 1*3, 1*4] = [1, 2, 3, 4]
        # mean = 10/4 = 2.5
        assert torch.isclose(loss, torch.tensor(2.5))
        assert info == {}

    def test_shape_preservation(self):
        """Test that loss handles different batch sizes."""
        loss_fn = WeightedL2()
        pred = torch.randn(4, 10, 5)
        targ = torch.randn(4, 10, 5)

        loss, info = loss_fn(pred, targ)

        assert loss.dim() == 0  # Scalar output
        assert loss.item() >= 0  # Loss should be non-negative


class TestLossRegistry:
    def test_weighted_l1_registered(self):
        """Test that WeightedL1 is registered."""
        assert "WeightedL1" in LOSSES._map
        loss_cls = LOSSES["WeightedL1"]
        assert loss_cls is WeightedL1

    def test_weighted_l2_registered(self):
        """Test that WeightedL2 is registered."""
        assert "WeightedL2" in LOSSES._map
        loss_cls = LOSSES["WeightedL2"]
        assert loss_cls is WeightedL2

    def test_mse_registered(self):
        """Test that MSE loss is registered."""
        assert "MSE" in LOSSES._map
        loss_cls = LOSSES["MSE"]
        assert loss_cls is nn.MSELoss

    def test_build_from_registry(self):
        """Test building loss from registry."""
        loss_cls = LOSSES["WeightedL1"]
        loss = loss_cls()
        assert isinstance(loss, WeightedL1)

        # Test with init args
        weights = torch.ones(1, 2, 3)
        loss_cls = LOSSES["WeightedL2"]
        loss = loss_cls(weights=weights)
        assert isinstance(loss, WeightedL2)
        assert torch.equal(loss.weights, weights)
