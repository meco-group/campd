import pytest
import torch.nn as nn
from campd.training.registry import LOSSES, CALLBACKS, SUMMARIES
from campd.utils.registry import Spec


class TestLossesRegistry:
    def test_registry_exists(self):
        """Test that LOSSES registry exists."""
        assert LOSSES is not None
        assert hasattr(LOSSES, '_map')

    def test_register_and_get(self):
        """Test registering and retrieving from LOSSES registry."""
        @LOSSES.register("TestLoss")
        class TestLoss(nn.Module):
            def forward(self, x):
                return x

        assert "TestLoss" in LOSSES._map
        retrieved = LOSSES["TestLoss"]
        assert retrieved is TestLoss

    def test_build_from_registry(self):
        """Test building loss from registry."""
        from campd.training.losses import WeightedL1
        spec = Spec(cls="WeightedL1", init={})
        loss = spec.build_from(LOSSES)
        assert isinstance(loss, WeightedL1)

    def test_build_with_init_args(self):
        """Test building loss with initialization arguments."""
        import torch
        from campd.training.losses import WeightedL2
        weights = torch.ones(1, 2, 3)
        spec = Spec(cls="WeightedL2", init={"weights": weights})
        loss = spec.build_from(LOSSES)
        assert isinstance(loss, WeightedL2)
        assert torch.equal(loss.weights, weights)


class TestCallbacksRegistry:
    def test_registry_exists(self):
        """Test that CALLBACKS registry exists."""
        assert CALLBACKS is not None
        assert hasattr(CALLBACKS, '_map')

    def test_register_and_get(self):
        """Test registering and retrieving from CALLBACKS registry."""
        @CALLBACKS.register("RegistryTestCallback")
        class RegistryTestCallback:
            pass

        assert "RegistryTestCallback" in CALLBACKS._map
        retrieved = CALLBACKS["RegistryTestCallback"]
        assert retrieved is RegistryTestCallback

    def test_build_from_registry(self):
        """Test building callback from registry."""
        from campd.training.callbacks import EMACallback
        spec = Spec(cls="EMACallback", init={"decay": 0.95})
        callback = spec.build_from(CALLBACKS)
        assert isinstance(callback, EMACallback)
        assert callback.decay == 0.95

    def test_predefined_callbacks_registered(self):
        """Test that predefined callbacks are registered."""
        assert "EMACallback" in CALLBACKS._map
        assert "CheckpointCallback" in CALLBACKS._map
        assert "EarlyStoppingCallback" in CALLBACKS._map
        assert "PrinterCallback" in CALLBACKS._map


class TestSummariesRegistry:
    def test_registry_exists(self):
        """Test that SUMMARIES registry exists."""
        assert SUMMARIES is not None
        assert hasattr(SUMMARIES, '_map')

    def test_register_and_get(self):
        """Test registering and retrieving from SUMMARIES registry."""
        @SUMMARIES.register("TestSummaryReg")
        class TestSummaryReg:
            pass

        assert "TestSummaryReg" in SUMMARIES._map
        retrieved = SUMMARIES["TestSummaryReg"]
        assert retrieved is TestSummaryReg


class TestSpecBuildFrom:
    def test_spec_build_from_losses(self):
        """Test Spec.build_from with LOSSES registry."""
        spec = Spec(cls="WeightedL1", init={})
        loss = spec.build_from(LOSSES)
        from campd.training.losses import WeightedL1
        assert isinstance(loss, WeightedL1)

    def test_spec_build_from_callbacks(self):
        """Test Spec.build_from with CALLBACKS registry."""
        spec = Spec(cls="EMACallback", init={"decay": 0.99})
        callback = spec.build_from(CALLBACKS)
        from campd.training.callbacks import EMACallback
        assert isinstance(callback, EMACallback)
        assert callback.decay == 0.99

    def test_spec_build_from_summaries_with_kwargs(self):
        """Test Spec.build_from with SUMMARIES registry and extra kwargs."""
        from campd.training.summary import Summary

        @SUMMARIES.register("KwargSummary")
        class KwargSummary(Summary):
            def __init__(self, every_n_steps: int = 1, **kwargs):
                super().__init__(every_n_steps=every_n_steps)
                self.kwargs = kwargs

            def _run(self, model, train_dataloader, val_dataloader, step):
                pass

        spec = Spec(cls="KwargSummary", init={"every_n_steps": 5})

        # Mock trainer
        class MockTrainer:
            pass

        trainer = MockTrainer()
        summary = spec.build_from(SUMMARIES, trainer=trainer)

        assert isinstance(summary, KwargSummary)
        assert summary.every_n_steps == 5

    def test_spec_with_full_import_path(self):
        """Test Spec with full import path (not in registry)."""
        spec = Spec(cls="torch.nn.MSELoss", init={})
        loss = spec.build()
        assert isinstance(loss, nn.MSELoss)
