from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
import os
import tempfile
from campd.training.callbacks import (
    Callback, EMACallback,
    EarlyStoppingCallback, PrinterCallback
)
from campd.training.registry import CALLBACKS
from campd.training.base import Trainer, TrainerCfg
from campd.utils.registry import Spec
from campd.utils.torch import TensorArgs
from torch.utils.data import DataLoader, Dataset
from campd.training.registry import LOSSES
from pydantic import BaseModel

from campd.data import TrajectorySample


@LOSSES.register("CallbackTestLoss")
class CallbackTestLoss(nn.Module):
    def forward(self, pred, target):
        loss = (pred - target).pow(2).mean()
        return loss, {}


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch):
        target = torch.stack(batch.metadata["y"])
        return self(batch.trajectory), target


class MockCallbackObjective(nn.Module):
    def step(self, model, batch):
        pred = model(batch.trajectory)
        target = torch.stack(batch.metadata["y"])
        loss = (pred - target).pow(2).mean()
        return {"loss": loss}, [pred], {}


class DictDataset(Dataset):
    """Simple dataset that returns dicts."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return {"x": self.X[idx], "y": self.y[idx]}
        return TrajectorySample(
            trajectory=self.X[idx],
            metadata={"y": self.y[idx]}
        )


class _DummyCfg(BaseModel):
    foo: int = 1


class _DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyCfg(foo=2)

    def forward(self, x):
        return x


class _DummyCtx(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyCfg(foo=3)

    def forward(self, x):
        return x


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyCfg(foo=4)
        self.network = _DummyNet()
        self.context_network = _DummyCtx()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch):
        target = torch.stack(batch.metadata["y"])
        return self(batch.trajectory), target


@CALLBACKS.register("MockCallbackTest")
class MockCallbackTest(Callback):
    """Test callback to track hook calls."""

    def __init__(self):
        self.calls = []

    def on_train_start(self, trainer):
        self.calls.append("on_train_start")

    def on_train_end(self, trainer):
        self.calls.append("on_train_end")

    def on_epoch_start(self, trainer):
        self.calls.append("on_epoch_start")

    def on_epoch_end(self, trainer, train_losses=None):
        self.calls.append("on_epoch_end")

    def on_batch_start(self, trainer, batch):
        self.calls.append("on_batch_start")

    def on_batch_end(self, trainer, batch, loss_dict):
        self.calls.append("on_batch_end")

    def on_validation_start(self, trainer):
        self.calls.append("on_validation_start")

    def on_validation_end(self, trainer, val_losses):
        self.calls.append("on_validation_end")


class TestEMACallback:
    def test_ema_initialization(self):
        """Test that EMA model is created on train start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=1,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="EMACallback", init={"decay": 0.99})],
                results_dir=tmpdir,
                use_torchjd=False,  # Disable for simple test
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)

            assert trainer.ema_model is not None
            assert isinstance(trainer.ema_model, SimpleModel)
            # EMA model should have no gradients
            for param in trainer.ema_model.parameters():
                assert not param.requires_grad

    def test_ema_updates(self):
        """Test that EMA model updates during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(20, 10)
            y = torch.randn(20, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            initial_params = [p.clone() for p in model.parameters()]

            cfg = TrainerCfg(
                max_epochs=2,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="EMACallback")],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)

            # Store initial EMA params
            initial_ema_params = [p.clone()
                                  for p in trainer.ema_model.parameters()]

            trainer.fit(dataloader)

            # Model params should have changed
            for p_init, p_final in zip(initial_params, model.parameters()):
                assert not torch.equal(p_init, p_final)

            # EMA params should have changed
            for p_init, p_final in zip(initial_ema_params, trainer.ema_model.parameters()):
                assert not torch.equal(p_init, p_final)

    def test_ema_start_step(self):
        """Test that EMA doesn't update before start_step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=1,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="EMACallback", init={
                                "start_step": 1000, "update_every": 1})],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)

            initial_ema_params = [p.clone()
                                  for p in trainer.ema_model.parameters()]
            trainer.fit(dataloader)

            # Since we only do 2 steps (8 samples / 4 batch_size),
            # and start_step is 1000, EMA should match current model
            for p_ema, p_model in zip(trainer.ema_model.parameters(), model.parameters()):
                assert torch.allclose(p_ema, p_model, atol=1e-5)


class TestCheckpointCallback:
    def test_checkpoint_creation(self):
        """Test that checkpoints are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=2,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="CheckpointCallback")],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)
            trainer.fit(dataloader)

            # Check that checkpoints exist
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            assert os.path.exists(checkpoint_dir)
            assert os.path.exists(os.path.join(checkpoint_dir, "last.pth"))

    def test_best_checkpoint(self):
        """Test that best checkpoint is saved based on validation loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=3,
                val_check_interval=1,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="CheckpointCallback",
                                init={"save_best": True})],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)
            trainer.fit(dataloader, dataloader)

            # Check that best checkpoint exists
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            assert os.path.exists(os.path.join(checkpoint_dir, "best.pth"))


class TestEarlyStoppingCallback:
    def test_early_stopping_triggers(self):
        """Test that early stopping stops training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=100,  # Set high, should stop early
                val_check_interval=1,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="EarlyStoppingCallback", init={
                                "patience": 5, "min_delta": 100.0})],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)
            trainer.fit(dataloader, dataloader)

            # Should stop before max_epochs
            assert trainer.current_epoch < 100

    def test_patience_mechanism(self):
        """Test that patience counter works correctly."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.0)

        # Create mock trainer
        class MockTrainer:
            current_epoch = 0
            stop_training = False

        trainer = MockTrainer()

        # First validation - improvement
        callback.on_validation_end(trainer, {"total": 1.0})
        assert callback.counter == 0
        assert not trainer.stop_training

        # Second validation - no improvement
        callback.on_validation_end(trainer, {"total": 1.5})
        assert callback.counter == 1
        assert not trainer.stop_training

        # Third validation - no improvement
        callback.on_validation_end(trainer, {"total": 1.5})
        assert callback.counter == 2
        assert not trainer.stop_training

        # Fourth validation - no improvement, should trigger
        callback.on_validation_end(trainer, {"total": 1.5})
        assert callback.counter == 3
        assert trainer.stop_training


class TestPrinterCallback:
    def test_printer_callback(self, capsys):
        """Test that printer callback prints messages."""
        callback = PrinterCallback()

        class MockTrainer:
            pass

        trainer = MockTrainer()

        callback.on_train_start(trainer)
        captured = capsys.readouterr()
        assert "Training started" in captured.out

        callback.on_train_end(trainer)
        captured = capsys.readouterr()
        assert "Training ended" in captured.out


class TestCallbackHooks:
    def test_all_hooks_called(self):
        """Test that all callback hooks are called in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            dataset = DictDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=4,
                                    collate_fn=TrajectorySample.collate)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=2,
                val_check_interval=1,
                tensor_args=TensorArgs(device="cpu"),
                callbacks=[Spec(cls="MockCallbackTest")],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(
                    cls="tests.test_callbacks.MockCallbackObjective")
            )
            trainer = Trainer(cfg, model)
            trainer.fit(dataloader, dataloader)

            callback = trainer.callbacks[0]

            # Check that hooks were called
            assert "on_train_start" in callback.calls
            assert "on_train_end" in callback.calls
            assert callback.calls.count("on_epoch_start") == 2
            assert callback.calls.count("on_epoch_end") == 2
            assert callback.calls.count("on_validation_start") == 2
            assert callback.calls.count("on_validation_end") == 2
            assert callback.calls.count("on_batch_start") > 0
            assert callback.calls.count("on_batch_end") > 0

            # Check order
            assert callback.calls[0] == "on_train_start"
            assert callback.calls[-1] == "on_train_end"


class TestCallbackRegistry:
    def test_callbacks_registered(self):
        """Test that callbacks are properly registered."""
        assert "EMACallback" in CALLBACKS._map
        assert "CheckpointCallback" in CALLBACKS._map
        assert "EarlyStoppingCallback" in CALLBACKS._map
        assert "PrinterCallback" in CALLBACKS._map

    def test_build_from_registry(self):
        """Test building callbacks from registry."""
        spec = Spec(cls="EMACallback", init={"decay": 0.99})
        callback = spec.build_from(CALLBACKS)
        assert isinstance(callback, EMACallback)
        assert callback.decay == 0.99
