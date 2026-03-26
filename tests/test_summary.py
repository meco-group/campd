import pytest
import torch
from campd.training.summary import Summary
from campd.training.registry import SUMMARIES
from campd.utils.registry import Spec
import torch.nn as nn


class SummaryTestObjective(nn.Module):
    def step(self, model, batch):
        pred = model(batch["x"])
        loss = (pred - batch["y"]).pow(2).mean()
        return {"loss": loss}, [pred]


@SUMMARIES.register("SummaryTestMock")
class SummaryTestMock(Summary):
    """Test summary class."""

    def __init__(self, every_n_steps: int = 1):
        super().__init__(every_n_steps=every_n_steps)
        self.run_count = 0
        self.last_step = None

    def _run(self, model, train_dataloader, val_dataloader, step):
        self.run_count += 1
        self.last_step = step


class MockTrainer:
    def __init__(self, model):
        self.model = model
        self.train_dataloader = None
        self.val_dataloader = None
        self.global_step = 0


class TestSummaryBase:
    def test_should_run_every_step(self):
        """Test should_run with every_n_steps=1."""
        summary = SummaryTestMock(every_n_steps=1)

        assert not summary.should_run(0)  # step 0 never runs
        assert summary.should_run(1)
        assert summary.should_run(2)
        assert summary.should_run(100)

    def test_should_run_every_n_steps(self):
        """Test should_run with every_n_steps=5."""
        summary = SummaryTestMock(every_n_steps=5)

        assert not summary.should_run(0)  # step 0 never runs
        assert not summary.should_run(1)
        assert not summary.should_run(2)
        assert not summary.should_run(3)
        assert not summary.should_run(4)
        assert summary.should_run(5)
        assert not summary.should_run(6)
        assert summary.should_run(10)
        assert summary.should_run(15)

    def test_should_run_large_interval(self):
        """Test should_run with large interval."""
        summary = SummaryTestMock(every_n_steps=100)

        assert not summary.should_run(0)  # step 0 never runs
        assert not summary.should_run(50)
        assert not summary.should_run(99)
        assert summary.should_run(100)
        assert summary.should_run(200)

    def test_run_calls_internal_run(self):
        """Test that run() calls _run() and manages model state."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

        model = DummyModel()
        trainer = MockTrainer(model)
        summary = SummaryTestMock(every_n_steps=1)

        # Set model to training mode
        model.train()
        assert model.training

        # Run summary
        summary.run(trainer, step=5)

        # Check that _run was called
        assert summary.run_count == 1
        assert summary.last_step == 5

        # Model should be back in training mode
        assert model.training

    def test_run_preserves_eval_mode(self):
        """Test that run() preserves eval mode if model was in eval."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

        model = DummyModel()
        trainer = MockTrainer(model)
        summary = SummaryTestMock(every_n_steps=1)

        # Set model to eval mode
        model.eval()
        assert not model.training

        # Run summary
        summary.run(trainer, step=5)

        # Model should still be in eval mode
        assert not model.training

    def test_run_assertion_on_wrong_step(self):
        """Test that run() asserts if called at wrong step."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

        model = DummyModel()
        trainer = MockTrainer(model)
        summary = SummaryTestMock(every_n_steps=5)

        # Should work at step 5
        summary.run(trainer, step=5)

        # Should fail at step 3
        with pytest.raises(AssertionError):
            summary.run(trainer, step=3)


class TestSummaryRegistry:
    def test_summary_registered(self):
        """Test that test summary is registered."""
        assert "SummaryTestMock" in SUMMARIES._map
        summary_cls = SUMMARIES["SummaryTestMock"]
        assert summary_cls is SummaryTestMock

    def test_build_from_registry(self):
        """Test building summary from registry."""
        spec = Spec(cls="SummaryTestMock", init={"every_n_steps": 10})
        summary = spec.build_from(SUMMARIES)
        assert isinstance(summary, SummaryTestMock)
        assert summary.every_n_steps == 10


class TestSummaryTrainerInjection:
    def test_trainer_injection_via_spec(self):
        """Test that trainer is injected when building from Spec."""
        from campd.training.base import Trainer, TrainerCfg
        from campd.utils.torch import TensorArgs
        import torch.nn as nn
        import tempfile

        @SUMMARIES.register("TrainerAwareSummary")
        class TrainerAwareSummary(Summary):
            def __init__(self, every_n_steps: int = 1):
                super().__init__(every_n_steps=every_n_steps)

            def _run(self, model, train_dataloader, val_dataloader, step):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 1)

            model = SimpleModel()
            cfg = TrainerCfg(
                max_epochs=1,
                tensor_args=TensorArgs(device="cpu"),
                summaries=[Spec(cls="TrainerAwareSummary",
                                init={"every_n_steps": 5})],
                results_dir=tmpdir,
                use_torchjd=False,
                objective=Spec(cls="tests.test_summary.SummaryTestObjective")
            )

            trainer = Trainer(cfg, model)

            # Check that trainer was injected
            summary = trainer.summaries[0]
            assert isinstance(summary, TrainerAwareSummary)
