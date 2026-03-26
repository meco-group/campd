import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from campd.training.objectives.diffusion import DiffusionObjective, DiffusionObjectiveCfg
from campd.models.diffusion.model import ContextTrajectoryDiffusionModel, ContextTrajectoryDiffusionModelCfg
from campd.architectures.context.encoder import ContextEncoderCfg
from campd.architectures.diffusion.base import ReverseDiffusionNetwork
from campd.architectures.context.encoder import ContextEncoder
from campd.architectures.registry import REVERSE_NETS, CONTEXT_NETS
from campd.utils.registry import Spec
from campd.data.normalization import NormalizationCfg
import diffusers as dm
from campd.data import TrajectorySample
from campd.data.embedded_context import EmbeddedContext


class DummyContext:
    def __init__(self, is_batched=False):
        self.is_batched = is_batched

    def keys(self): return []
    def get_mask(self, key): return None
    def __getitem__(self, key): return None
    def get_hard_conditions(self): return {}


@CONTEXT_NETS.register("DummyContextEncoder")
class DummyContextEncoder(ContextEncoder):
    def __init__(self, config=None):
        super().__init__(config=ContextEncoderCfg(key_networks={}))

    def forward(self, context):
        # Return a dummy EmbeddedContext
        return EmbeddedContext(
            embeddings={"all": torch.zeros(1, 1, 1)},
            masks={"all": torch.ones(1, 1)},
            is_batched=True
        )


@REVERSE_NETS.register("DummyNetwork")
class DummyNetwork(ReverseDiffusionNetwork):
    def __init__(self, state_dim=3):
        super().__init__()
        self.state_dim = state_dim

    def forward(self, x, t, context):
        return torch.zeros_like(x)


@pytest.fixture
def diffusion_model():
    cfg = ContextTrajectoryDiffusionModelCfg(
        state_dim=3,
        network=Spec(cls="DummyNetwork",
                     registry="reverse_diffusion_networks"),
        context_network=Spec(cls="DummyContextEncoder",
                             registry="context_networks"),
        normalizer=NormalizationCfg()
    )
    return ContextTrajectoryDiffusionModel(cfg)


def test_diffusion_objective_step(diffusion_model):
    # Use config dict for initialization
    config = {
        "loss_fn": {
            "cls": "torch.nn.MSELoss"
        }
    }
    objective = DiffusionObjective.from_config(config)

    # Create dummy data
    batch_size = 2
    n_points = 10
    state_dim = 3
    trajectory = torch.randn(batch_size, n_points, state_dim)
    # We need a dummy context that ContextEncoder can handle.
    # Since we mocked ContextEncoder to ignore input, we can pass anything.
    data = TrajectorySample(trajectory=trajectory, context=DummyContext())

    # Run step
    losses, features, _ = objective.step(diffusion_model, data)

    assert isinstance(losses, dict)
    # The key should be the class name of the loss function
    assert 'MSELoss' in losses
    assert isinstance(losses['MSELoss'], torch.Tensor)
    assert losses['MSELoss'].ndim == 0  # scalar loss
    # Features should correspond to model output (dummy zeros)
    assert isinstance(features, torch.Tensor)


def test_diffusion_objective_l1(diffusion_model):
    config = {
        "loss_fn": {
            "cls": "torch.nn.L1Loss"
        }
    }
    objective = DiffusionObjective.from_config(config)
    batch_size = 2
    trajectory = torch.randn(batch_size, 10, 3)
    data = TrajectorySample(trajectory=trajectory, context=DummyContext())

    losses, features, _ = objective.step(diffusion_model, data)
    assert 'L1Loss' in losses
