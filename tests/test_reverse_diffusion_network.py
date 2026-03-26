import torch
import pytest
from campd.architectures.diffusion.base import ReverseDiffusionNetwork
from campd.data.embedded_context import EmbeddedContext


class MockReverseDiffusionNetwork(ReverseDiffusionNetwork):
    def forward(self, x: torch.Tensor, t: torch.Tensor, embedded_context_batch: EmbeddedContext) -> torch.Tensor:
        return x * t


def test_reverse_diffusion_network_instantiation():
    """Test that we can instantiate a concrete implementation of ReverseDiffusionNetwork."""
    model = MockReverseDiffusionNetwork()
    assert isinstance(model, ReverseDiffusionNetwork)
    assert isinstance(model, torch.nn.Module)


def test_reverse_diffusion_network_abstract_error():
    """Test that we cannot instantiate the abstract base class directly."""
    with pytest.raises(TypeError):
        ReverseDiffusionNetwork()


def test_forward_signature():
    """Test the forward method signature with mock data."""
    model = MockReverseDiffusionNetwork()
    x = torch.randn(10, 5)
    t = torch.randn(10, 1)
    # Mock EmbeddedContextBatch since it's just a type hint verification mostly
    # But let's create a minimal valid object if possible or just mock it
    # EmbeddedContextBatch requires embeddings and masks
    embeddings = {"key": torch.randn(10, 5)}
    masks = {"key": torch.ones(10, 5)}
    context = EmbeddedContext(embeddings=embeddings,
                              masks=masks, is_batched=True)

    output = model(x, t, context)
    assert output.shape == (10, 5)
