
import torch
import torch.nn as nn

from campd.data.context import TrajectoryContext
from campd.data.embedded_context import EmbeddedContext
from campd.architectures.context import ContextEncoder
from campd.architectures.registry import MODULES


@MODULES.register("MockNetwork")
class MockNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def test_context_encoder():
    # Setup
    input_dim = 10
    output_dim = 5

    # Create mock context data
    data = {
        'boxes': torch.randn(2, 3, input_dim)  # Batch=2, 3 boxes, dim=10
    }
    start = torch.randn(2, input_dim)
    goal = torch.randn(2, input_dim)

    context = TrajectoryContext(
        data, start, goal, is_batched=True, is_normalized=True)

    # Define config
    config = {
        "key_networks": {
            "boxes": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            }
        }
    }

    encoder = ContextEncoder.from_config(config)

    # Forward
    embedded_context = encoder(context)

    # Verify
    assert isinstance(embedded_context, EmbeddedContext)
    assert 'boxes' in embedded_context.keys()

    embeddings = embedded_context['boxes']
    assert embeddings.shape == (2, 3, output_dim)

    mask = embedded_context.get_mask('boxes')
    assert mask is not None
    assert mask.shape == (2, 3)


def test_context_encoder_missing_key():
    # Setup
    input_dim = 10
    output_dim = 5

    data = {
        'boxes': torch.randn(2, 3, input_dim)
    }
    start = torch.randn(2, input_dim)
    goal = torch.randn(2, input_dim)
    context = TrajectoryContext(
        data, start, goal, is_batched=True, is_normalized=True)

    # Config defines network for 'spheres', but data has 'boxes'
    config = {
        "key_networks": {
            "spheres": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            }
        }
    }

    encoder = ContextEncoder.from_config(config)

    # Forward
    embedded_context = encoder(context)

    # 'spheres' should be missing from output because it wasn't in input
    assert 'spheres' not in embedded_context.keys()
    # 'boxes' should be missing because no network defined
    assert 'boxes' not in embedded_context.keys()


def test_context_encoder_device_move():
    if not torch.cuda.is_available():
        return

    device = torch.device('cuda')
    input_dim = 10
    output_dim = 5

    data = {'boxes': torch.randn(2, 3, input_dim)}
    start = torch.randn(2, input_dim)
    goal = torch.randn(2, input_dim)
    context = TrajectoryContext(
        data, start, goal, is_batched=True, is_normalized=True)

    config = {
        "key_networks": {
            "boxes": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            }
        }
    }
    encoder = ContextEncoder.from_config(config).to(device)

    # Move context to element
    context = context.to(device)

    # Forward
    embedded_context = encoder(context)

    assert embedded_context['boxes'].device.type == 'cuda'


def test_context_encoder_concatenation():
    # Setup
    input_dim = 10
    output_dim = 5
    batch_size = 2

    # Create mock context data with multiple keys
    data = {
        'boxes': torch.randn(batch_size, 3, input_dim),
        'spheres': torch.randn(batch_size, 2, input_dim),
        'points': torch.randn(batch_size, 5, input_dim)
    }
    start = torch.randn(batch_size, input_dim)
    goal = torch.randn(batch_size, input_dim)
    context = TrajectoryContext(
        data, start, goal, is_batched=True, is_normalized=True)

    # Define concat configuration
    concat_config = {
        'all': None,  # Default concatenation
        'objects': ['boxes', 'spheres'],
        'spatial': ['points']
    }

    config = {
        "key_networks": {
            "boxes": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            },
            "spheres": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            },
            "points": {
                "cls": "MockNetwork",
                "init": {"input_dim": input_dim, "output_dim": output_dim},
                "registry": "modules"
            }
        },
        "concat_config": concat_config
    }

    encoder = ContextEncoder.from_config(config)

    # Forward
    embedded_context = encoder(context)

    # Verify 'all' concatenation
    concatenated_all = embedded_context['all']
    # Total items = 3 + 2 + 5 = 10. Output dim = 5. Batch = 2.
    assert concatenated_all.shape == (batch_size, 10, output_dim)
    # Check mask for 'all'
    assert embedded_context.get_mask('all') is not None
    assert embedded_context.get_mask('all').shape == (batch_size, 10)

    # Verify 'objects' concatenation
    concatenated_objects = embedded_context['objects']
    # boxes(3) + spheres(2) = 5
    assert concatenated_objects.shape == (batch_size, 5, output_dim)
    assert 'objects' in embedded_context.keys()

    # Verify 'spatial' concatenation
    concatenated_spatial = embedded_context['spatial']
    # points(5)
    assert concatenated_spatial.shape == (batch_size, 5, output_dim)
