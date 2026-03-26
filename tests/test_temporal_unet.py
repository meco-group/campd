import torch
import pytest
from campd.architectures.diffusion.temporal_unet import TemporalUnet, TemporalUnetCfg
from campd.data.embedded_context import EmbeddedContext


def test_temporal_unet_initialization():
    config = TemporalUnetCfg(
        state_dim=10,
        n_support_points=16,
        unet_input_dim=16,
        dim_mults=(1, 2),
        time_emb_dim=32,
    )
    model = TemporalUnet(config)
    assert isinstance(model, TemporalUnet)


def test_temporal_unet_forward():
    batch_size = 2
    horizon = 16
    state_dim = 10

    config = TemporalUnetCfg(
        state_dim=state_dim,
        n_support_points=horizon,
        unet_input_dim=16,
        dim_mults=(1, 2),
        time_emb_dim=32,
        conditioning_embed_dim=0
    )
    model = TemporalUnet(config)

    x = torch.randn(batch_size, horizon, state_dim)
    t = torch.randint(0, 100, (batch_size,))

    # Empty context batch
    embedded_context_batch = EmbeddedContext(
        embeddings={}, masks={}, is_batched=True)

    output = model(x, t, embedded_context_batch)

    assert output.shape == (batch_size, horizon, state_dim)


def test_temporal_unet_forward_with_conditioning():
    batch_size = 2
    horizon = 16
    state_dim = 10
    cond_dim = 8

    config = TemporalUnetCfg(
        state_dim=state_dim,
        n_support_points=horizon,
        unet_input_dim=16,
        dim_mults=(1, 2),
        time_emb_dim=32,
        conditioning_embed_dim=cond_dim,
        enable_conditioning=True,
        attention_num_heads=4,
        attention_dim_head=32
    )
    model = TemporalUnet(config)

    x = torch.randn(batch_size, horizon, state_dim)
    t = torch.randint(0, 100, (batch_size,))

    cond_emb = torch.randn(batch_size, 1, cond_dim)
    # The model hardcodes conditioning_key = 'all'
    embedded_context_batch = EmbeddedContext(
        embeddings={'all': cond_emb},
        masks={'all': None},
        is_batched=True
    )

    output = model(x, t, embedded_context_batch)

    assert output.shape == (batch_size, horizon, state_dim)
