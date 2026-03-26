from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn

from campd.architectures.context.encoder import ContextEncoder
from campd.architectures.diffusion.base import ReverseDiffusionNetwork
from campd.data.embedded_context import EmbeddedContext
from campd.data.trajectory_sample import TrajectorySample
from campd.models.diffusion.model import ContextTrajectoryDiffusionModel, ContextTrajectoryDiffusionModelCfg, SamplingCfg
from campd.utils.registry import Spec
from campd.data.normalization import NormalizationCfg, DatasetNormalizer


class DummyContextEncoder(ContextEncoder):
    """Minimal ContextEncoder stub for wiring the model in tests."""

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.config = kwargs

    def forward(self, context):  # pragma: no cover
        raise NotImplementedError("Not used in these unit tests")

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance


class RecordingReverseDiffusionNetwork(ReverseDiffusionNetwork):
    """Reverse diffusion network that records calls and makes output depend on context."""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.call_count = 0
        self.last_x = None
        self.last_t = None
        self.last_context = None

    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance

    def forward(self, x: torch.Tensor, t: torch.Tensor, embedded_context_batch: EmbeddedContext | None) -> torch.Tensor:
        self.call_count += 1
        self.last_x = x
        self.last_t = t
        self.last_context = embedded_context_batch

        # Make the output depend on whether context is present and its embedding values.
        # For CFG we want:
        #  - conditional half => multiplier=2
        #  - unconditional half (zeros_like context) => multiplier=1
        # so we interpret multiplier = 1 + mean(context['all']).
        if embedded_context_batch is None:
            multiplier = torch.ones(
                (x.shape[0], 1, 1), device=x.device, dtype=x.dtype)
        else:
            all_emb = embedded_context_batch.embeddings["all"]
            mask = embedded_context_batch.masks.get("all")

            # mask is [B, 1], all_emb is [B, 1, 1]. Need to broadcast or unsqueeze.
            # unsqueeze last dim for mask to match all_emb if needed
            while mask.ndim < all_emb.ndim:
                mask = mask.unsqueeze(-1)
            all_emb = all_emb * mask.float()

            # all_emb expected shape [B, 1, 1] (broadcastable)
            multiplier = 1.0 + \
                all_emb.mean(dim=tuple(range(1, all_emb.ndim)), keepdim=True)

        return x * multiplier


@pytest.fixture
def diffusion_model():
    # We need to construct valid Spec objects that will build our dummy networks.
    # Since Spec uses import_string or registry, and our dummy classes are local to this test file,
    # we'll patch import_string to return our dummy classes when requested by a special name.

    with patch('campd.utils.registry.import_string') as mock_import:
        def side_effect(path):
            if path == "dummy.Network":
                return RecordingReverseDiffusionNetwork
            if path == "dummy.Context":
                return DummyContextEncoder
            # For other imports (like schedulers inside model), use real import
            # We need to import the real import_string to delegate, but we can't import it inside
            # here easily without recursion if we are patching it globally.
            # Ideally we patch it only where it's used.
            # However, ContextTrajectoryDiffusionModel uses it.

            # Simple workaround: manually handle known imports or fallback to real importlib
            try:
                import importlib
                if ":" in path:
                    module_path, attr = path.split(":", 1)
                else:
                    module_path, attr = path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, attr)
            except Exception:
                raise ImportError(f"Could not resolve {path}")

        mock_import.side_effect = side_effect

        net_spec = Spec(cls="dummy.Network", config={})
        ctx_spec = Spec(cls="dummy.Context", config={})

        norm_cfg = NormalizationCfg(
            normalizer_class="LimitsNormalizer",
            field_limits={"traj": {"mins": [0]*3, "maxs": [1]*3}}
        )

        cfg = ContextTrajectoryDiffusionModelCfg(
            state_dim=3,
            model_type="epsilon",
            n_diffusion_steps=10,
            variance_schedule="squaredcos_cap_v2",
            network=net_spec,
            context_network=ctx_spec,
            normalizer=norm_cfg,
        )

        model = ContextTrajectoryDiffusionModel(cfg)
        model.to('cpu')

        yield model


def _make_embedded_context(batch: int, value: float = 1.0) -> EmbeddedContext:
    # Use a single key `all` consistent with other tests.
    emb = torch.full((batch, 1, 1), fill_value=value)
    mask = torch.ones((batch, 1), dtype=torch.bool)
    return EmbeddedContext(embeddings={"all": emb}, masks={"all": mask}, is_batched=True)


def test_forward_with_cond_scale_no_context_calls_network_once(diffusion_model):
    b, h, d = 2, 4, 3
    x = torch.randn(b, h, d)
    t = torch.randint(0, 10, (b,))

    net = diffusion_model.network
    assert isinstance(net, RecordingReverseDiffusionNetwork)

    out = diffusion_model.forward_with_cond_scale(x, t, None, cond_scale=2.0)

    assert net.call_count == 1
    assert out.shape == x.shape


def test_forward_with_cond_scale_1_no_parallel_batch(diffusion_model):
    b, h, d = 2, 4, 3
    x = torch.randn(b, h, d)
    t = torch.randint(0, 10, (b,))
    ctx = _make_embedded_context(b, value=1.0)

    net = diffusion_model.network
    net.call_count = 0

    out = diffusion_model.forward_with_cond_scale(x, t, ctx, cond_scale=1.0)

    # No CFG: should be a single forward call with original batch
    assert net.call_count == 1
    assert net.last_x.shape[0] == b
    assert out.shape == x.shape


def test_forward_with_cond_scale_parallel_cfg_single_network_call(diffusion_model):
    b, h, d = 2, 4, 3
    x = torch.randn(b, h, d)
    t = torch.randint(0, 10, (b,))
    ctx = _make_embedded_context(b, value=1.0)

    net = diffusion_model.network
    net.call_count = 0

    cond_scale = 2.0
    out = diffusion_model.forward_with_cond_scale(
        x, t, ctx, cond_scale=cond_scale, rescaled_phi=0.0)

    # Parallel CFG: one network call with doubled batch
    assert net.call_count == 1
    assert net.last_x.shape[0] == 2 * b
    assert isinstance(net.last_context, EmbeddedContext)
    assert net.last_context.embeddings["all"].shape[0] == 2 * b

    # With our dummy network:
    #  cond pred = 2x, uncond pred = 1x, guided = uncond + (cond - uncond)*s = x + (x)*s
    expected = x + x * cond_scale
    assert torch.allclose(out, expected)


def test_forward_with_cond_scale_rescaled_cfg_phi_1_matches_conditional(diffusion_model):
    torch.manual_seed(0)

    b, h, d = 4, 3, 2
    x = torch.randn(b, h, d)
    t = torch.randint(0, 10, (b,))
    ctx = _make_embedded_context(b, value=1.0)

    net = diffusion_model.network
    net.call_count = 0

    # With our dummy network:
    #  cond pred = 2x, uncond pred = x, guided (cond_scale=2) = 3x.
    # Rescaled CFG rescales guided to have std of cond => 2x.
    out = diffusion_model.forward_with_cond_scale(
        x, t, ctx, cond_scale=2.0, rescaled_phi=1.0)

    assert net.call_count == 1
    assert torch.allclose(out, 2.0 * x, atol=1e-6)


def test_sample_unconditional(diffusion_model):
    cfg = SamplingCfg(
        n_support_points=8,
        batch_size=2,
        num_inference_steps=5,  # fewer steps for test speed
        inference_scheduler_cls='diffusers.DDPMScheduler'
    )

    # Mock context network to avoid error if it were called (it shouldn't be for unconditional)
    # But sample() calls context_network if context is not None.
    # Here context is None.

    # Here context is None.
    diffusion_model.prepare_for_sampling(cfg)
    out = diffusion_model.sample()

    assert isinstance(out, TrajectorySample)
    assert out.trajectory.shape == (2, 8, 3)  # [B, N, D]
    # Check that it runs without error


def test_sample_conditional(diffusion_model):
    # We need to mock the context network because the fixture's one raises generic error
    # and we need it to return an EmbeddedContext.
    b = 2

    # Mock context encoder
    fake_embedded = _make_embedded_context(b, value=0.5)

    class MockContextNetwork(nn.Module):
        def forward(self, ctx):
            return fake_embedded

    diffusion_model.context_network = MockContextNetwork()

    cfg = SamplingCfg(
        n_support_points=8,
        batch_size=b,
        num_inference_steps=5,
        inference_scheduler_cls='diffusers.DDPMScheduler',
        cond_scale=2.0
    )

    class MockContext:
        is_batched = True

        def get_hard_conditions(self):
            return {}

        def denormalize(self, normalizer):
            return self

        def denormalize(self, normalizer):
            return self

        def normalize(self, normalizer):
            return self

        def validate_batch_size(self, n):
            pass

    diffusion_model.prepare_for_sampling(cfg)
    out = diffusion_model.sample(context=MockContext())

    assert isinstance(out, TrajectorySample)
    assert out.trajectory.shape == (b, 8, 3)


@patch('campd.models.diffusion.model.torch.compile', side_effect=lambda x: x)
def test_snapshot_save_load(mock_compile, diffusion_model, tmp_path):
    import os

    # Save snapshot
    # We save directly to tmp_path so model.yaml and checkpoints/ are siblings
    diffusion_model.save_config(str(tmp_path))

    if not (tmp_path / "model_config" / "model.yaml").exists():
        pytest.fail("model_config/model.yaml was not created")

    # Create dummy checkpoint
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pth"
    torch.save({"state_dict": diffusion_model.state_dict()}, checkpoint_path)

    # Patch import_string to handle dummy classes during load
    with patch('campd.utils.registry.import_string') as mock_import:
        def side_effect(path):
            if path == "dummy.Network":
                return RecordingReverseDiffusionNetwork
            if path == "dummy.Context":
                return DummyContextEncoder

            # Normal imports
            import importlib
            if ":" in path:
                module_path, attr = path.split(":", 1)
            else:
                module_path, attr = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr)

        mock_import.side_effect = side_effect

        # Load snapshot
        try:
            loaded_model = ContextTrajectoryDiffusionModel.from_pretrained(
                model_dir=str(tmp_path),
                device='cpu'
            )
        except Exception as e:
            pytest.fail(f"from_pretrained failed with error: {e}")

        if not isinstance(loaded_model, ContextTrajectoryDiffusionModel):
            pytest.fail(
                f"loaded_model is {type(loaded_model)}, expected ContextTrajectoryDiffusionModel")

        if loaded_model.config.state_dim != diffusion_model.config.state_dim:
            pytest.fail(
                f"state_dim mismatch: {loaded_model.config.state_dim} != {diffusion_model.config.state_dim}")


def test_sample_context_expansion(diffusion_model):
    b = 2
    # Create unbatched context

    class MockCtx:
        is_batched = False
        def get_hard_conditions(self): return {}
        def normalize(self, n): return self

        def expand(self, n):
            self.expand_called_with = n
            return self  # return self which is now "expanded"

        def validate_batch_size(self, n): pass
        def denormalize(self, n): return self
        def __call__(self, *args): return self

    ctx = MockCtx()

    cfg = SamplingCfg(
        n_support_points=8,
        batch_size=b,
        num_inference_steps=1,
        inference_scheduler_cls='diffusers.DDPMScheduler'
    )

    # Needs to be nn.Module
    class MockContextNetwork(nn.Module):
        def forward(self, ctx):
            return _make_embedded_context(b)

    diffusion_model.context_network = MockContextNetwork()

    diffusion_model.prepare_for_sampling(cfg)
    diffusion_model.sample(context=ctx)

    assert ctx.expand_called_with == b


def test_sample_batched_context_mismatch(diffusion_model):
    b = 2
    # Create batched context

    class MockCtx:
        is_batched = True
        def get_hard_conditions(self): return {}
        def normalize(self, n): return self

        def validate_batch_size(self, n):
            if n != 2:
                raise ValueError("Batch size mismatch")

    ctx = MockCtx()

    cfg = SamplingCfg(
        n_support_points=8,
        batch_size=3,  # Mismatch
        num_inference_steps=1,
        inference_scheduler_cls='diffusers.DDPMScheduler'
    )

    diffusion_model.prepare_for_sampling(cfg)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        diffusion_model.sample(context=ctx)


def test_sample_batched_context_match(diffusion_model):
    b = 2
    # Create batched context

    class MockCtx:
        is_batched = True
        def get_hard_conditions(self): return {}
        def normalize(self, n): return self
        def denormalize(self, n): return self

        def validate_batch_size(self, n):
            if n != b:
                raise ValueError("Batch size mismatch")

    ctx = MockCtx()

    cfg = SamplingCfg(
        n_support_points=8,
        batch_size=b,
        num_inference_steps=1,
        inference_scheduler_cls='diffusers.DDPMScheduler'
    )

    # Needs to be nn.Module
    class MockContextNetwork(nn.Module):
        def forward(self, ctx):
            return _make_embedded_context(b)

    diffusion_model.context_network = MockContextNetwork()

    diffusion_model.prepare_for_sampling(cfg)
    diffusion_model.sample(context=ctx)
    # Should pass without error
