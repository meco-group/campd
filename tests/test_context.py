
import pytest
import torch
from campd.data.context import TrajectoryContext
from campd.data.normalization import DatasetNormalizer, Identity


@pytest.fixture
def sample_context_data():
    return {
        'cuboids': torch.randn(5, 10),
        'spheres': {
            'data': torch.randn(3, 4),
            'mask': torch.tensor([True, True, False])
        }
    }


@pytest.fixture
def sample_hard_conds():
    return {
        'start': torch.randn(7),
        'goal': torch.randn(7)
    }


def test_trajectory_context_init(sample_context_data, sample_hard_conds):
    ctx = TrajectoryContext(
        data=sample_context_data,
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )

    assert len(ctx) == 2
    assert 'cuboids' in ctx.keys()
    assert 'spheres' in ctx.keys()

    # Check auto-generated mask for cuboids
    assert ctx.get_mask('cuboids').shape == (5,)
    # All valid since randn unlikely to be all zeros
    assert torch.all(ctx.get_mask('cuboids'))

    # Check explicit mask for spheres
    assert torch.equal(ctx.get_mask('spheres'),
                       torch.tensor([True, True, False]))


def test_trajectory_context_getitem(sample_context_data, sample_hard_conds):
    ctx = TrajectoryContext(
        data=sample_context_data,
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )

    assert torch.equal(ctx['spheres'], sample_context_data['spheres']['data'])

    with pytest.raises(KeyError):
        _ = ctx['nonexistent']


def test_virtual_components(sample_hard_conds):
    # Data: [cx, cy, cz, dimx, dimy, dimz]
    data = torch.randn(2, 6)
    components = {
        'centers': ('cuboids', 0, 3),
        'dims': ('cuboids', 3, 6)
    }

    ctx = TrajectoryContext(
        data={'cuboids': data},
        components=components,
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )

    # Get mask via virtual key (should be same as parent)
    mask = ctx.get_mask('centers')
    assert torch.equal(mask, ctx.get_mask('cuboids'))

    # Get item (data + mask)
    item = ctx.get_item('centers')
    assert torch.equal(item, data[:, 0:3])

    # Alternative access via get_item(parent, sub_key)
    item2 = ctx.get_item('cuboids', 'centers')
    assert torch.equal(item2, data[:, 0:3])


def test_context_to_device(sample_context_data, sample_hard_conds):
    ctx = TrajectoryContext(
        data=sample_context_data,
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )

    # Mock device (cpu is the only safe one unless cuda is available, but logic is same)
    device = torch.device('cpu')
    ctx_gpu = ctx.to(device)

    assert ctx_gpu['cuboids'].device == device
    assert ctx_gpu.start.device == device


def test_normalization(sample_context_data, sample_hard_conds):
    ctx = TrajectoryContext(
        data=sample_context_data,
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )

    # Create a mock normalizer
    # We need to register normalizers for 'context_cuboids', 'context_spheres' and 'traj' (for start/goal)
    dataset = {
        'context_cuboids': sample_context_data['cuboids'],
        'context_spheres': sample_context_data['spheres']['data'],
        'traj': torch.randn(10, 7)
    }

    normalizer = DatasetNormalizer(dataset=dataset)

    # Normalize
    ctx_norm = ctx.normalize(normalizer)
    assert ctx_norm.is_normalized

    # Check logic (LimitsNormalizer maps to [-1, 1], so check range roughly)
    assert ctx_norm['cuboids'].max() <= 1.0
    assert ctx_norm['cuboids'].min() >= -1.0

    # Denormalize
    ctx_denorm = ctx_norm.denormalize(normalizer)
    assert not ctx_denorm.is_normalized
    assert torch.allclose(ctx_denorm['cuboids'], ctx['cuboids'], atol=1e-5)


def test_context_merge(sample_context_data, sample_hard_conds):
    ctx1 = TrajectoryContext(
        data={'cuboids': torch.randn(2, 10)},
        start=sample_hard_conds['start'],
        goal=sample_hard_conds['goal'],
        is_batched=False
    )
    ctx2 = TrajectoryContext(
        data={'cuboids': torch.randn(4, 10)},  # Difference number of objects
        start=sample_hard_conds['start'] + 1,
        goal=sample_hard_conds['goal'] + 1,
        is_batched=False
    )

    batch = TrajectoryContext.collate([ctx1, ctx2])
    assert batch.is_batched
    assert isinstance(batch, TrajectoryContext)

    # Check shape: [Batch=2, MaxObj=4, Dim=10]
    assert batch['cuboids'].shape == (2, 4, 10)

    # Check padding/masking
    # First sample had 2 objects, so last 2 should be padded (usually 0)
    # But mask should definitely indicate invalid
    mask = batch.get_mask('cuboids')
    assert mask.shape == (2, 4)
    assert mask[0, 2:].sum() == 0  # Padded area invalid
    assert mask[0, :2].sum() == 2  # Original valid (unless randn made 0s)
    assert mask[1, :].sum() == 4

    # Check start/goal batching
    assert batch.start.shape == (2, 7)
    assert torch.equal(batch.start[0], ctx1.start)
    assert torch.equal(batch.start[1], ctx2.start)


def test_context_batch_slice(sample_hard_conds):
    # Batch data: [2, 3, 5]
    batch_data = torch.randn(2, 3, 5)
    batch_start = torch.stack(
        [sample_hard_conds['start'], sample_hard_conds['start']])
    batch_goal = torch.stack(
        [sample_hard_conds['goal'], sample_hard_conds['goal']])

    batch = TrajectoryContext(
        data={'objs': batch_data},
        start=batch_start,
        goal=batch_goal,
        is_batched=True
    )

    sliced = batch.slice(1)

    assert isinstance(sliced, TrajectoryContext)
    # Should maintain object dim: [3, 5]
    assert sliced['objs'].shape == (3, 5)
    assert torch.equal(sliced['objs'], batch_data[1])
    assert torch.equal(sliced.start, batch_start[1])
