
import pytest
import torch
from campd.data.trajectory_sample import TrajectorySample
from campd.data.context import TrajectoryContext
from campd.data.normalization import DatasetNormalizer


@pytest.fixture
def mock_sample():
    traj = torch.randn(10, 7)
    context = TrajectoryContext(
        data={'obs': torch.randn(5, 3)},
        start=traj[0],
        goal=traj[-1],
        is_batched=False
    )
    return TrajectorySample(trajectory=traj, context=context)


def test_trajectory_sample_init(mock_sample):
    assert mock_sample.trajectory.shape == (10, 7)
    assert not mock_sample.is_normalized
    assert isinstance(mock_sample.context, TrajectoryContext)


def test_trajectory_sample_normalization(mock_sample):
    # Setup mock normalizer
    # We normalized TrajectorySample calls normalizer.normalize(traj, 'traj')
    # and context.normalize(normalizer)

    dataset = {
        'traj': mock_sample.trajectory,
        'context_obs': mock_sample.context['obs']
    }
    normalizer = DatasetNormalizer(dataset=dataset)

    # Normalize
    norm_sample = mock_sample.normalize(normalizer)
    assert norm_sample.is_normalized
    assert norm_sample.context.is_normalized
    assert norm_sample.trajectory.min() >= -1.0

    # Denormalize
    denorm_sample = norm_sample.denormalize(normalizer)
    assert not denorm_sample.is_normalized
    assert torch.allclose(denorm_sample.trajectory,
                          mock_sample.trajectory, atol=1e-5)


def test_trajectory_batch_collate(mock_sample):
    # Create list of samples
    samples = [mock_sample, mock_sample]  # 2 identical samples

    batch = TrajectorySample.collate(samples)

    assert isinstance(batch, TrajectorySample)
    assert batch.is_batched
    assert len(batch) == 2

    # Check trajectory batching
    assert batch.trajectory.shape == (2, 10, 7)

    # Check context batching (should be newly merged context)
    assert isinstance(batch.context, TrajectoryContext)
    assert batch.context.is_batched
    assert batch.context['obs'].shape == (2, 5, 3)  # [B, N, D]


def test_expand_shared_context():
    # Setup manually a TrajectoryBatch with shared context
    # Shared context means context is just a TrajectoryContext (not batched), or singular

    # Case 1: Context is a single object (1 env), shared across batch of trajs
    traj_batch = torch.randn(5, 10, 7)  # Batch 5
    shared_ctx = TrajectoryContext(
        # Single environment object [1, 3] or [3]
        data={'obs': torch.randn(1, 3)},
        start=torch.randn(7),
        goal=torch.randn(7),
        is_batched=False
    )

    batch = TrajectorySample(
        trajectory=traj_batch,
        context=shared_ctx,
        shared_context=True,
        is_batched=True
    )

    # Expand
    batch.expand_shared_context()

    assert not batch.shared_context
    assert isinstance(batch.context, TrajectoryContext)
    assert batch.context.is_batched

    # Context data should be expanded to [5, 1, 3] since original was [1, 3] (or [5, 3] if original was just [3] auto-unsqueezed?)
    # Context logic:
    # If d.ndim == 2 (N, D), it expands to (B, N, D).
    # If our data was [1, 3] -> TrajectoryContext treats it as N=1, D=3.
    # Expanded -> [5, 1, 3].

    obs = batch.context['obs']
    assert obs.shape == (5, 1, 3)  # Batch=5, N=1, D=3

    # Verify content is identical across batch
    assert torch.equal(obs[0], obs[1])
