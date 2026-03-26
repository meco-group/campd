
import pytest
import torch
from campd.data.embedded_context import EmbeddedContext


@pytest.fixture
def sample_embedded_context():
    embeddings = {
        'key1': torch.randn(2, 5, 10),
        'key2': torch.randn(2, 3, 10)
    }
    masks = {
        'key1': torch.ones(2, 5),
        'key2': torch.ones(2, 3)
    }
    masks['key2'][0, 2] = 0  # Some masking

    return EmbeddedContext(embeddings=embeddings, masks=masks, is_batched=True)


def test_embedded_context_init(sample_embedded_context):
    assert isinstance(sample_embedded_context, EmbeddedContext)
    assert 'key1' in sample_embedded_context.keys()
    assert 'key2' in sample_embedded_context.keys()


def test_getitem(sample_embedded_context):
    emb = sample_embedded_context['key1']
    assert emb.shape == (2, 5, 10)

    with pytest.raises(KeyError):
        _ = sample_embedded_context['nonexistent']


def test_get_mask(sample_embedded_context):
    mask = sample_embedded_context.get_mask('key2')
    assert mask.shape == (2, 3)
    assert mask[0, 2] == 0

    with pytest.raises(KeyError):
        sample_embedded_context.get_mask('nonexistent')


def test_concat_keys(sample_embedded_context):
    # Test concatenating specific keys
    config = {
        'merged': ['key1', 'key2']
    }

    # keys must be same length for concat?
    # Wait, _concat_items performs torch.cat(tensors, dim=1).
    # key1 is (2, 5, 10), key2 is (2, 3, 10).
    # Concat dim=1 -> (2, 8, 10). This works.

    merged_ctx = sample_embedded_context.concat_keys(config)
    assert 'merged' in merged_ctx.keys()

    merged_emb = merged_ctx['merged']
    assert merged_emb.shape == (2, 8, 10)

    merged_mask = merged_ctx.get_mask('merged')
    assert merged_mask.shape == (2, 8)

    # Check content
    assert torch.equal(merged_emb[:, :5, :], sample_embedded_context['key1'])
    assert torch.equal(merged_emb[:, 5:, :], sample_embedded_context['key2'])


def test_concat_keys_all(sample_embedded_context):
    # Test concatenating all keys
    config = {
        'all': None
    }
    # Note: dict ordering is preserved in modern python, but keys() order might vary?
    # Actually self.embeddings is a standard dict.

    merged_ctx = sample_embedded_context.concat_keys(config)
    assert 'all' in merged_ctx.keys()

    merged_emb = merged_ctx['all']
    # 5 + 3 = 8
    assert merged_emb.shape == (2, 8, 10)


def test_append():
    embeddings = {'k1': torch.randn(1, 10)}
    masks = {'k1': torch.ones(1)}
    ctx = EmbeddedContext(embeddings, masks, is_batched=False)

    new_emb = torch.randn(1, 10)
    new_mask = torch.ones(1)

    ctx.append('k1', new_emb, new_mask)

    assert ctx['k1'].shape == (2, 10)


def test_null(sample_embedded_context):
    null_ctx = sample_embedded_context.null()

    assert torch.all(null_ctx.get_mask('key1') == 0)
    assert torch.all(null_ctx.get_mask('key2') == 0)
    # Embeddings should be same
    assert torch.equal(null_ctx['key1'], sample_embedded_context['key1'])


def test_concat_batch(sample_embedded_context):
    # Create another batch
    embeddings2 = {
        'key1': torch.randn(3, 5, 10),  # Batch size 3
        'key2': torch.randn(3, 3, 10)
    }
    masks2 = {
        'key1': torch.ones(3, 5),
        'key2': torch.ones(3, 3)
    }
    other = EmbeddedContext(embeddings=embeddings2,
                            masks=masks2, is_batched=True)

    # Concat (2 + 3 = 5 batch size)
    batched = sample_embedded_context.concat_batch(other)

    assert batched['key1'].shape == (5, 5, 10)
    assert batched['key2'].shape == (5, 3, 10)
    assert batched.masks['key1'].shape == (5, 5)


def test_to_device(sample_embedded_context):
    # Just check CPU since GPU might not be available, logic is same
    device = torch.device('cpu')
    ctx_new = sample_embedded_context.to(device)
    assert ctx_new['key1'].device == device
