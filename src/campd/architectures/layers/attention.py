import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
from campd.architectures.layers.layers import group_norm_n_groups
from campd.architectures.layers.utils import default, exists


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    # return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    return torch.nn.GroupNorm(num_groups=group_norm_n_groups(in_channels), num_channels=in_channels, eps=1e-6,
                              affine=True)


class CrossAttention(nn.Module):
    """
    Cross-attention implemented with PyTorch SDPA.
    This will use FlashAttention / mem-efficient kernels on CUDA when available.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        x:       (b, n, query_dim)
        context: (b, m, context_dim) or None -> self-attn
        mask:    (b, m) boolean where True means "this key position is valid"
        """
        b, n, _ = x.shape
        context = default(context, x)
        _, m, _ = context.shape

        h, d = self.heads, self.dim_head

        # Project
        q = self.to_q(x)        # (b, n, h*d)
        k = self.to_k(context)  # (b, m, h*d)
        v = self.to_v(context)  # (b, m, h*d)

        # Reshape to (b, h, seq, d)
        q = q.view(b, n, h, d).transpose(1, 2)  # (b, h, n, d)
        k = k.view(b, m, h, d).transpose(1, 2)  # (b, h, m, d)
        v = v.view(b, m, h, d).transpose(1, 2)  # (b, h, m, d)

        attn_mask = None
        if exists(mask):
            # SDPA boolean mask: True = participate, False = masked out. :contentReference[oaicite:2]{index=2}
            # Make it broadcastable to (b, h, n, m)
            attn_mask = mask[:, None, None, :].to(torch.bool)

        # SDPA chooses the fastest available kernel on your GPU.
        # dropout_p must be 0 in eval for deterministic behavior.
        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False
        )  # (b, h, n, d)

        # Back to (b, n, h*d)
        out = out.transpose(1, 2).contiguous().view(b, n, h * d)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    # is self-attn if context is none
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        # Residual connections and layer normalization
        x = self.attn1(self.norm1(x)) + x  # self attention
        x = self.attn2(self.norm2(x), context=context,
                       mask=mask) + x  # attention to context
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for trajectory-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to trajectory
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
             for d in range(depth)]
        )

        self.proj_out = zero_module(
            nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None, mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h -> b h c')
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = rearrange(x, 'b h c -> b c h', h=h)
        x = self.proj_out(x)
        return x + x_in
