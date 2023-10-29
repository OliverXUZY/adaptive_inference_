import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding(seq_len, n_freqs, method=0):
    """
    Sinusoid position encoding
    """
    if method == 0: # transformer (https://arxiv.org/abs/1706.03762)
        tics = torch.arange(seq_len)
        freqs = 10000 ** (torch.arange(n_freqs) / n_freqs)
        x = tics[None, :] / freqs[:, None]                      # (n, t)
    else:           # perceiver (https://arxiv.org/abs/2103.03206)
        tics = (torch.arange(seq_len)) / seq_len * 2 - 1
        freqs = torch.linspace(1, seq_len / 2, n_freqs)
        x = math.pi * freqs[:, None] * tics[None, :]            # (n, t)
    pe = torch.cat([torch.sin(x), torch.cos(x)])                # (n * 2, t)

    return pe


class LayerNorm(nn.Module):
    """
    LayerNorm that supports input of size (bs, c, t)
    """
    def __init__(self, n_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
        """
        assert x.size(1) == self.n_channels

        # channel-wise normalization
        mu = torch.mean(x, dim=1, keepdim=True)
        x = x - mu
        sigma = torch.mean(x ** 2, dim=1, keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    NOTE: This implementation supports both self-attention and cross-attention.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        q_dim=None,         # query dimension
        kv_dim=None,        # key / value dimension
        out_dim=None,       # output dimension
        n_heads=4,          # number of attention heads
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
    ):
        super(MultiHeadAttention, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, q, k=None, v=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            k (float tensor, (bs, c, t2)): key feature sequence.
            v (float tensor, (bs, c, t2)): value feature sequence.
        """
        bs, c = q.size(0), self.embd_dim
        h, d = self.n_heads, self.n_channels

        if k is None:
            k = q
        if v is None:
            v = q

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(bs, h, d, -1).transpose(2, 3)            # (bs, h, t1, d)
        k = k.view(bs, h, d, -1)                            # (bs, h, d, t2)
        v = v.view(bs, h, d, -1).transpose(2, 3)            # (bs, h, t2, d)

        attn = (q * self.scale) @ k                         # (bs, h, t1, t2)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        q = attn @ v                                        # (bs, h, t1, d)
        
        q = q.transpose(2, 3).contiguous().view(bs, c, -1)  # (bs, c, t1)
        q = self.proj_drop(self.proj(q))
        
        return q


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block.
    """
    def __init__(
        self,
        embd_dim,               # embedding dimension
        n_heads=4,              # number of attention heads
        expansion=4,            # expansion factor for FFN
        attn_pdrop=0.0,         # dropout rate for attention map
        proj_pdrop=0.0,         # dropout rate for projection
        path_pdrop=0.0,         # dropout rate for residual paths
    ):
        super(TransformerEncoderBlock, self).__init__()

        # self-attention
        self.attn = MultiHeadAttention(
            embd_dim, 
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim * expansion, 1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(embd_dim * expansion, embd_dim, 1),
            nn.Dropout(proj_pdrop),
        )

        self.ln1 = LayerNorm(embd_dim)
        self.ln2 = LayerNorm(embd_dim)

        self.drop_path_attn = self.drop_path_ffn = nn.Identity()
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
        """
        # self-attention
        dx = self.attn(self.ln1(x))
        x = x + self.drop_path_attn(dx)
        
        # FFN
        dx = self.ffn(self.ln2(x))
        x = x + self.drop_path_ffn(dx)

        return x


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask
    
    return x


class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks) 
    with a per channel scaling factor (and zero init).

    https://arxiv.org/pdf/2103.17239.pdf
    """
    def __init__(self, dim, drop_prob=0.0, init_scale=1e-4):
        super(AffineDropPath, self).__init__()

        self.scale = nn.Parameter(init_scale * torch.ones((1, dim, 1)))
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)