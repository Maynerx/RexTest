import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.attn_bias import LowerTriangularMask
from xformers.ops.fmha.cutlass import FwOp as CutlassFwOp, BwOp as CutlassBwOp
from torch._dynamo import disable
from torch.nn.attention import SDPBackend

@disable
def scaled_dot_product_attention_grouped(
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        scale: float, 
        is_causal: bool = False
        ) -> torch.Tensor:
    """
    Compute scaled dot-product attention with grouped queries.
    
    Args:
        queries (torch.Tensor): Query tensor of shape (B, T_q, C).
        keys (torch.Tensor): Key tensor of shape (B, T_k, C).
        values (torch.Tensor): Value tensor of shape (B, T_v, C).
        scale (float): Scaling factor for the dot product.
        
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    q = rearrange(queries, 'b n h d -> b h n d')
    k = rearrange(keys, 'b s h d -> b h s d')
    v = rearrange(values, 'b t h d -> b h t d')

    bq, hq, nq, dq = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nv, dv = v.shape

    q = q * scale

    num_groups = hq // hk
    q = rearrange(q, 'b (g h) n d -> b g h n d', g=num_groups)


    similarity = einsum(q, k, "b g h n d, b h s d -> b g h n s")

    if is_causal:
        mask = torch.tril(torch.ones((bq, nq, nk), device=q.device)).bool()
        mask = rearrange(mask, 'b n s -> b () () n s')
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    att = F.softmax(similarity, dim=-1)
    out = einsum(att, v, "b g h n s, b h s d -> b g h n d")
    out = rearrange(out, 'b g h n d -> b n (g h) d')
    return out

@disable
def scaled_dot_product_attention_grouped_flash(
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        scale: float, 
        is_causal: bool = False,
        dropout_p: float = 0.0
        ) -> torch.Tensor:
    """
    Compute scaled dot-product attention with grouped queries.
    
    Args:
        queries (torch.Tensor): Query tensor of shape (B, T_q, C).
        keys (torch.Tensor): Key tensor of shape (B, T_k, C).
        values (torch.Tensor): Value tensor of shape (B, T_v, C).
        scale (float): Scaling factor for the dot product.
        
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    q = queries.permute(0, 2, 1, 3)
    k = keys.permute(0, 2, 1, 3)
    v = values.permute(0, 2, 1, 3)

    bq, hq, nq, dq = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nv, dv = v.shape

    repeat = hq // hk
    k = k.repeat_interleave(repeat, dim=1)  # (B, hq, Tk, d)
    v = v.repeat_interleave(repeat, dim=1)  # (B, hq, Tv, d)

    attn_bias = None
    if is_causal:
        attn_bias = LowerTriangularMask(device=q.device, dtype=q.dtype)
    
    q = q * scale

    out = memory_efficient_attention(
        query=q,
        key=k,
        value=v,
        attn_bias=attn_bias,
        op=[CutlassFwOp(), CutlassBwOp()],
    )
    out = out.permute(0, 2, 1, 3)

    return out


def scaled_dot_product_attention_grouped_flash_fix(
        queries: torch.Tensor,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        scale: float, 
        is_causal: bool = False,
        dropout_p: float = 0.0
        ) -> torch.Tensor:
    """
    Compute scaled dot-product attention with grouped queries.
    
    Args:
        queries (torch.Tensor): Query tensor of shape (B, T_q, C).
        keys (torch.Tensor): Key tensor of shape (B, T_k, C).
        values (torch.Tensor): Value tensor of shape (B, T_v, C).
        scale (float): Scaling factor for the dot product.
        
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    q = queries.permute(0, 2, 1, 3)
    k = keys.permute(0, 2, 1, 3)
    v = values.permute(0, 2, 1, 3)

    bq, hq, nq, dq = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nv, dv = v.shape

    repeat = hq // hk
    k = k.repeat_interleave(repeat, dim=1)  # (B, hq, Tk, d)
    v = v.repeat_interleave(repeat, dim=1)  # (B, hq, Tv, d)

    with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )
    out = out.permute(0, 2, 1, 3)

    return out




def precompute_freq_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    # dim must be even
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # [max_seq_len, dim/2]
    freqs = torch.outer(t, inv_freq)
    # pack into complex numbers: cos + i sin
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [max_seq_len, dim/2]


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    x: float tensor of shape [B, H, N, D], with D even
    freqs_cis: complex tensor [max_seq_len, D/2]
    returns: real tensor [B, H, N, D] with rotary applied
    """
    B, H, N, D = x.shape
    assert D % 2 == 0, "Embedding dimension must be even for RoPE"

    # view last dimension as complex: [..., D/2, 2] -> complex
    x_complex = torch.view_as_complex(
        x.float().reshape(B, H, N, D // 2, 2)
    )  # complex64 : [B, H, N, D/2]

    # slice & broadcast freqs to [1, 1, N, D/2]
    freqs = freqs_cis[:N, :].to(x.device)          # [N, D/2]
    freqs = freqs.unsqueeze(0).unsqueeze(0)        # [1, 1, N, D/2]

    # apply rotation in the complex plane
    x_rotated = x_complex * freqs                  # broadcasting across B, H

    # convert back to real
    x_out = torch.view_as_real(x_rotated)          # [B, H, N, D/2, 2]
    return x_out.reshape(B, H, N, D).to(x.dtype)   # [B, H, N, D]



class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor):
        h = F.silu(self.w1(x))
        h = self.dropout(h)
        out = self.w2(h * self.w3(x))
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
