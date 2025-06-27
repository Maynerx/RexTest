import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import scaled_dot_product_attention_grouped, apply_rotary_emb, precompute_freq_cis


ATTENTION_TYPE = ['MHA', 'GQA', 'MLA']


# ============== Attention Mechanisms ==============

# Classic Multi-Head Attention
class MultiAttentionHeads(nn.Module):
    def __init__(self, dim: int, k_dim: int, num_heads: int, max_length: int, dropout: int = 0.1, is_causal: bool = False, apply_rotary: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.is_causal = is_causal
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(k_dim, dim)
        self.v_proj = nn.Linear(k_dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // num_heads
        self.apply_rotary = apply_rotary
        
        self.scale = self.head_dim**-0.5
        if self.apply_rotary:
            self.register_buffer("freqs_cis", precompute_freq_cis(self.head_dim, max_length), persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, T, C = query.size()
        B2,  S, Ck  = key.shape 


        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.apply_rotary:
            self.freqs_cis = self.freqs_cis.to(query.device)
            q = apply_rotary_emb(q, self.freqs_cis[:T])
            k = apply_rotary_emb(k, self.freqs_cis[:S])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.is_causal:
            mask = torch.tril(torch.ones((T, T), device=query.device)).view(1, 1, T, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)
        att = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(att)
        return out
    

# Grouped Query Attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, k_dim: int, kv_heads: int, query_heads: int, max_length: int, dropout: int = 0.1, is_causal: bool = False, apply_rotary: bool = True):
        super().__init__()
        assert dim % query_heads == 0, "dim must be divisible by query_heads"
        self.dim = dim
        self.kv_heads = kv_heads
        self.query_heads = query_heads
        self.is_causal = is_causal
        kv_dim = (dim // query_heads) * kv_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(k_dim, kv_dim)
        self.v_proj = nn.Linear(k_dim, kv_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // query_heads

        self.apply_rotary = apply_rotary
        if self.apply_rotary:
            self.register_buffer("freqs_cis", precompute_freq_cis(self.head_dim, max_length), persistent=False)

        self.scale = self.head_dim**-0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)

        if self.apply_rotary:
            self.freqs_cis = self.freqs_cis.to(query.device)
            q = apply_rotary_emb(q, self.freqs_cis[:q.size(1)])
            k = apply_rotary_emb(k, self.freqs_cis[:k.size(1)])

        out = scaled_dot_product_attention_grouped(q, k, v, self.scale, self.is_causal)
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.out_proj(out)
        return out
