import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import scaled_dot_product_attention_grouped, apply_rotary_emb, precompute_freq_cis, scaled_dot_product_attention_grouped_flash, scaled_dot_product_attention_grouped_flash_fix
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch._dynamo

#torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


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
    def __init__(self, 
                dim: int, 
                k_dim: int, 
                kv_heads: int, 
                query_heads: int, 
                max_length: int, 
                dropout: int = 0.1, 
                is_causal: bool = False, 
                apply_rotary: bool = True, 
                flash_attention: bool = False
                ):
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

        self.flash_attention = flash_attention

        self.apply_rotary = apply_rotary
        if self.apply_rotary:
            self.register_buffer("freqs_cis", precompute_freq_cis(self.head_dim, max_length), persistent=False)

        self.scale = self.head_dim**-0.5

    #@torch._dynamo.disable
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        bq, nq, dq = q.shape
        bk, nk, dk = k.shape
        bv, nv, dv = v.shape

        q = q.view(bq, nq, self.query_heads, dq // self.query_heads)
        k = k.view(bk, nk, self.kv_heads, dk // self.kv_heads)
        v = v.view(bv, nv, self.kv_heads, dv // self.kv_heads)

        #q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        #k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        #v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)

        if self.apply_rotary:
            freqs_cis = self.freqs_cis.to(query.device)
            q = apply_rotary_emb(q, freqs_cis[:q.size(1)])
            k = apply_rotary_emb(k, freqs_cis[:k.size(1)])

        if self.flash_attention:
            #q = q.permute(0, 2, 1, 3).contiguous()  # (B, Hq, Nq, Dq)
            #k = k.permute(0, 2, 1, 3).contiguous()  # 
            #v = v.permute(0, 2, 1, 3).contiguous()  # (B, Hv, Nv, Dv)
            out = scaled_dot_product_attention_grouped_flash_fix(q, k, v, self.scale, self.is_causal)
            """
            with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout.p,
                is_causal=self.is_causal,
                scale=self.scale,
                enable_gqa=True
                )
                """
            #out = out.permute(0, 2, 1, 3).contiguous()  # (B, Nq, Hq, Dq)
        else:
            out = scaled_dot_product_attention_grouped(q, k, v, self.scale, self.is_causal)
        out = out.reshape(out.size(0), out.size(1), out.size(2) * out.size(3))  # Flatten the heads
        #out = rearrange(out, "b n h d -> b n (h d)")
        out = self.out_proj(out)
        return out
