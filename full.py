import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.attn_bias import LowerTriangularMask
from xformers.ops.fmha.cutlass import FwOp as CutlassFwOp, BwOp as CutlassBwOp
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch._dynamo


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

    @torch._dynamo.disable
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
            out = scaled_dot_product_attention_grouped_flash(q, k, v, self.scale, self.is_causal)
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


class DecoderLayer(nn.Module):
    def __init__(self,
                dim: int, 
                latent_dim: int,
                num_heads: int, 
                max_length: int, 
                dropout: float = 0.1, 
                attention_type: str = 'MHA',
                flash_attention: bool = False
                ):
        super(DecoderLayer, self).__init__()
        assert attention_type in ATTENTION_TYPE, f"Invalid attention type: {attention_type}. Choose from {ATTENTION_TYPE}"
        if attention_type == 'MHA':
            self.masked_attention = MultiAttentionHeads(
                dim,
                dim,
                num_heads, 
                max_length, 
                dropout, 
                is_causal=True
            )
        elif attention_type == 'GQA': 
            self.masked_attention = GroupedQueryAttention(
                dim,
                dim,
                max(1, num_heads // 4),
                num_heads, 
                max_length, 
                dropout, 
                is_causal=True,
                flash_attention=flash_attention
            )
        elif attention_type == 'MLA':
            raise NotImplementedError("MLA (Multi Latent Attention) is not implemented yet.")
        
        self.norm_masked_attn_in = RMSNorm(dim)
        self.norm_masked_attn_out = RMSNorm(dim)
        if attention_type == 'MHA':
            self.cross_attention = MultiAttentionHeads(
                dim,
                latent_dim,
                num_heads, 
                max_length, 
                dropout, 
                is_causal=False,
                apply_rotary=False
            )
        elif attention_type == 'GQA':
            self.cross_attention = GroupedQueryAttention(
                dim,
                latent_dim,
                max(1, num_heads // 4),
                num_heads, 
                max_length, 
                dropout, 
                is_causal=False,
                flash_attention=flash_attention
            )
        elif attention_type == 'MLA':
            raise NotImplementedError("MLA (Multi Latent Attention) is not implemented yet.")
        
        self.norm_cross_attn_in = RMSNorm(dim)
        self.norm_cross_attn_out = RMSNorm(dim)
        self.mlp = MLP(dim, dropout)
        self.norm_mlp_in = RMSNorm(dim)
        self.norm_mlp_out = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # Masked self-attention
        x = self.norm_masked_attn_in(x)
        masked_attn_output = self.attn_dropout(self.masked_attention(x, x, x))
        x = x + masked_attn_output
        x = self.norm_masked_attn_out(x)

        # Cross-attention with latent representation
        x = self.norm_cross_attn_in(x)
        cross_attn_output = self.attn_dropout(self.cross_attention(x, latent, latent))
        x = x + cross_attn_output
        x = self.norm_cross_attn_out(x)

        # MLP
        x = self.norm_mlp_in(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm_mlp_out(x)

        return x


class Decoder(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int,
                num_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1,
                attention_type: str = 'MHA',
                flash_attention: bool = False
                ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderLayer(dim, latent_dim, num_heads, max_length, dropout, attention_type, flash_attention) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, latent)#checkpoint(layer, x, latent, use_reentrant=False)
        x = self.norm(x)
        x = self.out(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, 
                dim: int, 
                num_heads: int, 
                max_length: int, 
                dropout: float = 0.1, 
                is_causal: bool = False,
                attention_type: str = 'MHA',
                flash_attention: bool = False
                ):
        super().__init__()
        assert attention_type in ATTENTION_TYPE, f"Invalid attention type: {attention_type}. Choose from {ATTENTION_TYPE}"
        if attention_type == 'MHA':
            self.attention = MultiAttentionHeads(
                dim,
                dim,
                num_heads, 
                max_length, 
                dropout, 
                is_causal)
        elif attention_type == 'GQA':
            self.attention = GroupedQueryAttention(
                dim,
                dim,
                max(1, num_heads // 4),  # GQA typically uses half the number of heads
                num_heads, 
                max_length, 
                dropout, 
                is_causal,
                flash_attention=flash_attention)
        elif attention_type == 'MLA':
            raise NotImplementedError("MLA (Multi Latent Attention) is not implemented yet.")

        self.norm_attn_in  = RMSNorm(dim)
        self.norm_attn_out = RMSNorm(dim)
        self.norm_mlp_in   = RMSNorm(dim)
        self.norm_mlp_out  = RMSNorm(dim)
        self.mlp = MLP(dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_attn_in(x)
        attn_output_ = self.attention(x, x, x)
        attn_output = self.attn_dropout(attn_output_)
        x = x + attn_output
        x = self.norm_attn_out(x)
        x = self.norm_mlp_in(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm_mlp_out(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int,
                num_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1, 
                is_causal: bool = False,
                attention_type: str = 'MHA',
                flash_attention: bool = False
                ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            EncoderLayer(dim, num_heads, max_length, dropout, is_causal, attention_type, flash_attention) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.latent_proj = nn.Linear(dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)#checkpoint(layer, x, use_reentrant=False)
        x = self.norm(x)
        return self.latent_proj(x)


class Transformer(nn.Module):
    def __init__(self, 
                dim: int, 
                vocab_size: int, 
                encoder_layers: int,
                decoder_layers: int, 
                num_heads: int, 
                max_length: int, 
                latent_dim: int, 
                dropout: float = 0.1, 
                is_causal: bool = False,
                attention_type: str = 'MHA',
                flash_attention: bool = False
                ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            dim=dim, 
            vocab_size=vocab_size, 
            num_layers=encoder_layers, 
            num_heads=num_heads, 
            max_length=max_length, 
            latent_dim=latent_dim, 
            dropout=dropout, 
            is_causal=is_causal,
            attention_type=attention_type,
            flash_attention=flash_attention
        )
        self.decoder = Decoder(
            dim=dim, 
            vocab_size=vocab_size, 
            num_layers=decoder_layers, 
            num_heads=num_heads, 
            max_length=max_length, 
            latent_dim=latent_dim, 
            dropout=dropout,
            attention_type=attention_type,
            flash_attention=flash_attention
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(src)
        output = self.decoder(tgt, latent)
        return output
    
