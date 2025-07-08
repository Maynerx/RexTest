import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from att import GroupedQueryAttention, MultiAttentionHeads, ATTENTION_TYPE
from utils import RMSNorm, MLP
from torch.utils.checkpoint import checkpoint
    

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



"""
# test 

model = Encoder(
    dim=512, 
    vocab_size=10000, 
    num_layers=6, 
    num_heads=8, 
    max_length=512, 
    latent_dim=256, 
    dropout=0.1, 
    is_causal=False,
    attention_type='GQA',
    flash_attention=True
).cuda()


def foo():
    x = torch.randint(0, 10000, (64, 512)).cuda()  # batch_size=32, seq_length=512
    return model(x)
# Example usage

import timeit

print(timeit.timeit(foo, number=100))
"""
