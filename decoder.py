import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLP, RMSNorm
from att import GroupedQueryAttention, MultiAttentionHeads, ATTENTION_TYPE
from torch.utils.checkpoint import checkpoint

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
    

"""
# Example usage:

model = Decoder(
    dim=512,
    vocab_size=10000,
    num_layers=6,
    num_heads=8,
    max_length=512,
    latent_dim=256,
    dropout=0.1,
    attention_type='GQA',  # or 'MHA' for Multi-Head Attention
    flash_attention=True  # Set to True if using Flash Attention
).cuda()  # Move model to GPU if available

# Example input
input_tensor = torch.randint(0, 10000, (32, 50)).cuda()  # Batch size of 32, sequence length of 50
latent_tensor = torch.randn((32, 50, 256)).cuda()  # Batch size of 32, latent dimension of 256
output = model(input_tensor, latent_tensor)
print(output.shape)  # Should be (32, 50, 10000) for
"""
