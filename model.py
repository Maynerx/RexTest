import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


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
    
