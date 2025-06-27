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
                attention_type: str = 'MHA'
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
            attention_type=attention_type
        )
        self.decoder = Decoder(
            dim=dim, 
            vocab_size=vocab_size, 
            num_layers=decoder_layers, 
            num_heads=num_heads, 
            max_length=max_length, 
            latent_dim=latent_dim, 
            dropout=dropout,
            attention_type=attention_type
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(src)
        output = self.decoder(tgt, latent)
        return output
    

# Test
"""
model = Transformer(
    dim=768, 
    vocab_size=10000, 
    enoder_layers=12,
    decoder_layers=6,
    num_heads=8, 
    max_length=32, 
    latent_dim=256, 
    dropout=0.1, 
    is_causal=False
).cuda()

src = torch.randint(0, 10000, (32, 32)).cuda()  # Batch of 32 sequences of length 512
tgt = torch.randint(0, 10000, (32, 32)).cuda()

print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M")  # Print number of trainable parameters

output = model(src, tgt)
print(output.shape)  # Should be [32, 512, 10000] for
"""