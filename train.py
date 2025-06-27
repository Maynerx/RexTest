import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path

from transformers import GPT2TokenizerFast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Transformer  # Your seq2seq model
import json
import torch.amp as amp
import time
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

# 1. Initialize a real GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ensure there's a pad token

# 2. Dataset for tokenizer-backed language modeling
class TextDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer):
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        self.token_count = tokens.size(0)
        self.seq_len = seq_len
        # create sliding windows
        self.examples = []
        for i in range(0, tokens.size(0) - seq_len):
            self.examples.append(tokens[i : i + seq_len + 1])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        seq = self.examples[idx]
        return {
            "input_ids": seq[:-1],
            "labels":    seq[1:]
        }

# 3. Load Tiny Shakespeare and split
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text[:len(text) // 10]

split = int(0.9 * len(text))
train_text = text[:split]
val_text   = text[split:]

with open("config.json", "r") as f:
    config = json.load(f)

seq_len    = config['max_length']  # Use max_length from config
batch_size = config['batch_size']  # Use batch_size from config

train_ds = TextDataset(train_text, seq_len, tokenizer)
val_ds   = TextDataset(val_text, seq_len, tokenizer)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)

print(f"Train dataset size: {train_ds.token_count} tokens")
print(f"Validation dataset size: {val_ds.token_count} tokens")

# 4. Instantiate your Transformer seq2seq model
model = Transformer(
    dim=config['embedding_dim'],
    vocab_size=tokenizer.vocab_size,
    encoder_layers=config['encoder_layers'],
    decoder_layers=config['decoder_layers'],
    num_heads=config['num_heads'],
    max_length=seq_len,
    latent_dim=config['latent_dim'],
    dropout=config['dropout_rate'],
    attention_type=config['attention_type'],
).to("cuda")

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # no need for ignore if no padding

print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

def perplexity(loss):
    return torch.exp(loss)

# 5. Training loop

epochs = 3
for epoch in range(1, epochs + 1):
    model.train()
    scaler = GradScaler()
    start  = time.time()
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for this epoch
    total_loss = 0
    batch_count = 0
    c = 0
    temp_time_start = time.time()
    for batch in train_loader:
        input_ids = batch["input_ids"].to("cuda")
        labels    = batch["labels"].to("cuda")
        
        optimizer.zero_grad()
        # Forward through encoder + decoder
        with amp.autocast(device_type="cuda", dtype=torch.float16):
            latent = model.encoder(input_ids)       # [B, L, latent_dim]
            logits = model.decoder(input_ids, latent)  # [B, L, vocab_size]
            loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))

        
        
        scaler.scale(loss).backward()  # Scale the loss for mixed precision
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)          # Step the optimizer
        scaler.update()                 # Update the scaler
        total_loss += loss.item() * input_ids.size(0)
        batch_count += input_ids.size(0)
        temp_time = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(temp_time - temp_time_start))
        print('count:', c, 'total ', len(train_loader), "Percentage:", c / len(train_loader) * 100, formatted_time, "mem ", torch.cuda.memory_allocated() / 1e6,end='\r')
        torch.cuda.empty_cache()  # Clear cache to avoid OOM
        c += 1
    end = time.time()
    time_taken = end - start
    total_num_tokens = batch_count * seq_len
    throughput = total_num_tokens / time_taken  # tokens per second
    avg_train_loss = total_loss / len(train_ds)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to("cuda")
            labels    = batch["labels"].to("cuda")
            latent    = model.encoder(input_ids)
            logits    = model.decoder(input_ids, latent)
            loss      = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            val_loss += loss.item() * input_ids.size(0)
    avg_val_loss = val_loss / len(val_ds)
    
    print(f"Epoch {epoch} | Train Loss {avg_train_loss:.4f} | "
          f"Train PPL {perplexity(torch.tensor(avg_train_loss)):.2f} | "
          f"Val Loss {avg_val_loss:.4f} | "
          f"Val PPL {perplexity(torch.tensor(avg_val_loss)):.2f} |"
          f"Memory Usage: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB",
          f"Throughput: {throughput:.2f} tokens/sec")

