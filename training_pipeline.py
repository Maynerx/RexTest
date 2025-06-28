"""
TODO:
- Implement a Text class for handling text data.
- Implement the Trainer class for training models:
    - Load the teacher model.
    - Predict the logits using the teacher model.
    - Implement the training loop with mixed precision support.
    - May use DeepSpeed for distributed training or FSDP.
    - If assymetric training, use online distillation.
    - else use offline distillation + DeepSpeed ZeRO or FSDP for memory efficiency.
    - Implement advanced metrucs like BLEU, ROUGE, and perplexity.
- Implement the training pipeline:
    - Load the dataset.
    - Initialize the model. 
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm.auto as tqdm
import math

DEVICE1 = 'cuda:0'
DEVICE2 = 'cuda:1'
    
class Trainer:
    def __init__(self, 
                student_model,
                teacher_model,
                train_dataset: Dataset,
                val_dataset: Dataset, 
                batch_size=32, 
                learning_rate=5e-5,
                temperature=1.0,
                alpha=0.5,
                beta=0.5,
                grad_accumulation_steps=4
                ):
        self.model = student_model
        self.teacher_model = teacher_model
        self.model.to(DEVICE1)
        self.teacher_model.to(DEVICE2)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_amount_of_tokens = self.count_number_of_tokens()
        self.current_amount_of_tokens = 0
        self.end_training = False
        self.last_validation_perplexity = float('inf')
        self.temperature = temperature  # Temperature for distillation
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta  # Weight for cross-entropy loss
        self.grad_accumulation_steps = grad_accumulation_steps
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')
        self.scaler = torch.amp.GradScaler()  # For mixed precision training

        print(f"Total tokens in training dataset: {self.total_amount_of_tokens}")
        print(f"Total tokens in validation dataset: {self.count_number_of_tokens()}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f} M parameters")
        print(f"Teacher model has {sum(p.numel() for p in self.teacher_model.parameters()) / 1e6:.2f} M parameters")
        print("Trainer initialized with the following parameters:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha (distillation loss weight): {self.alpha}")
        print(f"  Beta (cross-entropy loss weight): {self.beta}")


    def compute_perplexity(self, loss):
        """Compute perplexity from loss"""
        return torch.exp(loss)

    def count_number_of_tokens(self):
        """Count the total number of tokens in the dataset"""
        total_tokens = 0
        for item in self.train_dataset:
            total_tokens += item['input_ids'].numel()

        for item in self.val_dataset:
            total_tokens += item['input_ids'].numel()
        return total_tokens

    def teacher_predict(self, ids):
        """Predict logits using the teacher model"""
        self.teacher_model.eval()
        with torch.no_grad():
            ids = ids.to(self.teacher_model.device)
            with torch.autocast(device_type='cuda'):
                logits = self.teacher_model(ids).logits
            teacher_probs = torch.log_softmax(logits / self.temperature, dim=-1)
        return teacher_probs

    def train_one_epoch(self):
        '''Assume that two GPU devices are available for training.'''
        self.model.train()
        self.teacher_model.eval()

        total_loss = 0.0
        count = 0
        accumlated_gradients = 0
        self.optimizer.zero_grad()
        for batch in self.train_loader:
            ids = batch['input_ids'].to(DEVICE2)
            labels = batch['labels'].to(DEVICE1)
            teacher_probs = self.teacher_predict(ids)
            teacher_probs = teacher_probs.cpu() # Offload to CPU to save GPU memory
            ids = ids.to(DEVICE1)  # Move ids to the student model's device
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                latent = self.model.encoder(ids)
                student_logits = self.model.decoder(ids, latent)
                loss_ce = self.criterion(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
                log_ps = torch.log_softmax(student_logits / self.temperature, dim=-1)
                loss_kl = self.kl_divergence(log_ps, teacher_probs.to(DEVICE1)) * self.temperature**2
                assert not math.isnan(loss_ce), f"CE loss is NaN: {loss_ce.item()}"
                assert not math.isnan(loss_kl), f"KL loss is NaN: {loss_kl.item()}"
                loss = self.alpha * loss_ce + self.beta * loss_kl
            self.scaler.scale(loss).backward()
            accumlated_gradients += 1
            total_loss += loss.item()
            self.current_amount_of_tokens += ids.numel()
            count += 1

            if accumlated_gradients % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            
            if self.current_amount_of_tokens % self.total_amount_of_tokens // 10 == 0:
                print(f'Current loss: {total_loss / count:.4f}, Tokens processed: {self.current_amount_of_tokens}/{self.total_amount_of_tokens}')

        if accumlated_gradients > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        torch.cuda.empty_cache()  # Clear cache to free up memory
        return total_loss / count

    def validate(self):
        """Validate the model on the validation dataset"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                ids = batch['input_ids'].to(DEVICE1)
                labels = batch['labels'].to(DEVICE1)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    latent = self.model.encoder(ids)
                    logits = self.model.decoder(ids, latent)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                count += 1
        torch.cuda.empty_cache()  # Clear cache to free up memory
        return total_loss / count
    
    def train(self, epochs):
        """Train the model for a specified number of epochs"""
        print(f'Starting training for {epochs} epochs...')
        for epoch in tqdm.tqdm(range(epochs)):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss = self.train_one_epoch()
            print(f'Train Loss: {train_loss:.4f}')
            val_loss = self.validate()
            val_ppl = self.compute_perplexity(torch.tensor(val_loss))
            print(f'Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}')
            
            # Check for early stopping
            if val_ppl < self.last_validation_perplexity:
                self.last_validation_perplexity = val_ppl
                self.end_training = False
            else:
                self.end_training = True
            
            if self.end_training:
                print('Early stopping triggered.')
                break


