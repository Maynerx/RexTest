

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
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import gc

DEVICE1 = 'cuda:0'
DEVICE2 = 'cuda:1'
    
class Trainer:
    def __init__(self, 
                student_model,
                teacher_model,
                train_dataset: Dataset,
                val_dataset: Dataset,
                print_every: int = 5000, 
                num_epochs=10,
                batch_size=32, 
                learning_rate=5e-5,
                temperature=1.0,
                alpha=0.5,
                beta=0.5,
                grad_accumulation_steps=4
                ):
        self.model = student_model
        self.teacher_model = teacher_model
        """
        #self.model.to(DEVICE1)
        #self.teacher_model.to(DEVICE2)
        self.model = torch.compile(
            self.model,
            backend="inductor",       # default; good general‑purpose
            mode="max-autotune",      # autotune kernels for best throughput
            fullgraph=True  # fullgraph=True is needed for backward pass
        )
        self.teacher_model.half()
        self.teacher_model = torch.compile(
            self.teacher_model,
            backend="inductor",
            mode="max-autotune",
            fullgraph=False  # no backward, so no need for fullgraph
        )
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_amount_of_tokens = self.count_number_of_tokens()
        self.amount_of_tokens = 0
        self.current_amount_of_tokens = 0
        self.end_training = False
        self.last_validation_perplexity = float('inf')
        self.num_epochs = num_epochs
        self.temperature = temperature  # Temperature for distillation
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta  # Weight for cross-entropy loss
        self.grad_accumulation_steps = grad_accumulation_steps
        self.print_every = print_every
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size//grad_accumulation_steps, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accumulation_steps)
        total_steps    = steps_per_epoch * num_epochs
        warmup_steps   = int(0.1 * total_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
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

    def models_warmup(self, num_batches: int = 10):
        """
        Warm up both student and teacher models by running a few forward passes
        (for compilation, tracing & cache) prior to training.

        Args:
            num_batches (int): Number of batches to use for warmup (default: 10).
        """
        # Set correct modes
        self.model.eval()
        self.teacher_model.eval()

        it = iter(self.train_loader)
        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_batches)):
                ids = next(it)['input_ids'].to(DEVICE1)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = self.model(ids, ids)
                _ = self.teacher_model(ids.to(DEVICE2))
        print(f"Warmup complete: {num_batches} inference batches and 1 training batch.")
        torch.cuda.empty_cache()
        gc.collect()


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
            logits = self.teacher_model(ids).logits
            teacher_probs = teacher_probs = F.softmax(logits / self.temperature, dim=-1) 
        return teacher_probs

    def train_one_epoch(self):
        '''Assume that two GPU devices are available for training.'''
        self.model.train()
        self.teacher_model.eval()
        total_loss = 0.0
        accumlated_gradients = 0
        total_tokens = 0
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
            n_tokens = labels.numel()
            
            total_loss += loss_ce.item() * n_tokens
            total_tokens += n_tokens
            self.current_amount_of_tokens += ids.numel()

            if accumlated_gradients % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()  # Update learning rate
            
            
            

        if accumlated_gradients > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()  # Update learning rate after the epoch

        torch.cuda.empty_cache()  # Clear cache to free up memory
        return total_loss / total_tokens
    
    def train_continued(self):
        self.model.train()
        self.teacher_model.eval()
        accumlated_gradients = 0
        batch_count = 0
        cum_loss_ce = 0.0
        cum_tokens = 0
        self.optimizer.zero_grad()
        for batch in tqdm.tqdm(self.train_loader):
            ids = batch['input_ids'].to(DEVICE2)
            labels = batch['labels'].to(DEVICE1)
            teacher_probs = self.teacher_predict(ids)
            #teacher_probs = teacher_probs.cpu() # Offload to CPU to save GPU memory
            torch.compiler.cudagraph_mark_step_begin()
            ids = batch['input_ids'].to(DEVICE1)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                #latent = self.model.encoder(ids)
                student_logits = self.model(ids, ids) #self.model.decoder(ids, latent)
                loss_ce = self.criterion(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
                log_ps = torch.log_softmax(student_logits / self.temperature, dim=-1)
                loss_kl = self.kl_divergence(log_ps, teacher_probs.to(DEVICE1)) * self.temperature**2
                loss = self.alpha * loss_ce + self.beta * loss_kl
            self.scaler.scale(loss).backward()
            accumlated_gradients += 1
            n_tokens = labels.numel()
            batch_count += 1
            
            cum_loss_ce += loss_ce.item() * n_tokens
            cum_tokens  += n_tokens
            self.current_amount_of_tokens += ids.numel()

            if accumlated_gradients % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()  # Update learning rate
                accumlated_gradients = 0

            if batch_count % self.print_every == 0:
                avg_ce = cum_loss_ce / cum_tokens
                val_loss = self.validate()
                print(f"[step {batch_count}] train CE={avg_ce:.4f}  val CE={val_loss:.4f}  Perplexity={self.compute_perplexity(torch.tensor(val_loss)):.4f}  "
                    f"tokens={self.current_amount_of_tokens}/{self.total_amount_of_tokens} ({(self.current_amount_of_tokens / self.total_amount_of_tokens):.2%})")
                cum_loss_ce = cum_tokens = 0
        
        if accumlated_gradients > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()


        

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                ids = batch["input_ids"].to(DEVICE1)
                labels = batch["labels"].to(DEVICE1)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(ids, ids)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()
                # free tensors immediately
                del logits, loss
                torch.cuda.empty_cache()
                gc.collect()
        return total_loss / total_tokens
    
    def save_model(self, path):
        """Save the model to the specified path"""
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
    
    def train(self):
        """Train the model for a specified number of epochs"""
        epochs = self.num_epochs
        self.models_warmup(num_batches=10)
        print(f'Starting training for {epochs} epochs...')
        for epoch in tqdm.tqdm(range(epochs)):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss = self.train_one_epoch()
            torch.cuda.empty_cache()
            gc.collect()
            print(f'Train Loss: {train_loss:.4f}')
            val_loss = self.validate()
            val_ppl = self.compute_perplexity(torch.tensor(val_loss))
            print(f'Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}')

            
            """
            # Check for early stopping
            if val_ppl < self.last_validation_perplexity:
                self.last_validation_perplexity = val_ppl
                self.end_training = False
            else:
                self.end_training = True
            """
            
            if self.end_training:
                self.save_model(f'model_epoch_{epoch + 1}.pt')
                print('Early stopping triggered.')
                break
        print('Training complete.')
        self.save_model('final_model.pt')


    def fit(self):
        #self.models_warmup(num_batches=10)
        print('Starting training...')
        self.train_continued()
        print('Training complete.')
        self.save_model('final_model.pt')
