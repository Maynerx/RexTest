from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn

class TokenCounterCallback(TrainerCallback):
    def __init__(self, tokens_target: int, save_every_tokens: int = 100_000):
        self.tokens_seen = 0
        self.tokens_target = tokens_target
        self.save_every_tokens = save_every_tokens

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Estimate how many tokens were seen in this step
        # Assumes input_ids are the only thing being used
        train_batch_size = args.per_device_train_batch_size * args.world_size
        tokens_per_batch = args.max_seq_length * train_batch_size
        self.tokens_seen += tokens_per_batch

        # Save checkpoint every N tokens
        if self.tokens_seen >= self.save_every_tokens:
            control.should_save = True
            self.tokens_seen = 0

        # Stop training if we hit target
        if self.tokens_seen >= self.tokens_target:
            control.should_training_stop = True

        return control


class REXTrainer(Trainer):
    def __init__(self,
                 model,
                 training_dataset: Dataset,
                 eval_dataset: Dataset = None,
                 tokens_target: int = 1_000_000,
                 save_every_tokens: int = 100_000,
                 lr: float = 1e-4,
                 temperature: float = 1.0,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 weight_decay: float = 0.01,
                 per_device_train_batch_size: int = 8,
                 per_device_eval_batch_size: int = 8,
                 grad_accumulation_steps: int = 1,
                 num_train_epochs: int = 1,
                 logging_steps: int = 1000,
                 save_strategy: str = "no"
    ):
        # 1) Build TrainingArguments
        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            save_strategy=save_strategy,
            save_steps=save_every_tokens,    # used if save_strategy != "no"
            logging_steps=logging_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            fp16=True,
            remove_unused_columns=False,
            report_to="none",
        )

        # 2) Initialize Trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
        )

        # 3) Store hyperparams
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.tokens_target = tokens_target

        # 4) Add our token‑counter callback
        self.add_callback(TokenCounterCallback(tokens_target, save_every_tokens))

        # 5) Prepare KL loss
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Optional: custom optimizer/scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
        self.optimizer, self.lr_scheduler = optimizer, scheduler

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Move to device & unpack
        inputs = self._prepare_inputs(inputs)
        inputs_ids = inputs.pop("inputs_ids")
        labels = inputs.pop("labels")
        top_k_probs   = inputs.pop("top_k_probs")
        top_k_indices = inputs.pop("top_k_indices")

        # Forward
        outputs = model(inputs_ids)
        logits = outputs.logits

        # Cross‑entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean"
        )

        # Reconstruct teacher probs
        teacher_probs = torch.zeros_like(logits)
        teacher_probs.scatter_(
            dim=-1,
            index=top_k_indices,
            src=top_k_probs
        )

        # KL divergence
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss_fn(
            log_probs,
            teacher_probs / self.temperature
        ) * (self.temperature ** 2)

        loss = self.alpha * ce_loss + self.beta * kl_loss

        return (loss, outputs) if return_outputs else loss
