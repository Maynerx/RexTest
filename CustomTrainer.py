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
        self._save_trigger = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Actual batch from current step
        inputs = kwargs.get("inputs", None)
        if inputs is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            batch_tokens = input_ids.numel()
        else:
            batch_tokens = 0  # fallback if we can't access input_ids

        self.tokens_seen += batch_tokens
        self._save_trigger += batch_tokens

        # Checkpointing
        if self._save_trigger >= self.save_every_tokens:
            control.should_save = True
            self._save_trigger = 0

        # Stop condition
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
                 save_strategy: str = "steps"
    ):
        # 1) Build TrainingArguments
        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            eval_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_every_tokens,    # used if save_strategy != "no"
            logging_steps=logging_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            num_train_epochs=num_train_epochs,
            torch_compile=True,                      # Enable compilation
            torch_compile_backend="inductor",        # Choose backend
            torch_compile_mode="default",    # Choose compile mode
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

    def prediction_step(
        self,
        model,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        This method is called for each batch during evaluation.
        We pop out the keys our model doesn’t expect and pass them as positional args.
        """
        # 1) Prepare inputs
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        src = inputs.pop("input_ids")
        tgt = src.clone()  # or inputs.pop("labels"), depending on signature
        # 3) Do a forward pass
        with torch.no_grad():
            logits = model(src, tgt)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="mean"
            )
        # 4) If only loss is requested:
        if prediction_loss_only:
            return (ce_loss, None, None)

        # 5) Return (loss, logits, labels) as Trainer expects
        return (ce_loss, logits, labels)

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
        inputs_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        top_k_probs   = inputs.pop("top_k_probs")
        top_k_indices = inputs.pop("top_k_indices")

        # Forward
        outputs = model(inputs_ids, inputs_ids)
        logits = outputs

        # Cross‑entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean"
        )

        model_topk_logits = logits.gather(-1, top_k_indices)  # [B, S, k]

        # Compute log softmax over top-k
        log_probs_topk = F.log_softmax(model_topk_logits / self.temperature, dim=-1)  # [B, S, k]
        
        # KL divergence only over top-k
        kl_loss = F.kl_div(
            log_probs_topk,
            top_k_probs / self.temperature,  # teacher top-k probs
            reduction="batchmean"
        ) * (self.temperature ** 2)

        loss = self.alpha * ce_loss + self.beta * kl_loss

        return (loss, {"logits": logits, "labels": labels}) if return_outputs else loss
