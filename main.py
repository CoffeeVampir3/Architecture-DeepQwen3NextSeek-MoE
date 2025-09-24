import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from utils.trainutils import count_parameters_layerwise, save_checkpoint, TBLogger
from modeling.model import MoEModel
from modeling.model_config import ModelConfig
from cut_cross_entropy import linear_cross_entropy

from modeling.zRMSNorm import ZeroCenteredRMSNorm

#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx]}

def load_and_preprocess_data(max_length=510):
    dataset = load_dataset("skeskinen/TinyStories-hf", split="train[:1%]") #[:1%]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_input_ids = []
    for text in tqdm(dataset["text"]):
        text = text.strip()
        if not text:
            continue
        tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
        if len(tokens) >= max_length:
            continue
        sequence = tokens
        if sequence.size(0) < max_length:
            padding = torch.full((max_length - sequence.size(0),), tokenizer.eos_token_id, dtype=torch.long)
            sequence = torch.cat([sequence, padding])
        all_input_ids.append(sequence)

    input_ids_tensor = torch.stack(all_input_ids)
    return TextDataset(input_ids_tensor), tokenizer

# Auxillary loss free routing: https://arxiv.org/abs/2408.15664
# TLDR is that this does a direct update (not via backward) to the expert biases which will tip towards underfilled experts.
def auxillary_loss_free_update(model, all_topk_indices, update_rate):
    with torch.no_grad():
        for layer_idx, topk_idx in enumerate(all_topk_indices):
            expert_counts = torch.bincount(
                topk_idx.flatten(),
                minlength=model.layers[layer_idx].mlp.gate.n_routed_experts
            )
            avg_count = expert_counts.float().mean()
            for expert_idx, count in enumerate(expert_counts):
                error = avg_count - count.float()
                model.layers[layer_idx].mlp.gate.expert_biases[expert_idx] += update_rate * torch.sign(error)

def build_weight_decay_optm(model, learning_rate):
    zero_centered_rmsnorm_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = model.get_submodule('.'.join(name.split('.')[:-1]))

        if isinstance(module, ZeroCenteredRMSNorm):
            zero_centered_rmsnorm_params.append(param)
        elif any(exclude in name for exclude in [
            'bias', 'embedding', 'output_layer', 'norm.weight',
            'layernorm', 'dt_bias', 'A_log', 'expert_biases'
        ]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.Adam([
        {'params': zero_centered_rmsnorm_params, 'weight_decay': 1e-4},
        {'params': decay_params, 'weight_decay': 3e-3},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, eps=1e-16)

import math
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps=1000, total_steps=None, peak_lr=2e-6, min_lr=8e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        if self.step_count < self.warmup_steps:
            lr = self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1
        return lr

# Used for control testing
class LinearLRScheduler:
    def __init__(self, optimizer, warmup_steps=1000, total_steps=None, peak_lr=2e-6):
        self.peak_lr = peak_lr

    def step(self):
        return self.peak_lr

def train(model, train_dataset, tokenizer, num_epochs=1, batch_size=32, learning_rate=1e-4, update_rate=4e-5):
    device = torch.device("cuda")
    model.to(device)

    optimizer = build_weight_decay_optm(model, learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scaler = torch.amp.GradScaler("cuda")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = TBLogger(log_dir='logs/moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    global_step = 0

    total_steps = len(train_loader) * num_epochs
    scheduler = LinearLRScheduler(optimizer, warmup_steps=2000, total_steps=total_steps, peak_lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings, all_topk_indices = model.headless_forward(input_ids)
                classifier = model.get_classifier_weights()
                loss = linear_cross_entropy(embeddings, classifier, input_ids, shift=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            total_loss += loss.item()
            auxillary_loss_free_update(model, all_topk_indices, update_rate)

            metrics = logger.log_training_metrics(loss, optimizer, update_rate, global_step, epoch, batch_idx)
            metrics.update(logger.log_moe_metrics(all_topk_indices, global_step))

            detailed_logging = (global_step % logger.detailed_frequency == 0)
            logger.log(metrics, step=global_step, model=model, detailed_logging=detailed_logging)

            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            'loss/epoch_loss': avg_loss,
            'training/epoch_time': epoch_time,
            'training/batches_per_epoch': len(train_loader),
            'training/tokens_per_second': len(train_loader) * batch_size * 510 / epoch_time
        }
        logger.log(epoch_metrics, step=global_step, detailed_logging=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.safetensors"
        save_checkpoint(model, optimizer, str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")

    logger.close()

def main():
    config = ModelConfig()
    train_dataset, tokenizer = load_and_preprocess_data()

    model = MoEModel(config)

    count_parameters_layerwise(model)
    torch.compile(model, mode="max-autotune")
    train(model, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
