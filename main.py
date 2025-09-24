import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from utils.trainutils import count_parameters_layerwise, save_checkpoint, TBLogger
from modeling.model import MoEModel
from modeling.model_config import ModelConfig
from modeling.zRMSNorm import ZeroCenteredRMSNorm
from cut_cross_entropy import linear_cross_entropy

torch.set_float32_matmul_precision('high')

class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx]}

def load_and_preprocess_data(max_length=255, val_split=0.01):
    dataset = load_dataset("skeskinen/TinyStories-hf", split="train")
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
    dataset = TextDataset(input_ids_tensor)

    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset, tokenizer

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
        {'params': decay_params, 'weight_decay': 3e-4},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate)

def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings, _ = model.headless_forward(input_ids)
                classifier = model.get_classifier_weights()
                loss = linear_cross_entropy(embeddings, classifier, input_ids, shift=1)

            total_val_loss += loss.item()
            num_batches += 1

    return total_val_loss / num_batches

def train(model, train_dataset, val_dataset, tokenizer, num_epochs=1, batch_size=64, learning_rate=4e-5, update_rate=1e-4, validate_every=1000):
    device = torch.device("cuda")
    model.to(device)

    optimizer = build_weight_decay_optm(model, learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    scaler = torch.amp.GradScaler("cuda")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger = TBLogger(log_dir='logs/moe_training', detailed_frequency=20)
    logger.register_moe_layers(model)
    global_step = 0

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

            total_loss += loss.item()
            auxillary_loss_free_update(model, all_topk_indices, update_rate)

            metrics = logger.log_training_metrics(loss, optimizer, update_rate, global_step, epoch, batch_idx)
            metrics.update(logger.log_moe_metrics(all_topk_indices, global_step))

            if global_step % validate_every == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device)
                metrics['loss/validation_loss'] = val_loss
                print(f"Step {global_step}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                model.train()

            detailed_logging = (global_step % logger.detailed_frequency == 0)
            logger.log(metrics, step=global_step, model=model, detailed_logging=detailed_logging)

            global_step += 1

        avg_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            'loss/epoch_loss': avg_loss,
            'loss/epoch_validation_loss': val_loss,
            'training/epoch_time': epoch_time,
            'training/batches_per_epoch': len(train_loader),
            'training/tokens_per_second': len(train_loader) * batch_size * 255 / epoch_time
        }
        logger.log(epoch_metrics, step=global_step, detailed_logging=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.safetensors"
        save_checkpoint(model, optimizer, str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")

    logger.close()

def main():
    config = ModelConfig()
    train_dataset, val_dataset, tokenizer = load_and_preprocess_data()

    model = MoEModel(config)

    count_parameters_layerwise(model)
    torch.compile(model, mode="max-autotune")
    train(model, train_dataset, val_dataset, tokenizer)

if __name__ == "__main__":
    main()
