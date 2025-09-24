import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock
from .gdn_block import GatedDeltaNetBlock

class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(config)] +
            [GatedDeltaNetBlock(config, layer_idx=i) for i in range(config.transformer_depth - 1)]
        )

        self.output_layer = nn.Linear(config.embed_size, config.vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = self.embedding(x)

        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            all_topk_indices.append(topk_idx)

        x = self.output_layer(x)
        return x, all_topk_indices

    def headless_forward(self, x, attention_mask=None, position_ids=None):
        x = self.embedding(x)

        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            all_topk_indices.append(topk_idx)

        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
