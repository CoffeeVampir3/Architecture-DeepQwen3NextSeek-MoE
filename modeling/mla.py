import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from .zRMSNorm import ZeroCenteredRMSNorm

# Some of this originated from https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention
# Substantially changed as that code had some bugs

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # small index, low freq; big index, high freq
        # make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len = max_position_embeddings,
            device = self.inv_freq.device,
            dtype = torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # use a different permutation to obtain same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q = rearrange(q, 'b h s (d_pairs two) -> b h s (two d_pairs)', two=2)
    k = rearrange(k, 'b h s (d_pairs two) -> b h s (two d_pairs)', two=2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GatedMLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        self.out_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False
        )
        self.full_gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank

        self.kv_lora_rank = config.kv_lora_rank

        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias = False,
        )
        self.q_down_norm = ZeroCenteredRMSNorm(self.q_lora_rank)

        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_down_norm = ZeroCenteredRMSNorm(self.kv_lora_rank)

        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )

        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (
                self.q_head_dim - config.qk_rope_head_dim + self.v_head_dim
            ),
            bias = False,
        )

        self.rotary_emb = DeepseekV2RotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.q_down_proj.weight)
        nn.init.orthogonal_(self.kv_down_proj.weight)

        nn.init.normal_(self.q_up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.kv_up_proj.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.out_proj.weight)

        nn.init.zeros_(self.full_gate_proj.weight)

    def forward(self, hidden_states, position_ids, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device)
            position_ids = repeat(position_ids, 'l -> b l', b=bsz)

        q = self.q_down_proj(
            hidden_states
        )
        q = self.q_down_norm(q)
        q = self.q_up_proj(q)
        # q shape: (b, s, num_heads * q_head_dim)
        q = rearrange(q, 'b s (h d) -> b h s d',
                      h=self.num_heads, d=self.q_head_dim)

        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1
        )

        c_kv = self.kv_down_proj(hidden_states)
        c_kv, k_rope = torch.split(
            c_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        ) # k_rope shape: (b, seq_len, self.qk_rope_head_dim)
        k_rope = rearrange(k_rope, 'b s qk_rope_head_dim -> b 1 s qk_rope_head_dim')

        kv = self.kv_down_norm(c_kv)
        kv = self.kv_up_proj(kv)
        # (b, seq, num_head * (self.q_head_dim - config.qk_rope_head_dim + self.v_head_dim))

        kv = rearrange(kv, 'b s (h kv_dim) -> b h s kv_dim',
                    h=self.num_heads,
                    kv_dim=self.qk_nope_head_dim + self.v_head_dim
        )

        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        ) # value_states shape (b, nums_head, seq_len, v_head_dim)

        kv_seq_len = q_len

        cos, sin = self.rotary_emb(
            q_rope, seq_len = kv_seq_len,
        )
        q_rope, k_rope = apply_rotary_pos_emb(
            q_rope, k_rope, cos, sin, position_ids,
        )

        # MHA
        query_states = torch.concat(
            [q_nope, q_rope], dim=-1
        )
        key_states = torch.concat(
            [k_nope, repeat(k_rope, 'b 1 s qk_rope_head_dim -> b h s qk_rope_head_dim', h=self.num_heads)],
            dim=-1
        )  # (b, 1, seq_len, dim)

        attn_weights = torch.matmul(
            query_states, rearrange(key_states, 'b h s head_dim -> b h head_dim s')
        )
        attn_weights = attn_weights / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        attn_weights = F.softmax(attn_weights, dim = -1).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training = self.training
        )
        attn_output = torch.matmul(
            attn_weights, value_states
            # (b, num_head, q_len, q_len)
            # value (b, nums_head, seq_len, v_head_dim)
        )

        attn_output = rearrange(attn_output, 'b h s d -> b s h d')
        gate_scores = torch.sigmoid(self.full_gate_proj(hidden_states))
        per_head_gate_scores = gate_scores.view(bsz, q_len, self.num_heads, self.v_head_dim)
        attn_output = attn_output * per_head_gate_scores

        attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        return self.out_proj(attn_output)
