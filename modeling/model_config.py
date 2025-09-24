from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    embed_size: int = 256
    hidden_size: int = 256

    transformer_depth: int = 5

    # MoE
    intermediate_size: int = 256+64
    n_experts: int = 8
    n_shared_experts: int = 2
    n_routed_experts: int = n_experts - n_shared_experts
    n_experts_per_token = 3

    # Traditional Attention
    n_attention_heads: int = 16
    n_key_value_heads: int = 4

    # Gated Delta Net
    num_gdn_v_heads: int = 32
    num_gdn_k_heads: int = 16
    gdn_v_head_dim: int = 128
    gdn_k_head_dim: int = 128

    # Multihead Latent Attention
    num_heads: int = 8
    q_lora_rank: int = 128
    qk_rope_head_dim: int = 16
    kv_lora_rank: int = 128
    v_head_dim: int = 32
    qk_nope_head_dim: int = 16

    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1024
    rope_theta: int = 100000
    attention_dropout: float = 0.03
