[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

Combination of MLA and Gated Delta Net using Deep Seek's routing with a general large MoE. The early MoE doesn't stabilize very well and sadly we probably need an auxillary loss to stabilize the early training. Should probably gate the MLA here.

Architecture:
- Deep seek style MoE (Auxillary loss free routing -- Different from Qwen's usual routing: https://arxiv.org/abs/2408.15664)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Latent Multi-Head Attention (https://arxiv.org/abs/2412.19437 and https://arxiv.org/abs/2506.09342)
- Gated Delta Net (Using FLA https://arxiv.org/abs/2412.06464)
- Gated Attention (G1 per head variant specifically -- https://arxiv.org/abs/2505.06708)

Auxillary stuff:
- Cut cross entropy training (https://arxiv.org/abs/2411.09009)

### Do the thing
Using uv:
```
uv sync
```

Train (trains on TinyStories-hf for 1 epochs by default)
```
uv run python main.py
```

Infer (hard coded to use checkpoint 1):
```
uv run python basic_inf.py
```

Logs to: `logs/moe_training`
```
tensorboard --logdir=logs/moe_training --reload_multifile=true --reload_interval=15
```














