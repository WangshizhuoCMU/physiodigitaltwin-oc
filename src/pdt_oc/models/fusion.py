import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, kv_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError((embed_dim, num_heads))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            kdim=kv_dim,
            vdim=kv_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, q_vec: torch.Tensor, kv_stack: torch.Tensor) -> torch.Tensor:
        query = q_vec.unsqueeze(1)
        out, _ = self.attn(query, kv_stack, kv_stack, need_weights=False)
        return out.squeeze(1)
