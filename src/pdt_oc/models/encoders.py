from typing import Sequence, Type

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_modality_encoder(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
) -> ModalityEncoder:
    return ModalityEncoder(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim)
