"""Fully connected networks used as Q-functions, policies and value functions."""

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """A simple multi-layer perceptron.

    Args:
        in_dim: Size of the input vector (e.g. observation dimension).
        out_dim: Size of the output vector (e.g. number of actions).
        hidden_sizes: Width of each hidden layer.
        activation: Activation module class applied after each hidden layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), activation()]
            prev = size
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
