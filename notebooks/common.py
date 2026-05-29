"""Shared utilities for the diffusion tutorial notebooks.

Contains:
- load_swiss_roll: standardized 2D swiss-roll dataset.
- Residual, SinusoidalTimeEmbedding, TinyMLP: tiny MLP backbone used by both
  flow matching and DDPM. ``time_dim=0`` concatenates scalar t directly
  (flow-matching default); ``time_dim>0`` uses a sinusoidal embedding (DDPM
  default). ``x_prediction=True`` makes the net predict x and return the
  derived velocity (flow matching only).
- train: generic training loop used by both notebooks.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_swiss_roll
from torch import nn
from tqdm import tqdm


def load_swiss_roll(n_samples=10000, noise=0.0):
    """Return a standardized 2D swiss roll as a float tensor of shape (N, 2)."""
    raw = make_swiss_roll(n_samples=n_samples, noise=noise)[0][:, ::2]
    data = torch.tensor(raw).float()
    data = (data - data.mean(0)) / data.std(0)
    return data, raw


class Residual(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.lin = nn.Linear(num_in, num_in)
        self.norm = nn.LayerNorm(num_in)

    def forward(self, x):
        return F.gelu(self.norm(self.lin(x))) + x


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for continuous t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B, 1) float in [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t * 1000.0 * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TinyMLP(nn.Module):
    """Tiny MLP conditioned on time t (float in [0, 1]).

    Args:
        num_in: data dimensionality.
        hidden: hidden width.
        time_dim: 0 -> concat scalar t directly; >0 -> sinusoidal embedding.
        x_prediction: if True, the net predicts x and ``forward`` returns the
            implied velocity v = (z - x_pred) / t (flow matching only).
        n_residual: number of residual blocks in the trunk.
    """
    def __init__(self, num_in, hidden, time_dim=0, x_prediction=False, n_residual=4):
        super().__init__()
        self.time_dim = time_dim
        self.x_prediction = x_prediction
        if time_dim > 0:
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
            in_features = num_in + time_dim
        else:
            self.time_embed = None
            in_features = num_in + 1

        layers = [nn.Linear(in_features, hidden), nn.GELU()]
        layers += [Residual(hidden) for _ in range(n_residual)]
        layers += [nn.GELU(), nn.Linear(hidden, num_in)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, t):
        t_feat = self.time_embed(t) if self.time_embed is not None else t
        y = self.mlp(torch.cat((z, t_feat), dim=1))
        if self.x_prediction:
            return (z - y) / t
        return y


def train(data, net, train_step, niter, lr, batch_size=1000):
    """Generic training loop. ``train_step(x, net, optimizer) -> loss``."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    data = data.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    losses = []
    for _ in (pbar := tqdm(range(niter), ncols=100)):
        idx = torch.randperm(len(data))[:batch_size]
        x = data[idx].contiguous()
        loss = train_step(x, net, optim)
        losses.append(loss.item())
        pbar.set_description(f"loss = {loss.item():06f}")
    return losses
