"""Shared utilities for the diffusion tutorial notebooks.

Contains:
- load_swiss_roll: standardized 2D swiss-roll dataset.
- Residual, SinusoidalTimeEmbedding, TinyMLP: tiny MLP backbone used by both
  flow matching and DDPM. ``time_dim=0`` concatenates scalar t directly
  (flow-matching default); ``time_dim>0`` uses a sinusoidal embedding (DDPM
  default). ``x_prediction=True`` makes the net predict x and return the
  derived velocity (flow matching only).
- CondMLP: time- and label-conditioned MLP for conditional flow matching.
  Supports pred_type="v" (velocity) and pred_type="x" (clean data / x-pred).
- train: generic training loop used by both notebooks.
- cosine_beta_schedule, extract: DDPM noise schedule helpers.
- load_or_train: crash-safe checkpoint helper.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
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


class CondMLP(nn.Module):
    """Time- and label-conditioned MLP for conditional flow matching.

    pred_type="v"  (velocity prediction, default in standard CFM):
        Network predicts the velocity  v = e − x.
        Training loss: ||v̂ − (e − x)||²

    pred_type="x"  (data / x-prediction):
        Network predicts the clean data x directly.
        Training loss: ||x̂ − x||²
        Implicitly down-weights large-noise timesteps (t → 1), which can
        help the model focus on the signal-rich region near t = 0 and
        produce sharper samples on complex distributions.

    ``forward()`` always returns the *raw* network output (v̂ or x̂).
    Use ``loss_target(x, e)`` for the training objective and
    ``to_velocity(raw, z, t)`` to convert the output to velocity for
    the Euler sampler — so sample_cfg works identically for both modes.

    Label index ``n_classes`` is the null token ∅ used for CFG dropout
    and unlabeled training points.
    """

    def __init__(self, num_in, hidden, n_classes, time_dim=32, cond_dim=32,
                 n_residual=4,  x_prediction=False):
        super().__init__()
        self.x_prediction   = x_prediction
        self.n_classes   = n_classes
        self.null_idx    = n_classes
        self.time_embed  = SinusoidalTimeEmbedding(time_dim)
        self.label_embed = nn.Embedding(n_classes + 1, cond_dim)
        in_features = num_in + time_dim + cond_dim
        layers  = [nn.Linear(in_features, hidden), nn.GELU()]
        layers += [Residual(hidden) for _ in range(n_residual)]
        layers += [nn.GELU(), nn.Linear(hidden, num_in)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, t, y):
        """Raw network output: v̂ (pred_type='v') or x̂ (pred_type='x')."""
        o = self.mlp(torch.cat([z, self.time_embed(t), self.label_embed(y)], dim=1))
        if self.x_prediction:
            return (z - o) / t
        return o    


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


# ── DDPM schedule helpers ─────────────────────────────────────────────────────

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule from 'Improved DDPM' (https://arxiv.org/abs/2102.09672)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    """Gather schedule values at timestep t and broadcast to x_shape."""
    out = a.gather(-1, t.cpu())
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(t.device)


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def load_or_train(model, name, train_step, data, niter, lr, ckpt_dir, 
                  batch_size=1000, load=True, save=True):
    """Load model + smoothed losses from a checkpoint, or train and save.

    Args:
        model: the nn.Module to train or load into.
        name: checkpoint filename stem (saved as ``ckpt_dir/<name>.pt``).
        train_step: callable ``(x, net, optimizer) -> loss``.
        data: full training tensor (moved to device inside ``train``).
        niter: number of training iterations.
        lr: learning rate for AdamW.
        ckpt_dir: directory for checkpoint files (created if needed).
        batch_size: training batch size.
        load: if True and checkpoint exists, load instead of training.
        save: if True after training, persist weights + losses to disk.
    Returns:
        losses: smoothed loss array of shape ``(niter,)``.
    """
    ckpt_path = Path(ckpt_dir) / f"{name}.pt"
    if load and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        losses = np.asarray(payload["losses"])
        print(f"Loaded {name} from {ckpt_path}")
        return losses
    model.train()
    losses = train(data, model, train_step, niter, lr, batch_size)
    losses = sliding_window_view(np.pad(losses, (10, 10), mode='reflect'), 21).mean(axis=1)
    if save:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "losses": losses.tolist()}, ckpt_path)
        print(f"Saved {name} to {ckpt_path}")
    return losses
