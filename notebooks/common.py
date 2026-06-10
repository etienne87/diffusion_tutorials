"""Shared utilities for the diffusion tutorial notebooks.

Contains:
- load_swiss_roll: standardized 2D swiss-roll dataset.
- Residual, SinusoidalTimeEmbedding, TinyMLP: tiny MLP backbone used by both
  flow matching and DDPM. ``time_dim=0`` concatenates scalar t directly
  (flow-matching default); ``time_dim>0`` uses a sinusoidal embedding (DDPM
  default). ``x_prediction=True`` makes the net predict x and return the
  derived velocity (flow matching only).
- MeanFlowMLP: two-time backbone for Mean Flow experiments.
- CondMLP: time- and label-conditioned MLP for conditional flow matching.
  Supports pred_type="v" (velocity) and pred_type="x" (clean data / x-pred).
- train: generic training loop used by both notebooks.
- cosine_beta_schedule, extract: DDPM noise schedule helpers.
- load_or_train: crash-safe checkpoint helper.
- generate_samples_euler, generate_samples_ode, generate_samples_ddpm,
  generate_samples_mean_flow: shared toy-model samplers.
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
    def __init__(self, num_in, hidden, time_dim=0, x_prediction=False, n_residual=4, reverse_time=False):
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
        self.reverse_time = reverse_time 

    def forward(self, z, t):
        t_feat = self.time_embed(t) if self.time_embed is not None else t
        y = self.mlp(torch.cat((z, t_feat), dim=1))
        if self.x_prediction:
            if self.reverse_time:
                return (y - z) / (1 - t)
            else:
                return (z - y) / t
        return y


class MeanFlowMLP(nn.Module):
    """TinyMLP-style network conditioned on two times (t, r).

    time_dim=0 -> concat raw (t, r); >0 -> sinusoidal embedding of each.
    """

    def __init__(self, num_in, hidden, time_dim=0, n_residual=7):
        super().__init__()
        self.time_dim = time_dim
        if time_dim > 0:
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
            in_features = num_in + 2 * time_dim
        else:
            self.time_embed = None
            in_features = num_in + 2

        layers = [nn.Linear(in_features, hidden), nn.GELU()]
        layers += [Residual(hidden) for _ in range(n_residual)]
        layers += [nn.GELU(), nn.Linear(hidden, num_in)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, t, r):
        if self.time_embed is not None:
            t_feat = self.time_embed(t)
            r_feat = self.time_embed(r)
        else:
            t_feat, r_feat = t, r
        return self.mlp(torch.cat([z, t_feat, r_feat], dim=1))


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


class MeanFlowMLP(nn.Module):
    """TinyMLP-style network conditioned on two times (t, r).

    time_dim=0 -> concat raw (t, r); >0 -> sinusoidal embedding of each.
    """

    def __init__(self, num_in, hidden, time_dim=0, n_residual=7):
        super().__init__()
        self.time_dim = time_dim
        if time_dim > 0:
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
            in_features = num_in + 2 * time_dim
        else:
            self.time_embed = None
            in_features = num_in + 2

        layers = [nn.Linear(in_features, hidden), nn.GELU()]
        layers += [Residual(hidden) for _ in range(n_residual)]
        layers += [nn.GELU(), nn.Linear(hidden, num_in)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, t, r):
        if self.time_embed is not None:
            t_feat = self.time_embed(t)
            r_feat = self.time_embed(r)
        else:
            t_feat, r_feat = t, r
        return self.mlp(torch.cat([z, t_feat, r_feat], dim=1))


def _coerce_train_step_output(out):
    """Normalize train_step output to ``(loss, stats_dict_or_none)``.

    Supported train_step returns:
    - loss tensor/scalar
    - (loss, stats_dict)
    """
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("train_step tuple output must be (loss, stats_dict)")
        loss, stats = out
        if stats is not None and not isinstance(stats, dict):
            raise TypeError("train_step stats must be a dict or None")
        return loss, stats
    return out, None


def _to_serializable(value):
    """Convert tensors/arrays/scalars in stats dict to plain Python types."""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.reshape(-1).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, bool, str)):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def train(data, net, train_step, niter, lr, batch_size=1000, return_stats=False):
    """Generic training loop.

    ``train_step(x, net, optimizer)`` may return:
    - loss
    - (loss, stats_dict)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    data = data.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    losses = []
    stats_hist = [] if return_stats else None
    for _ in (pbar := tqdm(range(niter), ncols=100)):
        idx = torch.randperm(len(data))[:batch_size]
        x = data[idx].contiguous()
        step_out = train_step(x, net, optim)
        loss, stats = _coerce_train_step_output(step_out)
        losses.append(loss.item())
        if return_stats:
            stats_hist.append({k: _to_serializable(v) for k, v in (stats or {}).items()})
        pbar.set_description(f"loss = {loss.item():06f}")
    if return_stats:
        return losses, stats_hist
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
                  batch_size=1000, load=True, save=True, return_stats=False):
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
        return_stats: if True, also return per-iteration stats dictionaries
            emitted by ``train_step``.
    Returns:
        losses: smoothed loss array of shape ``(niter,)``.
        stats (optional): list[dict], one entry per training iteration.
    """
    ckpt_path = Path(ckpt_dir) / f"{name}.pt"
    if load and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        losses = np.asarray(payload["losses"])
        print(f"Loaded {name} from {ckpt_path}")
        if return_stats:
            return losses, payload.get("stats", [])
        return losses
    model.train()
    train_out = train(data, model, train_step, niter, lr, batch_size, return_stats=return_stats)
    if return_stats:
        losses, stats = train_out
    else:
        losses = train_out
    losses = sliding_window_view(np.pad(losses, (10, 10), mode='reflect'), 21).mean(axis=1)
    if save:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": model.state_dict(), "losses": losses.tolist()}
        if return_stats:
            payload["stats"] = stats
        torch.save(payload, ckpt_path)
        print(f"Saved {name} to {ckpt_path}")
    if return_stats:
        return losses, stats
    return losses

# ── Sample Generators ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples_euler(model, n_samples=1000, n_steps=1000, return_trajectories=False):
    """Euler sampler for time-conditioned flow models."""
    device = next(model.parameters()).device
    z = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps

    trajectories = []
    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            val = i * dt if model.reverse_time else 1 - i * dt
            t = torch.full((n_samples, 1), val, device=device)
            v = model(z, t)
            if model.reverse_time:
                z = z + v * dt
            else:
                z = z - v * dt
            if return_trajectories:
                trajectories.append(z.cpu().numpy())
    if return_trajectories:
        return trajectories
    return z.cpu()


@torch.no_grad()
def generate_samples_ode(model, n_samples=1000, method='RK45', rtol=1e-2, atol=1e-3,
                         return_trajectories=False, n_eval=100):
    """ODE sampler for time-conditioned flow models."""
    from scipy.integrate import solve_ivp
    

    device = next(model.parameters()).device
    z0 = torch.randn(n_samples, 2, device=device)
    model.eval()

    t_span = [1.0, 0.00001] if not model.reverse_time else [0.00001, 1.0]
    t_eval = np.linspace(t_span[0], t_span[1], n_eval) if return_trajectories else None

    with torch.no_grad():
        def ode_func(t, z_flat):
            z = torch.from_numpy(z_flat.reshape(n_samples, 2)).float().to(device)
            t_tensor = torch.full((n_samples, 1), t, device=device)
            v = model(z, t_tensor)
            return v.cpu().numpy().flatten()

        solution = solve_ivp(
            ode_func,
            t_span=t_span,
            y0=z0.cpu().numpy().flatten(),
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
        )

    if return_trajectories:
        traj = solution.y.T.reshape(n_eval, n_samples, 2)
        return [traj[i] for i in range(n_eval)]

    z_final = solution.y[:, -1].reshape(n_samples, 2)
    return torch.from_numpy(z_final)


@torch.no_grad()
def generate_samples_ddpm(model, betas, alphas_cumprod, n_samples=1000,
                          n_steps=100, return_trajectories=False, x0_clip=3.0):
    """Deterministic DDIM sampler for DDPM-style models."""
    T = len(betas)
    ts = np.round(np.linspace(T - 1, 0, n_steps)).astype(int)
    device = next(model.parameters()).device

    x = torch.randn(n_samples, 2, device=device)
    trajectories = []
    model.eval()
    with torch.no_grad():
        for i, t_idx in enumerate(ts):
            ac_t = alphas_cumprod[t_idx].to(device)
            ac_prev = (alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)).to(device)
            t_float = torch.full((n_samples, 1), float(t_idx) / T, device=device)
            eps_pred = model(x, t_float)
            x0_pred = ((x - (1 - ac_t).sqrt() * eps_pred) / ac_t.sqrt()).clamp(-x0_clip, x0_clip)
            x = ac_prev.sqrt() * x0_pred + (1 - ac_prev).sqrt() * eps_pred
            if return_trajectories:
                trajectories.append(x.cpu().numpy())
    if return_trajectories:
        return trajectories
    return x.cpu()


@torch.no_grad()
def generate_samples_mean_flow(model, n_samples=1000, n_steps=1, return_trajectories=False):
    """K-step sampler for Mean Flow models with inputs (z, t, r)."""
    device = next(model.parameters()).device
    z = torch.randn(n_samples, 2, device=device)
    t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    trajectories = [z.cpu().numpy()] if return_trajectories else None
    for i in range(n_steps):
        t = torch.full((n_samples, 1), float(t_vals[i]), device=device)
        r = torch.full((n_samples, 1), float(t_vals[i + 1]), device=device)
        u = model(z, t, r)
        z = z - (t - r) * u
        if return_trajectories:
            trajectories.append(z.cpu().numpy())
    if return_trajectories:
        return trajectories
    return z.cpu()
