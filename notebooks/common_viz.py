import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def animate_flow(model, generate_samples_fn, train_set, n_samples=1000, n_steps=80, cmap='viridis',
                 title='Flow trajectories (color follows time)', max_curves=1000):
    """Static trajectory plot with time-colored curves.

    Args:
        model: trained model to visualize.
        generate_samples_fn: callable returning trajectories when called as
            ``generate_samples_fn(model, n_samples=..., n_steps=..., return_trajectories=True)``.
        train_set: target data tensor/array for the background scatter.
        n_samples: number of trajectories to draw.
        n_steps: number of solver steps used by the sampler.
        cmap: matplotlib colormap name.
        title: plot title.
        max_curves: cap on the number of trajectories rendered.
    """
    trajectories = generate_samples_fn(
        model,
        n_samples=n_samples,
        n_steps=n_steps,
        return_trajectories=True,
    )
    traj = np.asarray(trajectories)

    if traj.shape[1] > max_curves:
        idx = np.random.choice(traj.shape[1], max_curves, replace=False)
        traj = traj[:, idx]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    real_data = train_set.cpu().numpy() if hasattr(train_set, 'cpu') else train_set
    ax.scatter(real_data[:, 0], real_data[:, 1], c='black', alpha=0.12, s=4, label='target data')

    segments = []
    color_values = []
    for i in range(traj.shape[1]):
        pts = traj[:, i, :]
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        segments.append(seg)
        color_values.append(np.linspace(0.0, 1.0, pts.shape[0] - 1))

    segments = np.concatenate(segments, axis=0)
    color_values = np.concatenate(color_values, axis=0)

    lc = LineCollection(segments, cmap=cmap, linewidths=0.8, alpha=0.9)
    lc.set_array(color_values)
    ax.add_collection(lc)

    start_pts = traj[0]
    end_pts = traj[-1]
    ax.scatter(start_pts[:, 0], start_pts[:, 1], c='tab:blue', s=7, alpha=0.5, label='start (t=1)')
    ax.scatter(end_pts[:, 0], end_pts[:, 1], c='tab:red', s=7, alpha=0.5, label='end (t=0)')

    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('normalized denoising time')

    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', markerscale=1.5)
    plt.tight_layout()
    plt.show()