import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_trajectory_general(sol_x, sol_y, regions, eta, title=None, n_boundary=700):
    """Plot one or multiple trajectories with region boundaries."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    if not isinstance(sol_x, list):
        sol_x = [sol_x]
        sol_y = [sol_y]

    for i, region in enumerate(regions):
        constraint_func = region["constraint"]
        x_min, x_max, y_min, y_max = region["bbox"]

        x_grid = np.linspace(x_min, x_max, n_boundary)
        y_grid = np.linspace(y_min, y_max, n_boundary)
        X, Y = np.meshgrid(x_grid, y_grid)

        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = np.array(constraint_func(jnp.array(points))).reshape(X.shape)

        color = "lightcoral" if i == eta else "lightblue"
        boundary_color = "darkred" if i == eta else "steelblue"
        ax.contourf(X, Y, Z.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.6)
        ax.contour(X, Y, Z.astype(float), levels=[0.5], colors=[boundary_color], linewidths=1.2)

    for j, (sol_x_i, sol_y_i) in enumerate(zip(sol_x, sol_y)):
        ax.plot(sol_x_i, sol_y_i, "k-", linewidth=0.8, alpha=0.6)
        if j == 0:
            ax.plot(sol_x_i[0], sol_y_i[0], "go", markersize=5, label="start")
            ax.plot(sol_x_i[-1], sol_y_i[-1], "rx", markersize=7, markeredgewidth=1.5, label="end")
        else:
            ax.plot(sol_x_i[0], sol_y_i[0], "go", markersize=5)
            ax.plot(sol_x_i[-1], sol_y_i[-1], "rx", markersize=7, markeredgewidth=1.5)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_density_comparison(samples_ode, samples_sde, bins=60):
    """Plot side-by-side 2D histograms for ODE and SDE endpoint clouds."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].hist2d(samples_ode[:, 0], samples_ode[:, 1], bins=bins, cmap="Blues")
    axes[0].set_title("ODE endpoints")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].hist2d(samples_sde[:, 0], samples_sde[:, 1], bins=bins, cmap="Oranges")
    axes[1].set_title("SDE endpoints")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    return fig, axes
