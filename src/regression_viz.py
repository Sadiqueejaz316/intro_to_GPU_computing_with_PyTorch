"""Visualisation helpers for the Linear Regression and PCA notebook."""

import numpy as np
import matplotlib.pyplot as plt


def plot_regression(x_vals, y_vals, a, b):
    """Scatter plot of observations with the least-squares regression line."""
    x_line = np.linspace(min(x_vals) - 2, max(x_vals) + 2, 100)
    y_line = a * x_line + b

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_vals, y_vals, color="steelblue", zorder=3, label="Measurements")
    ax.plot(x_line, y_line, color="tomato",
            label=f"f(x) = {a:.2f}x + {b:.2f}")
    ax.set_xlabel("Age")
    ax.set_ylabel("Systolic blood pressure")
    ax.set_title("Linear Regression — Least Squares")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_benchmark(sizes, cpu_times, gpu_times, gpu_type, title):
    """Log-log plot of CPU vs GPU timing with speedup summary.

    Parameters
    ----------
    sizes : list[int]
        Problem sizes (x-axis).
    cpu_times, gpu_times : list[float]
        Median times in seconds.  *gpu_times* may be ``None`` if no GPU.
    gpu_type : str
        Device type string (e.g. ``"cuda"``).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, [t * 1000 for t in cpu_times], "o-", label="CPU")
    if gpu_times is not None:
        ax.plot(sizes, [t * 1000 for t in gpu_times], "s-",
                label=f"GPU ({gpu_type})")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Median time (ms)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

    if gpu_times is not None:
        print("Speedup (CPU / GPU):")
        for s, ct, gt in zip(sizes, cpu_times, gpu_times):
            print(f"  n={s:>7,}:  {ct/gt:>6.2f}x")


def plot_pca(data_np, data_mean, eigenvalues, eigenvectors):
    """Scatter plot of 2-D data with principal component arrows.

    Parameters
    ----------
    data_np : ndarray, shape (N, 2)
        Raw 2-D observations.
    data_mean : Tensor, shape (2,)
        Per-column mean (used as arrow origin).
    eigenvalues : Tensor, shape (2,)
        Sorted eigenvalues (largest first).
    eigenvectors : Tensor, shape (2, 2)
        Corresponding eigenvectors as columns.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data_np[:, 0], data_np[:, 1], alpha=0.3, s=10, color="steelblue")

    origin = data_mean.numpy()
    colors = ["tomato", "forestgreen"]
    for i in range(2):
        vec = eigenvectors[:, i].numpy()
        scale = eigenvalues[i].sqrt().item()
        ax.quiver(origin[0], origin[1], vec[0] * scale, vec[1] * scale,
                  angles="xy", scale_units="xy", scale=1,
                  color=colors[i], width=0.015,
                  label=f"PC{i+1} (\u03bb={eigenvalues[i]:.2f})")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("PCA — Principal Components")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
