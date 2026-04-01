"""Visualisation helpers for the Introduction to PyTorch notebook."""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_sin_cos():
    """Plot sin and cos curves over [0, 2*pi]."""
    t = torch.linspace(0, 2 * math.pi, 200)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t.numpy(), torch.sin(t).numpy(), label="sin")
    ax.plot(t.numpy(), torch.cos(t).numpy(), label="cos")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True)
    ax.set_title("torch.sin / torch.cos")
    plt.tight_layout()
    plt.show()


def plot_dot_product(a, b):
    """Visualise the dot product of two 1-D tensors using the grid style.

    For each element pair shows highlighted cells in a, b and the resulting
    element-wise product.  A final row sums the products into the scalar result.
    """
    n = a.numel()
    products = a * b
    result = torch.dot(a, b)

    a_2d = a.reshape(1, n)
    b_2d = b.reshape(1, n)
    prod_2d = products.reshape(1, n)
    res_2d = result.reshape(1, 1)

    num_rows = n + 1  # one row per element + one summation row
    fig, axes = plt.subplots(
        num_rows, 7, figsize=(16, 2.0 * num_rows),
        gridspec_kw={"width_ratios": [n, 0.3, n, 0.3, n, 0.3, 1]},
    )

    # --- element-wise rows ---
    for i in range(n):
        ax_a, ax_op, ax_b, ax_eq, ax_p, ax_arr, ax_s = axes[i]

        hl = torch.zeros(1, n, dtype=torch.bool)
        hl[0, i] = True

        _draw_matrix(ax_a, a_2d, hl, "Blues")
        _draw_matrix(ax_b, b_2d, hl, "Oranges")
        _draw_matrix(ax_p, prod_2d, hl, "Purples")

        ax_op.text(0.5, 0.5, "\u2299", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_op.axis("off")
        ax_eq.text(0.5, 0.5, "=", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_eq.axis("off")
        ax_arr.axis("off")
        ax_s.axis("off")

        ax_p.set_xlabel(
            f"a[{i}]\u00b7b[{i}] = {a[i]:.0f}\u00b7{b[i]:.0f} = {products[i]:.0f}",
            fontsize=10,
        )

        if i == 0:
            ax_a.set_title("a", fontsize=13)
            ax_b.set_title("b", fontsize=13)
            ax_p.set_title("a \u2299 b", fontsize=13)

    # --- summation row ---
    ax_p2, ax_op2, ax_empty1, ax_eq2, ax_empty2, ax_arr2, ax_r = axes[n]

    hl_all = torch.ones(1, n, dtype=torch.bool)
    _draw_matrix(ax_p2, prod_2d, hl_all, "Purples")

    ax_op2.text(0.5, 0.5, "\u03a3", ha="center", va="center",
                fontsize=20, fontweight="bold")
    ax_op2.axis("off")
    ax_empty1.axis("off")
    ax_eq2.axis("off")
    ax_empty2.axis("off")
    ax_arr2.text(0.5, 0.5, "=", ha="center", va="center",
                 fontsize=18, fontweight="bold")
    ax_arr2.axis("off")

    hl_res = torch.ones(1, 1, dtype=torch.bool)
    _draw_matrix(ax_r, res_2d, hl_res, "Greens")

    terms = " + ".join(f"{p:.0f}" for p in products)
    ax_r.set_xlabel(f"{terms} = {result:.0f}", fontsize=10)

    plt.suptitle(
        f"Dot product:  a \u00b7 b = {result:.0f}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.show()


def plot_cross_product(v1, v2):
    """Visualise the cross product of two 3-D vectors using the grid style.

    Each row highlights which elements of v1 and v2 contribute to one
    element of the result, following the formula:
        r[0] = v1[1]*v2[2] - v1[2]*v2[1]
        r[1] = v1[2]*v2[0] - v1[0]*v2[2]
        r[2] = v1[0]*v2[1] - v1[1]*v2[0]
    """
    r = torch.linalg.cross(v1, v2)

    # Indices that contribute to each output element:
    #   r[i] = v1[pos_a]*v2[pos_b] - v1[neg_a]*v2[neg_b]
    contrib = [
        # (pos_a, pos_b, neg_a, neg_b)
        (1, 2, 2, 1),  # r[0]
        (2, 0, 0, 2),  # r[1]
        (0, 1, 1, 0),  # r[2]
    ]

    fig, axes = plt.subplots(
        3, 5, figsize=(14, 6.6),
        gridspec_kw={"width_ratios": [3, 0.4, 3, 0.4, 3]},
    )

    idx_labels = ["[0]", "[1]", "[2]"]

    for i, (pa, pb, na, nb) in enumerate(contrib):
        ax_v1, ax_op, ax_v2, ax_eq, ax_r = axes[i]

        # Highlight the two contributing elements in each input vector
        hl_v1 = torch.zeros(3, dtype=torch.bool)
        hl_v1[pa] = True
        hl_v1[na] = True

        hl_v2 = torch.zeros(3, dtype=torch.bool)
        hl_v2[pb] = True
        hl_v2[nb] = True

        hl_r = torch.zeros(3, dtype=torch.bool)
        hl_r[i] = True

        _draw_vector_by_index(ax_v1, v1, hl_v1, "Blues", labels=idx_labels)
        _draw_vector_by_index(ax_v2, v2, hl_v2, "Oranges", labels=idx_labels)
        _draw_vector_by_index(ax_r, r, hl_r, "Greens", labels=idx_labels)

        ax_op.text(0.5, 0.5, "\u00d7", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_op.axis("off")
        ax_eq.text(0.5, 0.5, "=", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_eq.axis("off")

        # Formula label — split across two lines to avoid clipping
        ax_r.set_xlabel(
            f"r[{i}] = v1[{pa}]\u00b7v2[{pb}] \u2212 v1[{na}]\u00b7v2[{nb}]\n"
            f"= {v1[pa]:.0f}\u00b7{v2[pb]:.0f} \u2212 {v1[na]:.0f}\u00b7{v2[nb]:.0f}"
            f" = {r[i]:.0f}",
            fontsize=10,
        )

        if i == 0:
            ax_v1.set_title("v1", fontsize=13)
            ax_v2.set_title("v2", fontsize=13)
            ax_r.set_title("v1 \u00d7 v2", fontsize=13)

    plt.suptitle(
        f"Cross product: v1 \u00d7 v2 = [{r[0]:.0f}, {r[1]:.0f}, {r[2]:.0f}]",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.show()


def _auto_fmt(t):
    """Pick a numeric format string for a tensor's values."""
    ft = t.float()
    if (ft - ft.round()).abs().max().item() < 0.005:
        return ".0f"
    return ".2f"


def _draw_matrix(ax, mat, highlight, cmap_name, fmt=".0f"):
    """Draw a matrix with *highlight* cells vivid and the rest dimmed."""
    rows, cols = mat.shape
    cmap = plt.colormaps[cmap_name]
    vals = mat.float()
    vmin, vmax = vals.min().item(), vals.max().item()
    if vmin == vmax:
        vmax = vmin + 1

    for i in range(rows):
        for j in range(cols):
            norm = (vals[i, j].item() - vmin) / (vmax - vmin)
            color = cmap(0.3 + 0.5 * norm)
            hl = bool(highlight[i, j])
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=(*color[:3], 1.0 if hl else 0.12),
                edgecolor="black" if hl else "lightgray",
                linewidth=2.0 if hl else 0.5,
            )
            ax.add_patch(rect)
            ax.text(
                j, i, f"{mat[i, j]:{fmt}}",
                ha="center", va="center", fontsize=14,
                fontweight="bold" if hl else "normal",
                color="black" if hl else "lightgray",
            )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _draw_vector_by_index(ax, vec, highlight, cmap_name, labels=None):
    """Draw a 1×n vector with cell colour determined by column index.

    Unlike ``_draw_matrix`` (which maps *values* through a colormap),
    this maps the *position index* so that each slot gets a distinct
    colour regardless of the stored value.
    """
    n = vec.numel()
    cmap = plt.colormaps[cmap_name]

    for j in range(n):
        color = cmap(0.3 + 0.5 * j / max(n - 1, 1))
        hl = bool(highlight[j])
        rect = plt.Rectangle(
            (j - 0.5, -0.5), 1, 1,
            facecolor=(*color[:3], 1.0 if hl else 0.12),
            edgecolor="black" if hl else "lightgray",
            linewidth=2.0 if hl else 0.5,
        )
        ax.add_patch(rect)
        ax.text(
            j, 0, f"{vec[j]:.0f}",
            ha="center", va="center", fontsize=14,
            fontweight="bold" if hl else "normal",
            color="black" if hl else "lightgray",
        )
        if labels is not None:
            ax.text(
                j, -0.85, labels[j],
                ha="center", va="top", fontsize=9, color="dimgray",
            )

    y_lo = -1.1 if labels is not None else -0.5
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y_lo, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_matmul(A, B):
    """Visualise A @ B, highlighting the contributing row/column for each output element."""
    C = A @ B
    m, k = A.shape
    _, n = B.shape
    num = m * n

    fig, axes = plt.subplots(
        num, 5, figsize=(14, 2.2 * num),
        gridspec_kw={"width_ratios": [k, 0.4, n, 0.4, n]},
    )
    if num == 1:
        axes = axes.reshape(1, -1)

    for idx, (ci, cj) in enumerate(
        [(i, j) for i in range(m) for j in range(n)]
    ):
        ax_a, ax_op, ax_b, ax_eq, ax_c = axes[idx]

        # Highlight row ci in A, column cj in B, element [ci,cj] in C
        hl_a = torch.zeros(m, k, dtype=torch.bool)
        hl_a[ci, :] = True
        hl_b = torch.zeros(k, n, dtype=torch.bool)
        hl_b[:, cj] = True
        hl_c = torch.zeros(m, n, dtype=torch.bool)
        hl_c[ci, cj] = True

        _draw_matrix(ax_a, A, hl_a, "Blues")
        _draw_matrix(ax_b, B, hl_b, "Oranges")
        _draw_matrix(ax_c, C, hl_c, "Greens")

        # Operator labels
        ax_op.text(0.5, 0.5, "@", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_op.axis("off")
        ax_eq.text(0.5, 0.5, "=", ha="center", va="center",
                   fontsize=18, fontweight="bold")
        ax_eq.axis("off")

        # Show the dot-product formula beneath C
        terms = " + ".join(
            f"{A[ci, kk]:.0f}\u00b7{B[kk, cj]:.0f}" for kk in range(k)
        )
        ax_c.set_xlabel(
            f"C[{ci},{cj}] = {terms} = {C[ci, cj]:.0f}", fontsize=10,
        )

        # Column titles on the first row only
        if idx == 0:
            ax_a.set_title("A", fontsize=13)
            ax_b.set_title("B", fontsize=13)
            ax_c.set_title("C = A @ B", fontsize=13)

    plt.suptitle("Matrix multiplication: element-by-element breakdown",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_broadcasting(col, row, result):
    """Visualise broadcasting of a column vector + row vector."""
    nrows, ncols = result.shape

    fig, axes = plt.subplots(
        1, 9, figsize=(18, 3),
        gridspec_kw={"width_ratios": [1, 0.3, ncols, 0.3, ncols, 0.3, ncols, 0.3, ncols]},
    )

    # Original column vector
    axes[0].imshow(col.numpy(), cmap="Oranges", alpha=0.6)
    for i in range(col.shape[0]):
        axes[0].text(0, i, f"{col[i, 0].item()}",
                     ha="center", va="center", fontsize=13)
    axes[0].set_title(f"col ({col.shape[0]}x1)", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plus sign
    axes[1].text(0.5, 0.5, "+",
                 ha="center", va="center", fontsize=22, fontweight="bold")
    axes[1].axis("off")

    # Original row vector
    axes[2].imshow(row.numpy(), cmap="Blues", alpha=0.6)
    for j in range(row.shape[1]):
        axes[2].text(j, 0, f"{row[0, j].item()}",
                     ha="center", va="center", fontsize=13)
    axes[2].set_title(f"row (1x{ncols})", fontsize=11)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Equals sign
    axes[3].text(0.5, 0.5, "=",
                 ha="center", va="center", fontsize=22, fontweight="bold")
    axes[3].axis("off")

    # Column vector broadcast
    col_expanded = col.expand(nrows, ncols)
    axes[4].imshow(col_expanded.numpy(), cmap="Oranges")
    for i in range(nrows):
        for j in range(ncols):
            axes[4].text(j, i, f"{col_expanded[i, j].item()}",
                         ha="center", va="center", fontsize=13)
    axes[4].set_title(f"col broadcast", fontsize=11)
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    # Plus sign
    axes[5].text(0.5, 0.5, "+",
                 ha="center", va="center", fontsize=22, fontweight="bold")
    axes[5].axis("off")

    # Row vector broadcast
    row_expanded = row.expand(nrows, ncols)
    axes[6].imshow(row_expanded.numpy(), cmap="Blues")
    for i in range(nrows):
        for j in range(ncols):
            axes[6].text(j, i, f"{row_expanded[i, j].item()}",
                         ha="center", va="center", fontsize=13)
    axes[6].set_title(f"row broadcast", fontsize=11)
    axes[6].set_xticks([])
    axes[6].set_yticks([])

    # Equals sign
    axes[7].text(0.5, 0.5, "=",
                 ha="center", va="center", fontsize=22, fontweight="bold")
    axes[7].axis("off")

    # Result
    axes[8].imshow(result.numpy(), cmap="Greens")
    for i in range(nrows):
        for j in range(ncols):
            axes[8].text(j, i, f"{result[i, j].item()}",
                         ha="center", va="center", fontsize=13)
    axes[8].set_title(f"result ({nrows}x{ncols})", fontsize=11)
    axes[8].set_xticks([])
    axes[8].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_slicing(x):
    """Visualise three common slicing patterns on a 2-D tensor."""
    nrows, ncols = x.shape

    slices = [
        ("x[:, 2] (column)", lambda: (slice(None), 2)),
        ("x[1:3, 1:4] (block)", lambda: (slice(1, 3), slice(1, 4))),
        ("x[::2] (every other row)", lambda: (slice(None, None, 2), slice(None))),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    for ax, (title, idx_fn) in zip(axes, slices):
        mask = torch.zeros(nrows, ncols)
        mask[idx_fn()] = 1

        ax.imshow(mask.numpy(), cmap="YlOrRd", vmin=0, vmax=1.5, alpha=0.6)
        for i in range(nrows):
            for j in range(ncols):
                ax.text(
                    j, i, f"{x[i, j].item()}", ha="center", va="center",
                    fontsize=11,
                    fontweight="bold" if mask[i, j] else "normal",
                    color="darkred" if mask[i, j] else "gray",
                )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))

    plt.tight_layout()
    plt.show()


def plot_reshape(x):
    """Visualise the same 1-D tensor in three different shapes."""
    n = x.numel()

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 2.5),
        gridspec_kw={"width_ratios": [n, n // 3, n // 4]},
    )

    # 1-D
    axes[0].imshow(x.unsqueeze(0).numpy(), cmap="viridis", aspect="auto")
    for j in range(n):
        axes[0].text(j, 0, str(x[j].item()), ha="center", va="center",
                     color="white", fontsize=11, fontweight="bold")
    axes[0].set_title(f"shape ({n},)", fontsize=11)
    axes[0].set_yticks([])
    axes[0].set_xticks(range(n))

    # 3x4
    r34 = x.reshape(3, n // 3)
    axes[1].imshow(r34.numpy(), cmap="viridis")
    for i in range(r34.shape[0]):
        for j in range(r34.shape[1]):
            axes[1].text(j, i, str(r34[i, j].item()), ha="center", va="center",
                         color="white", fontsize=11, fontweight="bold")
    axes[1].set_title(f"reshape({r34.shape[0]}, {r34.shape[1]})", fontsize=11)
    axes[1].set_xticks(range(r34.shape[1]))
    axes[1].set_yticks(range(r34.shape[0]))

    # 4x3
    r43 = x.reshape(n // 3, 3)
    axes[2].imshow(r43.numpy(), cmap="viridis")
    for i in range(r43.shape[0]):
        for j in range(r43.shape[1]):
            axes[2].text(j, i, str(r43[i, j].item()), ha="center", va="center",
                         color="white", fontsize=11, fontweight="bold")
    axes[2].set_title(f"reshape({r43.shape[0]}, {r43.shape[1]})", fontsize=11)
    axes[2].set_xticks(range(r43.shape[1]))
    axes[2].set_yticks(range(r43.shape[0]))

    plt.suptitle("torch.reshape — same data, different layout", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_masking(data, mask, mask_title):
    """Visualise a tensor, a boolean mask, and the masked result side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    nrows, ncols = data.shape

    # Original data
    im0 = axes[0].imshow(data.numpy(), cmap="RdBu_r")
    axes[0].set_title("Original tensor", fontsize=11)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Add numerical values for original data
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j].item()
            axes[0].text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(val) > 1.0 else "black",
                        fontweight="bold")

    # Mask
    axes[1].imshow(mask.int().numpy(), cmap="Greys_r", vmin=0, vmax=1)
    axes[1].set_title(mask_title, fontsize=11)

    # Masked result
    masked = torch.where(mask, data, torch.tensor(float("nan")))
    im2 = axes[2].imshow(masked.numpy(), cmap="RdBu_r")
    axes[2].set_title("Masked result", fontsize=11)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Add numerical values for masked data (only where mask is True)
    for i in range(nrows):
        for j in range(ncols):
            if mask[i, j]:
                val = data[i, j].item()
                axes[2].text(j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=10,
                            color="white" if abs(val) > 1.0 else "black",
                            fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_relu():
    """Plot ReLU activation implemented via torch.where."""
    x = torch.linspace(-3, 3, 200)
    relu = torch.where(x > 0, x, torch.zeros_like(x))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x.numpy(), x.numpy(), "--", alpha=0.4, label="identity")
    ax.plot(x.numpy(), relu.numpy(), linewidth=2, label="ReLU via mask")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.grid(True)
    ax.set_title("ReLU implemented with torch.where")
    plt.tight_layout()
    plt.show()


def plot_tensors(*items, title=None):
    """Display one or more tensors as annotated colour grids.

    Parameters
    ----------
    *items : (Tensor, str) tuples or str
        Each positional argument is either:

        - A ``(tensor, label)`` tuple rendered as a coloured grid.
        - A plain string rendered as an operator symbol between grids.
    title : str, optional
        Figure super-title displayed above all grids.
    """
    # ---- parse items into a flat list of parts ----
    parts = []          # ("grid", tensor_2d, label, fmt) | ("op", symbol)
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds"]
    cmap_idx = 0
    for item in items:
        if isinstance(item, str):
            parts.append(("op", item))
        else:
            t, label = item
            if t.ndim == 0:
                t = t.reshape(1, 1)
            elif t.ndim == 1:
                t = t.unsqueeze(0)
            is_bool = t.dtype == torch.bool
            t_draw = t.int().float() if is_bool else t.float()
            fmt = ".0f" if is_bool else _auto_fmt(t_draw)
            cmap = "Greys_r" if is_bool else cmaps[cmap_idx % len(cmaps)]
            if not is_bool:
                cmap_idx += 1
            parts.append(("grid", t_draw, label, fmt, cmap))

    # ---- layout ----
    widths = []
    for p in parts:
        widths.append(0.4 if p[0] == "op" else max(p[1].shape[1], 1))

    max_rows = max(p[1].shape[0] for p in parts if p[0] == "grid")
    fig_w = max(sum(w * 1.1 for w in widths) + 1, 4)
    fig_h = max(max_rows * 0.9 + 1.5, 2.5)

    n = len(parts)
    fig, axes = plt.subplots(
        1, n, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": widths},
    )
    if n == 1:
        axes = [axes]

    for ax, part in zip(axes, parts):
        if part[0] == "op":
            ax.text(0.5, 0.5, part[1], ha="center", va="center",
                    fontsize=18, fontweight="bold")
            ax.axis("off")
        else:
            _, t_draw, label, fmt, cmap = part
            hl = torch.ones(t_draw.shape[0], t_draw.shape[1], dtype=torch.bool)
            _draw_matrix(ax, t_draw, hl, cmap, fmt=fmt)
            ax.set_title(label, fontsize=18)

    if title:
        plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()
