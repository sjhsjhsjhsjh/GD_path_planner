import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


def _build_segments(points):
    if len(points) < 2:
        return np.empty((0, 2, 2))
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def plot_trajectory(
    map_array,
    trajectory,
    out_path,
    ins_errors=None,
    goal=None,
    start=None,
    title="Trajectory",
):
    """绘制轨迹并保存为图片。

    :param map_array: 2D numpy array（归一化地形），索引为 [x][y]
    :param trajectory: 列表 of (x,y) 坐标
    :param out_path: 输出文件路径
    """
    if map_array is None:
        raise ValueError("map_array is None")

    # map_array indexed as [x][y], transpose for imshow
    map_img = np.array(map_array).T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, cmap="viridis", origin="lower", alpha=0.82)

    if len(trajectory) > 0:
        points = np.asarray(trajectory, dtype=float)
        if start is None:
            start = trajectory[0]
        if ins_errors is None or len(ins_errors) != len(trajectory):
            ins_errors = np.linspace(0.0, 1.0, len(trajectory))
        ins_errors = np.asarray(ins_errors, dtype=float)
        norm = Normalize(
            vmin=float(np.min(ins_errors)), vmax=float(np.max(ins_errors)) + 1e-9
        )
        cmap = LinearSegmentedColormap.from_list(
            "red_to_purple",
            ["#ff3b3b", "#d81b60", "#9c27b0", "#6a00ff"],
        )

        segments = _build_segments(points)
        if len(segments) > 0:
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=3.0,
                alpha=0.95,
            )
            lc.set_array(ins_errors[1:])
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, shrink=0.82)
            cbar.set_label("INS error")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=ins_errors,
            cmap=cmap,
            norm=norm,
            s=18,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    if start is not None:
        ax.scatter(
            start[0], start[1], s=90, color="#2ecc71", edgecolors="black", zorder=4
        )
    if goal is not None:
        ax.scatter(
            goal[0],
            goal[1],
            s=120,
            color="#ffd166",
            marker="*",
            edgecolors="black",
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlim(-0.5, map_img.shape[1] - 0.5)
    ax.set_ylim(-0.5, map_img.shape[0] - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
