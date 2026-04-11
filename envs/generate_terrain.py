import numpy as np
import numpy.typing as npt
import matplotlib
from omegaconf import DictConfig
import os
from utils import log as log
from .environment_grid import update_environment_grid


def _get_pyplot(cfg: DictConfig):
    runtime_cfg = cfg.get("runtime", {}) if hasattr(cfg, "get") else {}
    if bool(runtime_cfg.get("server_mode", False)):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_terrain(cfg: DictConfig, arr: npt.NDArray[np.float64], run_dir: str, console):
    """
    在run_dir下保存生成的地形数据和可视化图片

    :param arr: 生成的地形 numpy 数组
    :param run_dir: 保存文件的目录路径
    """

    log(
        console,
        "INFO",
        "min depth: {:.2f}, max depth: {:.2f}".format(arr.min(), arr.max()),
    )
    # print("shape:", arr.shape, "min,max:", arr.min(), arr.max())
    # 保存原始数据
    csv_path = update_environment_grid(run_dir, seafloor=arr)
    # 保存为jpg图片
    plt = _get_pyplot(cfg)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(arr, cmap="viridis", origin="lower")
    plt.colorbar(im, label="depth (negative, 0 = sea level)")
    plt.title("50x50 Seafloor Elevation (negative values)")
    jpg_path = os.path.join(run_dir, "seafloor.jpg")
    plt.savefig(jpg_path, format="jpg")
    plt.close()
    log(console, "INFO", f"地形数据和图片已保存到: {run_dir}")
    log(console, "INFO", f"环境网格 CSV 已更新: {csv_path}")
    # print(f"地图已保存到: {run_dir}")


def generate_terrain(
    cfg: DictConfig,
    run_dir,
    console,
    shape=(50, 50),
    seed=2025,
    n_seamounts=19,
    n_canyons=0,
):
    """
    生成一个随机的海底地形，包含基础起伏、若干海山和峡谷

    :param cfg: 配置对象，包含生成参数
    :param run_dir: 保存生成地形的目录路径
    :param shape: 地形的尺寸 (height, width)
    :param seed: 随机种子，确保可复现
    :param n_seamounts: 海山数量
    :param n_canyons: 峡谷数量
    :return: 生成的地形 numpy 数组
    """

    rng = np.random.default_rng(seed)
    h, w = shape
    # 基础平滑噪声（Perlin-like via random + gaussian filter approximation）
    base = rng.normal(scale=1.0, size=shape)
    # 使用频域平滑（简单的多层高斯模糊效果）

    def smooth(arr, iters=3):
        out = arr.copy()
        for _ in range(iters):
            out = (
                np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
                + 4 * out
            ) / 8
        return out

    terrain = smooth(base, iters=6) * 2.0  # 基础起伏

    # 添加若干海山（用二维高斯峰）
    for i in range(n_seamounts):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        amp = rng.uniform(1.0, 4.0)
        sx = rng.uniform(2.0, 6.0)
        sy = rng.uniform(2.0, 6.0)
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        gauss = amp * np.exp(
            -(((X - cx) ** 2) / (2 * sx * sx) + ((Y - cy) ** 2) / (2 * sy * sy))
        )
        terrain += gauss

    # 添加若干狭长峡谷（用沿某方向的凹槽：长条形高斯/指数衰减）
    for i in range(n_canyons):
        # 随机确定一条线段作为峡谷中心线
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        x1, y1 = rng.uniform(0, w), rng.uniform(0, h)
        length = np.hypot(x1 - x0, y1 - y0) + 1e-6
        width = rng.uniform(1.0, 4.0)
        depth = rng.uniform(2.0, 6.0)  # 峡谷深度影响量(后面会整体翻转为负)
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        # 计算点到线段的距离
        px = X - x0
        py = Y - y0
        vx = x1 - x0
        vy = y1 - y0
        t = (px * vx + py * vy) / (vx * vx + vy * vy)
        t = np.clip(t, 0, 1)
        projx = x0 + t * vx
        projy = y0 + t * vy
        dist = np.hypot(X - projx, Y - projy)
        canyon = (
            -depth
            * np.exp(-(dist**2) / (2 * width * width))
            * (1 - 0.5 * np.abs(2 * t - 1))
        )
        # 让峡谷在中心线附近更深，端点处衰减（通过 t 权重）
        terrain += canyon

    # 最终调整：海平面为0，地形应为负数（海底），把海山+基础起伏取反并调节偏移
    # 先归一化到合适范围，然后翻转为负值
    mn = terrain.min()
    mx = terrain.max()
    if mx - mn > 1e-8:
        norm = (terrain - mx) / (mx - mn)  # 限制到 [-1, 0]（最高点接近 0）
    else:
        norm = -np.abs(terrain)

    # 放大一些细节，同时确保值全为负且接近海底深度范围（例如 0 到 -6）
    final = norm * 6.0  # 范围大致在 [-6,0]

    save_terrain(cfg, final, run_dir, console)
    log(console, "SUCC", "地形已生成！")
    return final
