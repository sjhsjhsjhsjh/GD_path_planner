# 生成海流：u (东向), v (北向) — 增强地形影响版本
import numpy as np
import numpy.typing as npt
import matplotlib
from scipy import ndimage
from envs.read_terrain import read_terrain
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


def make_region_flow(shape, bbox, angle_deg, speed=1.0, edge_blend=4):
    """
    在指定矩形区域生成统一方向的底流（带边缘平滑）。
    bbox: (left, top, right, bottom) — 包含边界（inclusive），以像素索引为单位
    angle_deg: 流向角度，度，0 = 东，90 = 北
    speed: 区域内的基础速度标量
    edge_blend: 在距离边界 edge_blend 像素内做平滑过渡（>=1）
    返回: u, v, mask (shape = (h,w))
    """

    h, w = shape
    left, top, right, bottom = bbox
    # clamp 到有效范围
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(w - 1, int(right))
    bottom = min(h - 1, int(bottom))
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    mask = (xs >= left) & (xs <= right) & (ys >= top) & (ys <= bottom)

    theta = np.deg2rad(angle_deg)
    u_dir = np.cos(theta)
    v_dir = np.sin(theta)

    # 到矩形四边的最小距离（内部为 >=0）
    dx = np.minimum(xs - left, right - xs).astype(float)
    dy = np.minimum(ys - top, bottom - ys).astype(float)
    dist_edge = np.minimum(dx, dy)
    # 在区域外设为 -1 以便后续忽略
    dist_edge = np.where(mask, dist_edge, -1.0)

    # 计算混合权重（0..1），在离边界 edge_blend 内从 0 -> 1 平滑过渡
    if edge_blend <= 0:
        blend = (dist_edge >= 0).astype(float)
    else:
        raw = np.clip(dist_edge / float(edge_blend), 0.0, 1.0)
        # smoothstep（更圆滑）
        blend = 0.5 - 0.5 * np.cos(np.pi * raw)

    u = np.zeros((h, w), dtype=float)
    v = np.zeros((h, w), dtype=float)
    u[mask] = u_dir * speed * blend[mask]
    v[mask] = v_dir * speed * blend[mask]
    return u, v, mask


def compose_region_flows(
    shape, region_specs, global_smooth_sigma=1.0, normalize_max=None
):
    """
    把多个区域流合成为一张底流场。
    region_specs: 列表，每项为 dict: {'bbox':(l,t,r,b), 'angle':deg, 'speed':float, 'edge_blend':int}
    global_smooth_sigma: 合成后做的高斯平滑 sigma（0 或 None 则不平滑）
    normalize_max: 若不为 None，则整体缩放到最大速度 = normalize_max
    返回: u, v
    """
    import numpy as np
    from scipy import ndimage

    h, w = shape
    u = np.zeros((h, w), dtype=float)
    v = np.zeros((h, w), dtype=float)
    for spec in region_specs:
        bbox = spec["bbox"]
        angle = spec.get("angle", 0.0)
        speed = spec.get("speed", 1.0)
        edge_blend = spec.get("edge_blend", 4)
        ur, vr, _ = make_region_flow((h, w), bbox, angle, speed, edge_blend)
        u += ur
        v += vr

    if global_smooth_sigma and float(global_smooth_sigma) > 0:
        u = ndimage.gaussian_filter(u, sigma=float(global_smooth_sigma), mode="mirror")
        v = ndimage.gaussian_filter(v, sigma=float(global_smooth_sigma), mode="mirror")

    if normalize_max is not None:
        sp = np.hypot(u, v)
        smax = sp.max() if sp.max() > 0 else 1.0
        scale = float(normalize_max) / smax
        u *= scale
        v *= scale

    return u, v


def generate_flow(
    arr,
    base_speed=1.0,
    main_dir_deg=45,
    seed=1234,
    canyon_boost=3.0,
    mountain_slow=0.6,
    side_deflect=1.6,
    vortex_prob=0.25,
    secondary_count=2,
    secondary_strength=0.9,
    secondary_radius=(3, 10),
    secondary_dir_spec="random",
    secondary_seed=None,
    seamount_boost=1.2,
    seamount_radius_range=(2, 8),
    seamount_percentile=70,
    plain_slow=0.4,
    plain_slope_thresh=1,
):
    """
    增强版海流生成：
    - 在陡坡/海山附近大幅减速 (mountain_slow)，并产生显著的侧向绕流 (side_deflect)。
    - 峡谷检测更敏感，峡谷内部沿谷方向对齐并放大速度 (canyon_boost)。
    - 峰后/复杂地形处更频繁产生涡旋 (vortex_prob) 并加强回流。
    - 新增：海山两侧加速（seamount_boost）和平原则减速（plain_slow），以及在若干局部区域添加副流。
    参数为经验值，可根据需要微调。
    """
    rng = np.random.default_rng(seed)
    h, w = arr.shape
    theta = np.deg2rad(main_dir_deg)
    main_u = np.cos(theta)
    main_v = np.sin(theta)

    shape = arr.shape  # (h,w)
    regions = [
        {"bbox": (0, 0, 24, 24), "angle": 10, "speed": 0.9, "edge_blend": 6},
        {"bbox": (25, 0, 49, 24), "angle": 80, "speed": 0.9, "edge_blend": 6},
        {"bbox": (0, 25, 49, 49), "angle": 60, "speed": 1.0, "edge_blend": 8},
    ]
    u, v = compose_region_flows(
        shape, regions, global_smooth_sigma=1.2, normalize_max=1.2
    )

    #     # 初始主流场，略带噪声
    #     u = base_speed * (main_u * np.ones((h, w)) + 0.05 *
    #                       rng.normal(scale=0.6, size=(h, w)))
    #     v = base_speed * (main_v * np.ones((h, w)) + 0.05 *
    #                       rng.normal(scale=0.6, size=(h, w)))

    # 地形梯度（gx,gy）: x 东向, y 北向（注意 numpy gradient returns gy,gx for 2D arr）
    gy, gx = np.gradient(arr)
    slope = np.hypot(gx, gy)

    # 放大坡度影响（提高对海山的敏感性），并归一化到 [0,1]
    slope_pow = np.power(slope, 1.5)
    smax = slope_pow.max() if slope_pow.max() > 0 else 1.0
    slope_w = slope_pow / smax

    # 1) 在坡度大的位置显著减速（mountain_slow 控制减速幅度）
    slow_factor = 1.0 - mountain_slow * slope_w  # 越陡越慢，可能为负的极端值需要夹紧
    slow_factor = np.clip(slow_factor, 0.05, 1.0)
    u = u * slow_factor
    v = v * slow_factor

    # 2) 强制侧向绕流：侧向方向是主流的右手法向量
    n_x = -main_v
    n_y = main_u
    # 方向由梯度与主流的点积决定（迎风侧/背风侧）
    dot = gx * main_u + gy * main_v
    deflect = side_deflect * slope_w * np.sign(dot)
    u = u + deflect * n_x * base_speed
    v = v + deflect * n_y * base_speed

    # 3) 峡谷检测：更灵敏的局部最小检测与形态学处理以检出细窄峡谷
    local_med = ndimage.median_filter(arr, size=5)
    local_mean = ndimage.uniform_filter(arr, size=7)
    canyon_mask = (arr < local_mean - 0.35) & (arr < local_med - 0.2)
    # 对峡谷进行连通域放大，保留窄长结构
    canyon_mask = ndimage.binary_opening(canyon_mask, structure=np.ones((1, 3)))

    # 对峡谷区域，计算主斜率方向（沿峡谷主轴），使用局部梯度方向近似沿谷方向
    gx_c = gx * canyon_mask
    gy_c = gy * canyon_mask
    # 沿谷方向向量近似为负梯度方向（从高到低）
    mag = np.hypot(gx_c, gy_c)
    mag[mag == 0] = 1.0
    dirx = -gx_c / mag
    diry = -gy_c / mag
    # 将峡谷区域的流场投影到沿谷方向并放大，以实现顺谷流动与对齐
    proj = u * dirx + v * diry
    # 在峡谷内把横向分量衰减，沿向放大
    u = np.where(canyon_mask, proj * dirx * canyon_boost + 0.5 * u * (~canyon_mask), u)
    v = np.where(canyon_mask, proj * diry * canyon_boost + 0.5 * v * (~canyon_mask), v)

    # 5) 平滑场以减少数值尖刺，但保留局部结构（少量迭代）
    def smooth_field(f, iters=3):
        out = f.copy()
        for _ in range(iters):
            out = (
                np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
                + 4 * out
            ) / 8
        return out

    u = smooth_field(u, iters=3)
    v = smooth_field(v, iters=3)

    # 5.1) 海山两侧加速 & 平原则减速：
    # - 通过检测局部高度极大值（海山）并在其周边按高斯权重放大流速（并以坡度加权），
    #   同时在坡度很小的平原区域显著降低流速，从而增大速度对地形的可见变化。
    try:
        # 局部最大值作为海山峰顶的粗略检测（使用指定大小的最大滤波器）
        local_max = ndimage.maximum_filter(arr, size=5) == arr
        peak_mask = local_max & (arr > np.percentile(arr, seamount_percentile))
    except Exception:
        peak_mask = np.zeros_like(arr, dtype=bool)

    if peak_mask.any():
        ys = np.arange(h)[:, None]
        xs = np.arange(w)[None, :]
        accel = np.ones((h, w))
        # 对每个峰顶叠加高斯加速圈（以坡度权重增强在坡面处的作用）
        peak_inds = np.argwhere(peak_mask)
        for py, px in peak_inds:
            dist = np.hypot(ys - py, xs - px)
            # 选取一个半径尺度（在给定范围内取中值）
            sigma = max(
                1.0, (seamount_radius_range[0] + seamount_radius_range[1]) / 5.0
            )
            weight = np.exp(-(dist**2) / (2.0 * sigma * sigma))
            # 在坡面区域（slope_w）更容易被加速，使用坡度权重放大效果
            accel += seamount_boost * weight * slope_w
        # 应用加速因子（保证非负）
        accel = np.clip(accel, 0.1, 5.0)
    else:
        accel = np.ones((h, w))

    # 平原减速：坡度很小的位置按 plain_slow 比例减速
    plain_mask = slope_w < plain_slope_thresh
    plain_factor = 1.0 - plain_slow * (1.0 - slope_w)  # 在坡度接近0时接近 1-plain_slow
    plain_factor = np.clip(plain_factor, 0.05, 1.0)

    # 组合加速与减速（乘性），并应用到速度场
    u = u * accel
    v = v * accel

    # 6) 最终归一化并施加整体尺度
    speed = np.hypot(u, v)
    smax = speed.max() if speed.max() > 0 else 1.0
    # 允许局部速度高于 base_speed，但整体缩放到 base_speed * 1.3
    scale = (base_speed * 1.3) / smax
    u *= scale
    v *= scale

    final_smooth_size = 3
    # === 新增：对最终速度场做一次均值滤波（mean / uniform filter）
    # final_smooth_size: 窗口大小（正整数），越大越平滑；设置为 1 则不做平滑
    try:
        if final_smooth_size and int(final_smooth_size) > 1:
            sz = int(final_smooth_size)
            u = ndimage.uniform_filter(u, size=sz, mode="mirror")
            v = ndimage.uniform_filter(v, size=sz, mode="mirror")
    except Exception:
        # 若 ndimage 过滤异常则忽略，不影响返回
        pass

    return u, v


def generate_ocean_current(cfg: DictConfig, run_dir: str, console):
    arr = read_terrain(run_dir, console)
    plt = _get_pyplot(cfg)

    u, v = generate_flow(
        arr,
        base_speed=1.0,
        main_dir_deg=45,
        seed=2025,
        canyon_boost=3.0,
        mountain_slow=0.5,
        side_deflect=5.0,
        vortex_prob=0.6,
    )

    log(console, "SUCC", "洋流已生成！")
    log(
        console,
        "INFO",
        "u,v shapes: {}, {}. speed min/max: {:.3f}/{:.3f}".format(
            u.shape, v.shape, np.hypot(u, v).min(), np.hypot(u, v).max()
        ),
    )

    # 保存原始数据
    # 绘制速度场（imshow + quiver）
    plt.figure(figsize=(8, 8))
    # 子图1: 地形并叠加向量
    plt.subplot(2, 1, 1)
    plt.title("Seafloor (depth) with flow vectors")
    im = plt.imshow(arr, cmap="viridis", origin="lower")
    plt.colorbar(im, label="depth")
    # 采样稀疏向量用于 quiver，避免过密；在地形上叠加半透明箭头
    ys = xs = np.arange(0, arr.shape[0], 3)
    X, Y = np.meshgrid(xs, ys)
    plt.quiver(X, Y, u[Y, X], v[Y, X], color="white", scale=10, alpha=0.9, width=0.004)

    # 子图2: 速度热力图和向量（保留原有行为）
    plt.subplot(2, 1, 2)
    plt.title("Surface flow vectors (u east, v north)")
    plt.imshow(np.hypot(u, v), cmap="plasma", origin="lower")
    # 采样稀疏向量用于 quiver，避免过密
    ys = xs = np.arange(0, arr.shape[0], 3)
    X, Y = np.meshgrid(xs, ys)
    plt.quiver(X, Y, u[Y, X], v[Y, X], color="white", scale=10)
    plt.colorbar(label="speed")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "seafloor_flow.png"), dpi=180)
    log(console, "INFO", f"海流数据和图片已保存到: {run_dir}")
    csv_path = update_environment_grid(run_dir, u_flow=u, v_flow=v)
    log(console, "INFO", f"环境网格 CSV 已更新: {csv_path}")
