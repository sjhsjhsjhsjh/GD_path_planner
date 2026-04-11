import numpy as np
import numpy.typing as npt
import matplotlib
from omegaconf import DictConfig
import os
from utils import log as log
from envs import read_terrain as read_terrain
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .environment_grid import update_environment_grid


def _get_pyplot(cfg: DictConfig):
    runtime_cfg = cfg.get("runtime", {}) if hasattr(cfg, "get") else {}
    if bool(runtime_cfg.get("server_mode", False)):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_beacon_area(cfg: DictConfig, run_dir: str, console):
    """
    生成信标覆盖区域

    :param cfg: 配置对象，包含信标位置、信号覆盖范围等参数
    :param run_dir: 当前运行的输出目录，用于保存生成的信标覆盖图和数据
    :param console: Rich Console对象，用于输出日志信息

    :return beacon_map: 2D数组，记录每个点的信号强度；
    :return beacon_number_map: 2D数组，记录每个点覆盖的信标编号
    """
    # 从配置中加载参数
    beacons_pos_list = [tuple(p) for p in cfg.beacon.beacon_locations]
    beacon_signal_area = cfg.beacon.beacon_signal_area
    map_width = int(cfg.env.rows)
    map_height = int(cfg.env.cols)

    # 生成信标覆盖区域
    beacon_map = np.zeros((map_width, map_height))  # 记录每个点的信号强度
    beacon_number_map = np.zeros(
        (map_width, map_height)
    )  # 记录每个点覆盖的信标编号（0表示无信标覆盖，1开始表示第1个信标，以此类推）
    temp_index = 0
    for beacon in beacons_pos_list:
        bx = beacon[0]
        by = beacon[1]
        temp_index = temp_index + 1  # 信标编号从1开始，0表示无信标覆盖
        for i in range(map_width):
            for j in range(map_height):
                dist = abs(bx - i) + abs(by - j)
                if dist <= beacon_signal_area:
                    signal_strength = (beacon_signal_area - dist) / beacon_signal_area
                    # 信标覆盖强度叠加，但记录最强信号对应的信标编号
                    if beacon_map[i][j] < signal_strength:
                        beacon_number_map[i][j] = temp_index
                    beacon_map[i][j] = max(beacon_map[i][j], signal_strength)

    map_data = read_terrain(run_dir, console)
    plt = _get_pyplot(cfg)

    # 信标信号覆盖图
    fig1, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Beacon signal overlay")

    im1 = ax.imshow(map_data, cmap="viridis", origin="lower")  # 底层：地形
    divider = make_axes_locatable(ax)  # colorbar 1
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label("depth (normalized)")

    im2 = ax.imshow(
        beacon_number_map, cmap="tab20", origin="lower", alpha=0.4
    )  # 叠加层：信标信号
    cax2 = divider.append_axes("right", size="5%", pad=0.15)  # colorbar 2
    cbar2 = plt.colorbar(im2, cax=cax2, fraction=0.046, pad=0.04)
    cbar2.set_label("beacon signal")

    # 保存
    csv_path = update_environment_grid(
        run_dir,
        beacon_strength=beacon_map,
        beacon_id=beacon_number_map,
    )
    fig_path = os.path.join(run_dir, "beacon_overlay.jpg")
    plt.savefig(fig_path, format="jpg")
    plt.close()

    log(
        console,
        "SUCC",
        f"信标覆盖区域生成完成！已保存信标覆盖图、信号覆盖数组、信标覆盖数组: {fig_path}",
    )
    log(console, "INFO", f"环境网格 CSV 已更新: {csv_path}")

    return beacon_map, beacon_number_map
