import os
from utils import log as log
from .environment_grid import environment_grid_path, load_environment_grid


def read_beacon_info(run_dir: str, console):
    """
    从指定目录中读取环境网格 CSV 中的信标层。
    :param run_dir: 包含 environment_grid.csv 的目录路径
    :return: beacon_map 和 beacon_number_map 的 numpy 数组
    """
    csv_path = environment_grid_path(run_dir)
    if not os.path.exists(csv_path):
        log(console, "ERROR", f"{csv_path} 不存在")
        raise FileNotFoundError(f"{csv_path} 不存在")

    layers = load_environment_grid(run_dir)
    beacon_map = layers["beacon_strength"]
    beacon_number_map = layers["beacon_id"].astype(int)

    return beacon_map, beacon_number_map
