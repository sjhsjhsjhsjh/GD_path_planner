import os
from utils import log as log
from .environment_grid import environment_grid_path, load_environment_grid


def read_terrain(run_dir: str, console):
    """
    从指定目录读取环境网格 CSV 中的 seafloor 数据

    :param run_dir: 包含 environment_grid.csv 的目录路径
    :return: 读取的 numpy 数组
    """
    csv_path = environment_grid_path(run_dir)
    if not os.path.exists(csv_path):
        log(console, "ERROR", f"{csv_path} 不存在")
        raise FileNotFoundError(f"{csv_path} 不存在")
    layers = load_environment_grid(run_dir)
    return layers["seafloor"]
