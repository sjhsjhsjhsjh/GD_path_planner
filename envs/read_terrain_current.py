from envs.read_terrain import read_terrain
from envs.read_ocean_current import read_latest_wave


def read_seafloor_wave(run_dir, console):
    """
    从指定目录中读取最新的地形和洋流数据

    :param run_dir: 包含地形和洋流数据的目录路径
    :param console: rich Console 对象，用于打印日志
    :return: seafloor, u, v 的 numpy 数组
    """

    seafloor = read_terrain(run_dir, console)
    u, v = read_latest_wave(run_dir, console)
    return seafloor, u, v
