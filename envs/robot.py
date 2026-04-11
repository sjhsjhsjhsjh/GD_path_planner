class Robot:
    energy: float
    """当前剩余能量"""
    pos_x: int
    """当前所在位置的 x(列) 坐标"""
    pos_y: int
    """当前所在位置的 y(行) 坐标"""
    INS_error: float
    """当前惯导误差"""
    上一个信标区域编号: int
    历史途径信标区域: list
    历史轨迹: list

    def __init__(self, 初始能量=100.0, 初始地点=(0, 0)):
        self.energy = 初始能量
        self.pos_x, self.pos_y = 初始地点
        self.INS_error = 0  # 初始惯导误差

        self.上一个信标区域编号 = -1
        self.历史途径信标区域 = []

        self.历史轨迹 = []

    def move(self, action, env):
        """
        执行移动动作，更新位置和能量

        :param action: 移动动作，0-7 分别对应八个方向
        :param env: 环境对象，用于获取地形和洋流信息

        :return: success: 是否成功移动
        :return: truncated: 是否因能量耗尽或其他原因被截断; 0-> 正常移动, 1 -> 能量耗尽, 2 -> 定位发散 3 -> 撞墙
        """

        # 定义四个方向的移动向量
        move_vectors = [
            (0, 1),  # 0: 向右
            (1, 0),  # 1: 向下
            (0, -1),  # 2: 向左
            (-1, 0),  # 3: 向上
        ]
        # 1. 计算移动后位置
        dx, dy = move_vectors[action]
        new_x = self.pos_x + dx
        new_y = self.pos_y + dy

        # 2. 检查是否越界
        if new_x == -1 or new_x >= env.width or new_y == -1 or new_y >= env.height:
            return False, 3  # 撞墙

        # 3. 计算能量消耗
