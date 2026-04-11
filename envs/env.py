import atexit
import csv
import os
import random

import numpy as np
from .robot import Robot
from .async_step_logger import AsyncStepLogger
from omegaconf import DictConfig
from .read_beacon_info import read_beacon_info
from .read_terrain_current import read_seafloor_wave


class Env:
    map_height: int
    """地图高度(行数)"""
    map_width: int
    """地图宽度(列数)"""
    defaulf_energy: float
    """默认初始能量，基于地图尺寸计算"""
    robot: Robot
    """环境中的机器人对象，包含能量、位置、惯导误差等状态信息"""
    start_x: int
    """起点 x(列) 坐标"""
    start_y: int
    """起点 y(行) 坐标"""
    goal_x: int
    """终点 x(列) 坐标"""
    goal_y: int
    """终点 y(行) 坐标"""
    terminated: int
    """环境是否终止的标志，0 表示未终止，1 表示已终止"""
    地形奖励塑形因子: float
    """地形奖励塑形因子，用于调整地形对奖励的影响程度"""
    靠近目标奖励塑形因子: float
    """靠近目标奖励塑形因子，用于调整靠近目标对奖励的影响程度"""
    step_total: int
    """从起点到终点的曼哈顿距离，作为总步数的估计"""
    now_init_energy: float
    """根据起点和终点计算的实际初始能量，通常是 step_total 的某个倍数"""
    goal_heatmap: np.ndarray
    """目标点的高斯热力图，用于奖励塑形，值越大表示越靠近目标点"""
    truncated: int
    """仿真是否被截断的标志，0 表示未截断，1 表示已截断"""
    map: np.ndarray
    """环境地图，包含地形信息，通常是一个二维数组"""
    depth_reward_map: np.ndarray
    """地形奖励地图，根据阈值计算得到，-1~1 之间，值越大表示地形越有利于航行"""

    STEP_REWARD_COLUMNS = [
        "episode",
        "step",
        "stage",
        "action",
        "last_pos_x",
        "last_pos_y",
        "pos_x",
        "pos_y",
        "goal_x",
        "goal_y",
        "distance_before",
        "distance_after",
        "energy_remaining",
        "ins_error",
        "terminated",
        "termination_reason",
        "step_penalty",
        "boundary_penalty",
        "energy_penalty",
        "ins_penalty",
        "revisit_penalty",
        "goal_reward",
        "approach_reward",
        "terrain_reward",
        "current_reward",
        "energy_reward",
        "energy_step_cost",
        "current_alignment",
        "beacon_reward",
        "step_reward",
    ]

    @staticmethod
    def _fmt_float5(value):
        return round(float(value), 5)

    def __init__(self, cfg: DictConfig, run_dir=None, console=None):
        """环境类，包含地图信息、机器人状态、奖励塑形因子等

        :param cfg: 配置对象，包含环境参数
        :param run_dir: 当前运行的输出目录，用于保存环境相关文件
        :param console: rich Console 对象，用于打印日志
        """
        self.cfg = cfg
        # 兼容：如果外部未传入 run_dir，则尝试读取 Hydra 的运行目录（通常为当前工作目录）
        resolved_run_dir = run_dir
        if not resolved_run_dir:
            try:
                resolved_run_dir = os.getcwd()
            except Exception:
                resolved_run_dir = "."
        self.run_dir = resolved_run_dir
        self.console = console
        self.step_rewards_filename = "step_rewards.csv"
        # [性能优化] 日志开关 - 服务器训练应设为false，本地验证可设为true
        self.enable_step_logging = True
        # [性能优化] 目标先验图按episode缓存成员
        self.cached_goal_prior = None
        self.cached_goal_prior_params = None
        reward_cfg = cfg.get("reward", {}) if hasattr(cfg, "get") else {}
        self.step_penalty_value = -float(reward_cfg.get("step_penalty", 0.02))
        self.out_of_bounds_penalty = -float(
            reward_cfg.get("out_of_bounds_penalty", 4.0)
        )
        self.energy_exhausted_penalty = -float(
            reward_cfg.get("energy_exhausted_penalty", 3.0)
        )
        self.ins_diverged_penalty = -float(reward_cfg.get("ins_diverged_penalty", 3.0))
        self.goal_reward_base = float(reward_cfg.get("goal_reward_base", 15.0))
        self.goal_energy_bonus = float(reward_cfg.get("goal_energy_bonus", 5.0))
        self.progress_reward_scale = float(
            reward_cfg.get("progress_reward_scale", 25.0)
        )
        self.approach_reward_weight = float(
            reward_cfg.get("approach_reward_weight", 1.0)
        )
        self.move_away_penalty_scale = float(
            reward_cfg.get("move_away_penalty_scale", 1.2)
        )
        self.revisit_penalty_base = float(reward_cfg.get("revisit_penalty_base", 0.12))
        self.revisit_penalty_cap = float(reward_cfg.get("revisit_penalty_cap", 0.6))
        self.terrain_reward_weight = float(reward_cfg.get("terrain_reward_weight", 0.2))
        self.current_reward_weight = float(reward_cfg.get("current_reward_weight", 0.1))
        self.terrain_step_scale = float(reward_cfg.get("terrain_step_scale", 1.0))
        self.current_step_scale = float(reward_cfg.get("current_step_scale", 1.0))
        self.terrain_component_clip = float(
            reward_cfg.get("terrain_component_clip", 1.0)
        )
        self.current_component_clip = float(
            reward_cfg.get("current_component_clip", 1.0)
        )
        self.energy_base_cost = float(reward_cfg.get("energy_base_cost", 1.0))
        self.reverse_current_energy_scale = float(
            reward_cfg.get("reverse_current_energy_scale", 0.3)
        )
        self.energy_reward_weight = float(reward_cfg.get("energy_reward_weight", 0.3))
        self.beacon_reward_value = float(reward_cfg.get("beacon_reward_value", 0.2))
        self.ins_error_threshold = int(cfg.env.get("ins_error_threshold", 10))

        # 记录当前训练的 episode 和 step（用于日志）
        self.episode_id = 0
        self.step_in_episode = 0

        self.map_height = cfg.env.rows
        self.map_width = cfg.env.cols
        self.width = self.map_width
        self.height = self.map_height
        self.defaulf_energy = min(self.map_width, self.map_height) * 0.8
        self.depth_reward_map = np.zeros((self.map_width, self.map_height))

        self.robot = Robot(self.defaulf_energy)

        self.start_x = -1
        self.start_y = -1
        self.goal_x = self.map_width
        self.goal_y = self.map_height

        self.terminated = 0

        self.地形奖励塑形因子 = 0
        self.靠近目标奖励塑形因子 = 0

        # 初始化用于信标记录的变量，避免 step 中未初始化访问
        self.上一个信标区域编号 = -1
        self.历史途径信标区域 = []
        self.current_stage = "unscheduled"
        self.last_termination_reason = "running"
        self.position_visit_counts = {}

        self._step_logger = None
        self._step_logger_path = None
        self._maps_loaded = False
        self.u_norm = None
        self.v_norm = None
        self.terrain_array = None
        self.beacon_array = None
        self.beacon_number_array = None

        atexit.register(self.close)

    def set_reward_weights(self, **kwargs):
        if "progress_reward_scale" in kwargs:
            self.progress_reward_scale = float(kwargs["progress_reward_scale"])
        if "approach_reward_weight" in kwargs:
            self.approach_reward_weight = float(kwargs["approach_reward_weight"])
        if "terrain_reward_weight" in kwargs:
            self.terrain_reward_weight = float(kwargs["terrain_reward_weight"])
        if "current_reward_weight" in kwargs:
            self.current_reward_weight = float(kwargs["current_reward_weight"])
        if "energy_reward_weight" in kwargs:
            self.energy_reward_weight = float(kwargs["energy_reward_weight"])

    def set_stage(self, stage_name):
        self.current_stage = str(stage_name)

    def set_step_rewards_filename(self, filename):
        self.step_rewards_filename = str(filename)
        self._reset_step_logger()

    def set_step_logging_enabled(self, enabled):
        self.enable_step_logging = bool(enabled)

    def close(self):
        if self._step_logger is not None:
            self._step_logger.close()
            self._step_logger = None
            self._step_logger_path = None

    def _reset_step_logger(self):
        if self._step_logger is not None:
            self._step_logger.close()
        self._step_logger = None
        self._step_logger_path = None

    def _ensure_step_logger(self):
        if not self.enable_step_logging:
            return None

        csv_path = os.path.join(self.run_dir, self.step_rewards_filename)
        if self._step_logger is None or self._step_logger_path != csv_path:
            self._reset_step_logger()
            self._step_logger = AsyncStepLogger(csv_path, self.STEP_REWARD_COLUMNS)
            self._step_logger_path = csv_path
        return self._step_logger

    def _load_static_maps(self):
        if self._maps_loaded:
            return

        terrain_map, u_map, v_map = read_seafloor_wave(self.run_dir, self.console)
        terrain_map = np.asarray(terrain_map, dtype=np.float32)
        u_map = np.asarray(u_map, dtype=np.float32)
        v_map = np.asarray(v_map, dtype=np.float32)

        min_depth = float(np.min(terrain_map))
        max_depth = float(np.max(terrain_map))
        depth_span = max(1e-6, max_depth - min_depth)
        self.map = ((terrain_map - min_depth) / depth_span).astype(np.float32)

        max_u = max(1e-6, float(np.max(np.abs(u_map))))
        max_v = max(1e-6, float(np.max(np.abs(v_map))))
        self.u = u_map
        self.v = v_map
        self.u_norm = (u_map / max_u).astype(np.float32)
        self.v_norm = (v_map / max_v).astype(np.float32)

        beacon_map, beacon_number_map = read_beacon_info(self.run_dir, self.console)
        self.beacon_map = np.asarray(beacon_map, dtype=np.float32)
        self.beacon_number_map = np.asarray(beacon_number_map, dtype=np.int32)

        self.terrain_array = self.map
        self.beacon_array = self.beacon_map
        self.beacon_number_array = self.beacon_number_map

        self.generate_terrain_reward_map()
        self._maps_loaded = True

    def get_observation_cache(self):
        return {
            "terrain": self.terrain_array,
            "u_norm": self.u_norm,
            "v_norm": self.v_norm,
            "beacon": self.beacon_array,
            "depth_reward": self.depth_reward_map,
        }

    def reset(self):
        # 随机生成起点和终点
        self.start_x = random.randint(0, self.map_width - 1)
        self.start_y = random.randint(0, self.map_height - 1)
        self.goal_x = random.randint(0, self.map_width - 1)
        self.goal_y = random.randint(0, self.map_height - 1)

        # 终点和起点重合，重新生成
        while self.start_x == self.goal_x and self.start_y == self.goal_y:
            self.goal_x = random.randint(0, self.map_width - 1)
            self.goal_y = random.randint(0, self.map_height - 1)

        # !!!!!!!!注意：当前使用固定起终点进行试验！！！！
        self.start_x = 6
        self.start_y = 6
        self.goal_x = 43
        self.goal_y = 43
        self.robot = Robot(self.defaulf_energy, (self.start_x, self.start_y))

        # 设置仿真运行状态
        self.truncated = 0
        self.terminated = 0

        # 根据生成的起点和终点，计算初始能量
        self.step_total = abs(self.goal_x - self.start_x) + abs(
            self.goal_y - self.start_y
        )
        self.now_init_energy = self.step_total * 1.5
        self.robot = Robot(self.now_init_energy, (self.start_x, self.start_y))
        self.robot.INS_error = 0
        self.generate_goal_Gauss_heatmap(10.0)

        # [性能优化] 如果启用了goal_prior缓存，则该一次性计算并存储
        if self.cfg.dqn.get("cache_goal_prior", True):
            width = int(self.map_width)
            height = int(self.map_height)
            gx = int(self.goal_x)
            gy = int(self.goal_y)
            denom = int(self.step_total)

            # 快速计算 goal_prior（维持在 CPU 上，需要时再传到GPU）
            xs = np.arange(width, dtype=np.float32).reshape(-1, 1)
            ys = np.arange(height, dtype=np.float32).reshape(1, -1)
            dist = (np.abs(xs - float(gx)) + np.abs(ys - float(gy))).astype(np.float32)
            scale = max(1.0, float(denom))
            self.cached_goal_prior = np.exp(-dist / scale).astype(np.float32)
            self.cached_goal_prior_params = (gx, gy, denom)
        else:
            self.cached_goal_prior = None

        # 设置靠近目标奖励相关
        self.靠近目标奖励单步系数 = 1 / self.step_total
        self.远离目标奖励单步系数 = 1 / self.step_total * 1.5

        self._load_static_maps()

        # 设置地形奖励相关
        平均水深期望 = self.计算区域平均水深期望(
            (self.start_x, self.start_y), (self.goal_x, self.goal_y)
        )
        self.地形奖励单步系数 = (
            self.terrain_step_scale / (self.step_total * max(0.05, 平均水深期望))
            if self.step_total != 0
            else 0
        )

        # 设置单步海流奖励
        self.海流奖励单步系数 = self.current_step_scale / max(1, self.step_total)

        # 重置信标记录
        self.上一个信标区域编号 = -1
        self.历史途径信标区域 = []
        self.position_visit_counts = {}

        self._sync_ins_error_with_beacon()

        # 增加 episode 计数并重置 step 计数
        try:
            self.episode_id = int(getattr(self, "episode_id", 0)) + 1
        except Exception:
            self.episode_id = 1
        self.step_in_episode = 0

        os.makedirs(self.run_dir, exist_ok=True)
        self._ensure_step_logger()

        return self._get_obs()

    # 生成目标点高斯热力图
    def generate_goal_Gauss_heatmap(self, sigma=25.0):
        ys = np.arange(self.map_height)[:, None]
        xs = np.arange(self.map_width)[None, :]
        dist_sq = (xs - self.goal_x) ** 2 + (ys - self.goal_y) ** 2
        self.goal_heatmap = np.exp(-dist_sq / (2 * sigma**2))
        return self.goal_heatmap

    def get_goal_Gauss_heatmap(self):
        return self.goal_heatmap

    # 生成地形奖励地图
    def generate_terrain_reward_map(self):
        # 使用 threshold 生成每一个点深度 reward
        self.depth_reward_map = np.zeros((self.map_width, self.map_height))
        不可航行深度 = 0.5
        深水区深度 = 0.13
        可航行深度段 = 不可航行深度 - 深水区深度
        for i in range(self.map_width):
            for j in range(self.map_height):
                if self.map[i][j] > 不可航行深度:
                    self.depth_reward_map[i][j] = -1
                elif self.map[i][j] > 深水区深度 and self.map[i][j] <= 不可航行深度:
                    self.depth_reward_map[i][j] = (
                        不可航行深度 - self.map[i][j]
                    ) / 可航行深度段
                else:
                    self.depth_reward_map[i][j] = 1.0

    def 计算区域平均水深期望(self, start_point, goal_point):
        lu = min(start_point[0], goal_point[0]), min(start_point[1], goal_point[1])
        rd = max(start_point[0], goal_point[0]), max(start_point[1], goal_point[1])
        arv = 0.0
        for i in range(lu[0], rd[0]):
            for j in range(lu[1], rd[1]):
                arv = arv + self.depth_reward_map[i][j]
        area = (rd[0] - lu[0] + 1) * (rd[1] - lu[1] + 1)
        return arv / area

    def 计算靠近目标奖励(self, last_pos, cur_pos, goal):
        # 使用归一化的曼哈顿距离变化作为主导 shaping reward
        last_dist = abs(last_pos[0] - goal[0]) + abs(last_pos[1] - goal[1])
        cur_dist = abs(cur_pos[0] - goal[0]) + abs(cur_pos[1] - goal[1])
        delta = last_dist - cur_dist
        normalized_delta = delta / max(1, self.step_total)
        if normalized_delta >= 0:
            return normalized_delta * self.progress_reward_scale
        return (
            normalized_delta * self.progress_reward_scale * self.move_away_penalty_scale
        )

    def 计算地形奖励(self, cur_pos):
        # 返回深度奖励表中的值乘以系数
        try:
            val = self.depth_reward_map[cur_pos[0]][cur_pos[1]]
        except Exception:
            val = 0
        coef = getattr(self, "地形奖励单步系数", 0.0)
        return val * coef

    def 计算海流奖励(self, last_pos, cur_pos):
        # 简化：使用当前位置与上一个位置的位移方向与海流方向的点积作为奖励
        try:
            ux = self.u[cur_pos[0]][cur_pos[1]]
            uy = self.v[cur_pos[0]][cur_pos[1]]
        except Exception:
            return 0
        dx = cur_pos[0] - last_pos[0]
        dy = cur_pos[1] - last_pos[1]
        # 规范化位移
        mag = (dx * dx + dy * dy) ** 0.5
        if mag == 0:
            return 0
        dot = (ux * dx + uy * dy) / (mag + 1e-9)
        coef = getattr(self, "海流奖励单步系数", 0.0)
        return dot * coef

    def 计算海流对齐值(self, last_pos, cur_pos):
        try:
            ux = self.u[cur_pos[0]][cur_pos[1]]
            uy = self.v[cur_pos[0]][cur_pos[1]]
        except Exception:
            return 0.0
        dx = cur_pos[0] - last_pos[0]
        dy = cur_pos[1] - last_pos[1]
        mag = (dx * dx + dy * dy) ** 0.5
        if mag == 0:
            return 0.0
        return float((ux * dx + uy * dy) / (mag + 1e-9))

    def 计算单步能耗(self, current_alignment):
        reverse_current_cost = self.reverse_current_energy_scale * max(
            0.0, -current_alignment
        )
        return self.energy_base_cost + reverse_current_cost

    def _sync_ins_error_with_beacon(self):
        if (
            0 <= self.robot.pos_x < self.map_width
            and 0 <= self.robot.pos_y < self.map_height
            and self.beacon_map[self.robot.pos_x][self.robot.pos_y] > 0
        ):
            self.robot.INS_error = 0

    def _get_obs(self):
        dx = self.goal_x - self.robot.pos_x
        dy = self.goal_y - self.robot.pos_y
        ins_error = min(int(self.robot.INS_error), self.ins_error_threshold)
        return (self.robot.pos_x, self.robot.pos_y, dx, dy, ins_error)

    def move_robot(self, action):
        # 执行 action
        self.robot.INS_error = self.robot.INS_error + 1
        # 四向动作
        # 0: down (y-1)
        if action == 0:
            self.robot.pos_y = self.robot.pos_y - 1
        # 1: right (x+1)
        elif action == 1:
            self.robot.pos_x = self.robot.pos_x + 1
        # 2: up (y+1)
        elif action == 2:
            self.robot.pos_y = self.robot.pos_y + 1
        # 3: left (x-1)
        elif action == 3:
            self.robot.pos_x = self.robot.pos_x - 1
        self._sync_ins_error_with_beacon()

    def step(self, action):
        step_reward = 0
        step_terminated = 0
        last_pos = (self.robot.pos_x, self.robot.pos_y)
        termination_reason = "running"
        step_penalty = self.step_penalty_value
        boundary_penalty = 0.0
        energy_penalty = 0.0
        ins_penalty = 0.0
        revisit_penalty = 0.0
        goal_reward = 0.0
        临时靠近目标奖励 = 0.0
        临时地形奖励 = 0.0
        临时海流奖励 = 0.0
        临时能耗奖励 = 0.0
        单步能耗 = 0.0
        海流对齐值 = 0.0
        临时信标奖励 = 0.0

        # 增加本 episode 的 step 计数（用于日志）
        self.step_in_episode = int(getattr(self, "step_in_episode", 0)) + 1

        self.move_robot(action)

        cur_pos = (self.robot.pos_x, self.robot.pos_y)

        # 连续能耗：基础耗能 + 逆流附加耗能
        海流对齐值 = self.计算海流对齐值(last_pos, cur_pos)
        单步能耗 = self.计算单步能耗(海流对齐值)
        self.robot.energy = self.robot.energy - 单步能耗
        临时能耗奖励 = -self.energy_reward_weight * 单步能耗

        # 先进行结束条件判断
        # 超出地图边界，死了
        if (
            self.robot.pos_x == -1
            or self.robot.pos_x == self.map_width
            or self.robot.pos_y == -1
            or self.robot.pos_y == self.map_height
        ):
            boundary_penalty = self.out_of_bounds_penalty
            step_reward = boundary_penalty
            step_terminated = 1
            termination_reason = "out_of_bounds"

        # 能量耗尽，死了
        if self.robot.energy <= 0:
            energy_penalty = self.energy_exhausted_penalty
            step_reward = energy_penalty
            step_terminated = 1
            termination_reason = "energy_exhausted"

        # 定位发散，死了
        if self.robot.INS_error >= self.ins_error_threshold:
            ins_penalty = self.ins_diverged_penalty
            step_reward = ins_penalty
            step_terminated = 1
            termination_reason = "ins_diverged"

        visit_count = int(self.position_visit_counts.get(cur_pos, 0))
        if visit_count >= 1:
            revisit_penalty = -min(
                self.revisit_penalty_base * visit_count,
                self.revisit_penalty_cap,
            )
            step_reward = step_reward + revisit_penalty
        if step_terminated != 1:
            self.robot.历史轨迹.append(cur_pos)
            self.position_visit_counts[cur_pos] = visit_count + 1

        # 到达目标点
        if self.robot.pos_x == self.goal_x and self.robot.pos_y == self.goal_y:
            # 重点：给一个固定的 reward ，将剩余能量作为 reward
            goal_reward = (
                self.goal_reward_base
                + self.goal_energy_bonus * self.robot.energy / self.now_init_energy
            )
            step_reward = goal_reward
            step_terminated = 1
            termination_reason = "goal_reached"

        self.last_termination_reason = termination_reason

        info = "step_reward = %.4f" % (step_reward)

        # 单轮仿真游戏结束判断
        if step_terminated == 1:
            dx = self.goal_x - self.robot.pos_x
            dy = self.goal_y - self.robot.pos_y
            self._append_step_reward_log(
                action=action,
                last_pos=last_pos,
                cur_pos=cur_pos,
                terminated=step_terminated,
                termination_reason=termination_reason,
                step_penalty=step_penalty,
                boundary_penalty=boundary_penalty,
                energy_penalty=energy_penalty,
                ins_penalty=ins_penalty,
                revisit_penalty=revisit_penalty,
                goal_reward=goal_reward,
                approach_reward=临时靠近目标奖励,
                terrain_reward=临时地形奖励,
                current_reward=临时海流奖励,
                energy_reward=临时能耗奖励,
                energy_step_cost=单步能耗,
                current_alignment=海流对齐值,
                beacon_reward=临时信标奖励,
                step_reward=step_reward,
            )
            return (
                self._get_obs(),
                step_reward,
                step_terminated,
                info,
            )

        # 如果走进了新的信标区域，并且之前没有走过，那么更新历史记录并记录下来
        if (
            (self.beacon_number_map[cur_pos[0]][cur_pos[1]] != 0)
            and (
                self.beacon_number_map[cur_pos[0]][cur_pos[1]]
                != self.上一个信标区域编号
            )
            and (
                self.历史途径信标区域.count(
                    self.beacon_number_map[cur_pos[0]][cur_pos[1]]
                )
                == 0
            )
        ):
            self.上一个信标区域编号 = self.beacon_number_map[cur_pos[0]][cur_pos[1]]
            self.历史途径信标区域.append(self.上一个信标区域编号)
            临时信标奖励 = self.beacon_reward_value

        # 如果没有结束，那么计算单步奖励
        临时靠近目标奖励 = self.计算靠近目标奖励(
            last_pos, cur_pos, (self.goal_x, self.goal_y)
        )
        临时地形奖励 = np.clip(
            self.计算地形奖励(cur_pos),
            -self.terrain_component_clip,
            self.terrain_component_clip,
        )
        临时海流奖励 = np.clip(
            self.计算海流奖励(last_pos, cur_pos),
            -self.current_component_clip,
            self.current_component_clip,
        )
        approach_component = 临时靠近目标奖励 * self.approach_reward_weight
        terrain_component = 临时地形奖励 * self.terrain_reward_weight
        current_component = 临时海流奖励 * self.current_reward_weight
        step_reward = (
            step_reward
            + step_penalty
            + approach_component
            + terrain_component
            + current_component
            + 临时能耗奖励
            + 临时信标奖励
        )
        临时靠近目标奖励 = approach_component
        临时地形奖励 = terrain_component
        临时海流奖励 = current_component
        self._append_step_reward_log(
            action=action,
            last_pos=last_pos,
            cur_pos=cur_pos,
            terminated=step_terminated,
            termination_reason=termination_reason,
            step_penalty=step_penalty,
            boundary_penalty=boundary_penalty,
            energy_penalty=energy_penalty,
            ins_penalty=ins_penalty,
            revisit_penalty=revisit_penalty,
            goal_reward=goal_reward,
            approach_reward=临时靠近目标奖励,
            terrain_reward=临时地形奖励,
            current_reward=临时海流奖励,
            energy_reward=临时能耗奖励,
            energy_step_cost=单步能耗,
            current_alignment=海流对齐值,
            beacon_reward=临时信标奖励,
            step_reward=step_reward,
        )

        info = (
            "靠近目标奖励: %.4f, 实际地形奖励: %.4f, 实际海流奖励: %.4f, 靠近:地形：%.4f, 靠近:海流: %.4f, 地形:海流: %.4f, 信标奖励: %.2f"
            % (
                临时靠近目标奖励,
                临时地形奖励,
                临时海流奖励,
                临时靠近目标奖励 / 临时地形奖励 if 临时地形奖励 != 0 else 0,
                临时靠近目标奖励 / 临时海流奖励 if 临时海流奖励 != 0 else 0,
                临时地形奖励 / 临时海流奖励 if 临时海流奖励 != 0 else 0,
                临时信标奖励,
            )
        )
        dx = self.goal_x - self.robot.pos_x
        dy = self.goal_y - self.robot.pos_y
        return (
            self._get_obs(),
            step_reward,
            step_terminated,
            info,
        )

    def _append_step_reward_log(
        self,
        action,
        last_pos,
        cur_pos,
        terminated,
        termination_reason,
        step_penalty,
        boundary_penalty,
        energy_penalty,
        ins_penalty,
        revisit_penalty,
        goal_reward,
        approach_reward,
        terrain_reward,
        current_reward,
        energy_reward,
        energy_step_cost,
        current_alignment,
        beacon_reward,
        step_reward,
    ):
        if not getattr(self, "enable_step_logging", True):
            return
        try:
            logger = self._ensure_step_logger()
            if logger is None:
                return
            last_dist = abs(last_pos[0] - self.goal_x) + abs(last_pos[1] - self.goal_y)
            cur_dist = abs(cur_pos[0] - self.goal_x) + abs(cur_pos[1] - self.goal_y)
            logger.submit(
                [
                    int(getattr(self, "episode_id", 0)),
                    int(self.step_in_episode),
                    str(getattr(self, "current_stage", "unscheduled")),
                    int(action),
                    int(last_pos[0]),
                    int(last_pos[1]),
                    int(cur_pos[0]),
                    int(cur_pos[1]),
                    int(self.goal_x),
                    int(self.goal_y),
                    int(last_dist),
                    int(cur_dist),
                    self._fmt_float5(self.robot.energy),
                    self._fmt_float5(self.robot.INS_error),
                    int(terminated),
                    str(termination_reason),
                    self._fmt_float5(step_penalty),
                    self._fmt_float5(boundary_penalty),
                    self._fmt_float5(energy_penalty),
                    self._fmt_float5(ins_penalty),
                    self._fmt_float5(revisit_penalty),
                    self._fmt_float5(goal_reward),
                    self._fmt_float5(approach_reward),
                    self._fmt_float5(terrain_reward),
                    self._fmt_float5(current_reward),
                    self._fmt_float5(energy_reward),
                    self._fmt_float5(energy_step_cost),
                    self._fmt_float5(current_alignment),
                    self._fmt_float5(beacon_reward),
                    self._fmt_float5(step_reward),
                ]
            )
        except Exception:
            pass
