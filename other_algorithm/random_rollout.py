import numpy as np


class RandomRolloutAgent:
    """纯随机回滚策略（Monte Carlo Random Rollout 基线）。

    每步从合法动作集合中均匀随机采样，不维护任何状态或 Q 表。
    用途：作为最简基线，衡量 Q-learning 等学习算法带来的收益。
    """

    def __init__(self, n_actions: int = 4, seed: int = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs) -> int:
        """随机选取动作，忽略观测值。

        :param obs: 当前观测（兼容接口，不使用）
        :return: 随机动作索引 [0, n_actions)
        """
        return int(self.rng.integers(0, self.n_actions))
