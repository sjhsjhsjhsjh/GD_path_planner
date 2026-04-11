import numpy as np
from utils.rich_print import log
from rich.console import Console


class QLearningAgent:
    def __init__(
        self,
        width,
        height,
        n_actions=4,
        alpha=0.01,
        gamma=0.99,
        ins_error_threshold=10,
    ):
        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.ins_error_threshold = ins_error_threshold
        # include goal offset (dx,dy) ranges
        self.dx_range = 2 * self.width - 1
        self.dy_range = 2 * self.height - 1
        self.ins_range = self.ins_error_threshold + 1
        # Q table shape: (width, height, dx_range, dy_range, ins_range, n_actions)
        self.Q = np.zeros(
            (
                self.width,
                self.height,
                self.dx_range,
                self.dy_range,
                self.ins_range,
                self.n_actions,
            ),
            dtype=np.float32,
        )

    def select_action(self, state, epsilon):
        x, y, dx, dy, ins_error = (
            int(state[0]),
            int(state[1]),
            int(state[2]),
            int(state[3]),
            int(state[4]),
        )
        # map dx,dy to non-negative indices
        dx_idx = dx + (self.width - 1)
        dy_idx = dy + (self.height - 1)
        ins_idx = max(0, min(ins_error, self.ins_range - 1))
        # bounds safety
        dx_idx = max(0, min(dx_idx, self.dx_range - 1))
        dy_idx = max(0, min(dy_idx, self.dy_range - 1))
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        acts = self.Q[x, y, dx_idx, dy_idx, ins_idx]
        return int(np.argmax(acts))

    def update(self, state, action, reward, next_state, done):
        x, y, dx, dy, ins_error = (
            int(state[0]),
            int(state[1]),
            int(state[2]),
            int(state[3]),
            int(state[4]),
        )
        nx, ny, ndx, ndy, nins_error = (
            int(next_state[0]),
            int(next_state[1]),
            int(next_state[2]),
            int(next_state[3]),
            int(next_state[4]),
        )
        dx_idx = dx + (self.width - 1)
        dy_idx = dy + (self.height - 1)
        ndx_idx = ndx + (self.width - 1)
        ndy_idx = ndy + (self.height - 1)
        ins_idx = max(0, min(ins_error, self.ins_range - 1))
        nins_idx = max(0, min(nins_error, self.ins_range - 1))
        dx_idx = max(0, min(dx_idx, self.dx_range - 1))
        dy_idx = max(0, min(dy_idx, self.dy_range - 1))
        ndx_idx = max(0, min(ndx_idx, self.dx_range - 1))
        ndy_idx = max(0, min(ndy_idx, self.dy_range - 1))

        q_sa = self.Q[x, y, dx_idx, dy_idx, ins_idx, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(
                self.Q[nx, ny, ndx_idx, ndy_idx, nins_idx]
            )
        self.Q[x, y, dx_idx, dy_idx, ins_idx, action] = q_sa + self.alpha * (
            target - q_sa
        )

    def save(self, path):
        save_path = path if path.endswith(".npz") else f"{path}.npz"
        np.savez_compressed(save_path, Q=self.Q.astype(np.float16))
        return save_path

    def load(self, path):
        loaded = np.load(path)
        if isinstance(loaded, np.ndarray):
            self.Q = loaded.astype(np.float32)
            return
        if "Q" in loaded:
            self.Q = loaded["Q"].astype(np.float32)
            return
        raise ValueError(f"无法从文件加载 Q 表: {path}")
