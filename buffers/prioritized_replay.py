import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class PrioritizedNStepReplayBuffer:
    """Prioritized replay buffer with n-step return support."""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        n_step: int,
        gamma: float,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)

        self.storage: List[Dict] = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

        self.nstep_queue: Deque[Tuple[Dict, int, float, Dict, bool]] = deque()

    def __len__(self) -> int:
        return len(self.storage)

    def _build_nstep_transition(self):
        # Aggregate rewards from the oldest transition using up to n steps.
        reward_sum = 0.0
        next_state = None
        done_flag = False
        steps_used = 0

        for idx, (_, _, reward, n_state, done) in enumerate(self.nstep_queue):
            reward_sum += (self.gamma**idx) * float(reward)
            next_state = n_state
            steps_used = idx + 1
            if done or steps_used >= self.n_step:
                done_flag = bool(done)
                break

        state, action, _, _, _ = self.nstep_queue[0]
        gamma_pow = self.gamma**steps_used
        return (
            state,
            int(action),
            float(reward_sum),
            next_state,
            done_flag,
            float(gamma_pow),
        )

    def _push_one(self, transition):
        state, action, reward_n, next_state, done, gamma_pow = transition
        item = {
            "state": state,
            "action": int(action),
            "reward": float(reward_n),
            "next_state": next_state,
            "done": float(done),
            "gamma_pow": float(gamma_pow),
        }

        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def add(
        self, state: Dict, action: int, reward: float, next_state: Dict, done: bool
    ):
        self.nstep_queue.append(
            (state, int(action), float(reward), next_state, bool(done))
        )

        if len(self.nstep_queue) >= self.n_step:
            transition = self._build_nstep_transition()
            self._push_one(transition)
            self.nstep_queue.popleft()

        if done:
            while self.nstep_queue:
                transition = self._build_nstep_transition()
                self._push_one(transition)
                self.nstep_queue.popleft()

    def sample(self, batch_size: int, beta: float):
        size = len(self.storage)
        if size == 0:
            raise ValueError("Replay buffer is empty.")

        priorities = self.priorities[:size]
        scaled = np.power(priorities + 1e-8, self.alpha)
        probs = scaled / np.sum(scaled)

        indices = np.random.choice(size, batch_size, replace=size < batch_size, p=probs)
        samples = [self.storage[int(i)] for i in indices]

        weights = np.power(size * probs[indices] + 1e-8, -float(beta))
        weights = weights / (np.max(weights) + 1e-8)

        batch = {
            "state_scalar": np.stack([s["state"]["scalar"] for s in samples], axis=0),
            "state_patch_s": np.stack([s["state"]["patch_s"] for s in samples], axis=0),
            "state_patch_l": np.stack([s["state"]["patch_l"] for s in samples], axis=0),
            "state_action_feat": np.stack(
                [s["state"]["action_feat"] for s in samples], axis=0
            ),
            "next_scalar": np.stack(
                [s["next_state"]["scalar"] for s in samples], axis=0
            ),
            "next_patch_s": np.stack(
                [s["next_state"]["patch_s"] for s in samples], axis=0
            ),
            "next_patch_l": np.stack(
                [s["next_state"]["patch_l"] for s in samples], axis=0
            ),
            "next_action_feat": np.stack(
                [s["next_state"]["action_feat"] for s in samples], axis=0
            ),
            "action": np.array([s["action"] for s in samples], dtype=np.int64),
            "reward": np.array([s["reward"] for s in samples], dtype=np.float32),
            "done": np.array([s["done"] for s in samples], dtype=np.float32),
            "gamma_pow": np.array([s["gamma_pow"] for s in samples], dtype=np.float32),
            "weights": weights.astype(np.float32),
            "indices": indices.astype(np.int64),
        }
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        abs_err = np.abs(td_errors).astype(np.float32) + 1e-6
        for idx, value in zip(indices, abs_err):
            self.priorities[int(idx)] = float(value)
        self.max_priority = max(self.max_priority, float(np.max(abs_err)))
