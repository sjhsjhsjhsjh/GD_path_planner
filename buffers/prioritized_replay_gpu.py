from collections import deque
from collections import OrderedDict
from typing import Deque, Dict, List, Optional, Tuple

import torch


class PrioritizedNStepReplayBufferGPU:
    """GPU-only prioritized replay buffer with n-step return support."""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        n_step: int,
        gamma: float,
        device: str,
        observation_cache: Optional[Dict] = None,
        patch_small: int = 11,
        patch_large: int = 21,
        reconstruct_patch_on_sample: bool = False,
        cache_maps_on_gpu: bool = True,
        goal_prior_cache_size: int = 64,
        replay_store_mode: str = "full_patch",
        replay_dtype: str = "float32",
        use_batch_patch_indexing: bool = True,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.patch_small = int(patch_small)
        self.patch_large = int(patch_large)
        self.reconstruct_patch_on_sample = bool(reconstruct_patch_on_sample)
        self.cache_maps_on_gpu = bool(cache_maps_on_gpu)
        self.goal_prior_cache_size = max(1, int(goal_prior_cache_size))

        mode = str(replay_store_mode).strip().lower()
        if mode not in {"full_patch", "meta_only"}:
            raise ValueError("replay_store_mode must be one of: full_patch, meta_only")
        self.replay_store_mode = mode
        self.meta_only = self.replay_store_mode == "meta_only"
        self.use_batch_patch_indexing = bool(use_batch_patch_indexing)

        dtype_name = str(replay_dtype).strip().lower()
        if dtype_name in {"float16", "fp16", "half"}:
            self.replay_dtype = torch.float16
        elif dtype_name in {"float32", "fp32"}:
            self.replay_dtype = torch.float32
        else:
            raise ValueError("replay_dtype must be float16 or float32")

        if (
            self.reconstruct_patch_on_sample or self.meta_only
        ) and not self.cache_maps_on_gpu:
            raise ValueError(
                "Patch reconstruction/meta-only mode requires cache_maps_on_gpu=True."
            )

        self.storage: List[Dict] = []
        self.priorities = torch.zeros(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self.pos = 0
        self.max_priority = 1.0

        self.nstep_queue: Deque[Tuple[Dict, int, float, Dict, bool]] = deque()
        self.map_cache = self._init_map_cache(observation_cache)
        self.goal_prior_cache: (
            "OrderedDict[Tuple[int, int, int, int, int], torch.Tensor]"
        ) = OrderedDict()

    def __len__(self) -> int:
        return len(self.storage)

    def _to_device_2d_float(self, value) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=self.device, dtype=self.replay_dtype)
        else:
            tensor = torch.as_tensor(value, dtype=self.replay_dtype, device=self.device)
        return tensor

    def _init_map_cache(
        self, observation_cache: Optional[Dict]
    ) -> Dict[str, torch.Tensor]:
        if observation_cache is None:
            return {}
        if not (self.reconstruct_patch_on_sample or self.meta_only):
            return {}

        terrain = self._to_device_2d_float(observation_cache["terrain"])
        u_norm = self._to_device_2d_float(observation_cache["u_norm"])
        v_norm = self._to_device_2d_float(observation_cache["v_norm"])
        beacon = self._to_device_2d_float(observation_cache["beacon"])
        depth_reward = self._to_device_2d_float(observation_cache["depth_reward"])
        goal_distance_library = observation_cache.get("goal_distance_library")
        goal_distance_bank = self._build_goal_distance_bank(
            goal_distance_library,
            width=int(terrain.shape[0]),
            height=int(terrain.shape[1]),
        )
        return {
            "terrain": terrain,
            "u_norm": u_norm,
            "v_norm": v_norm,
            "beacon": beacon,
            "depth_reward": depth_reward,
            "goal_distance_bank": goal_distance_bank,
            "width": int(terrain.shape[0]),
            "height": int(terrain.shape[1]),
        }

    def _build_goal_distance_bank(
        self, goal_distance_library, width: int, height: int
    ) -> Optional[torch.Tensor]:
        if goal_distance_library is None:
            return None

        if torch.is_tensor(goal_distance_library):
            return goal_distance_library.to(device=self.device, dtype=self.replay_dtype)

        if isinstance(goal_distance_library, dict):
            bank = []
            for gx in range(width):
                for gy in range(height):
                    dist = goal_distance_library.get((gx, gy))
                    if dist is None:
                        raise KeyError(f"goal_distance_library 缺少键 {(gx, gy)}")
                    bank.append(
                        torch.as_tensor(
                            dist, dtype=self.replay_dtype, device=self.device
                        )
                    )
            return torch.stack(bank, dim=0)

        return None

    def _extract_patch(
        self, arr: torch.Tensor, cx: int, cy: int, size: int
    ) -> torch.Tensor:
        half = size // 2
        patch = torch.zeros((size, size), dtype=torch.float32, device=self.device)

        x0 = max(0, cx - half)
        x1 = min(int(arr.shape[0]), cx + half + 1)
        y0 = max(0, cy - half)
        y1 = min(int(arr.shape[1]), cy + half + 1)

        px0 = x0 - (cx - half)
        py0 = y0 - (cy - half)
        px1 = px0 + (x1 - x0)
        py1 = py0 + (y1 - y0)
        if x1 > x0 and y1 > y0:
            patch[px0:px1, py0:py1] = arr[x0:x1, y0:y1]
        return patch

    def _goal_prior_map(
        self, width: int, height: int, gx: int, gy: int, denom: int
    ) -> torch.Tensor:
        key = (int(gx), int(gy), int(denom), int(width), int(height))
        cached = self.goal_prior_cache.get(key)
        if cached is not None:
            self.goal_prior_cache.move_to_end(key)
            return cached

        xs = torch.arange(width, dtype=torch.float32, device=self.device).view(-1, 1)
        ys = torch.arange(height, dtype=torch.float32, device=self.device).view(1, -1)
        dist = (xs - float(gx)).abs() + (ys - float(gy)).abs()
        scale = max(1.0, float(denom))
        prior = torch.exp(-dist / scale)

        self.goal_prior_cache[key] = prior
        self.goal_prior_cache.move_to_end(key)
        while len(self.goal_prior_cache) > self.goal_prior_cache_size:
            self.goal_prior_cache.popitem(last=False)
        return prior

    def _goal_prior_batch_from_meta(
        self,
        goal_x: torch.Tensor,
        goal_y: torch.Tensor,
        step_total: torch.Tensor,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        if self.map_cache.get("goal_distance_bank") is None:
            priors = []
            batch_size = int(goal_x.shape[0])
            for idx in range(batch_size):
                prior = self._goal_prior_map(
                    width=int(self.map_cache["width"]),
                    height=int(self.map_cache["height"]),
                    gx=int(goal_x[idx].item()),
                    gy=int(goal_y[idx].item()),
                    denom=int(step_total[idx].item()),
                )
                priors.append(prior)
            prior_maps = torch.stack(priors, dim=0)
        else:
            bank = self.map_cache["goal_distance_bank"]
            height = int(self.map_cache["height"])
            goal_idx = (goal_x.to(torch.int64) * height + goal_y.to(torch.int64)).view(
                -1
            )
            dist_maps = bank.index_select(0, goal_idx)
            denom = step_total.to(dtype=self.replay_dtype).clamp_min(1.0).view(-1, 1, 1)
            prior_maps = torch.exp(-(dist_maps.to(dtype=self.replay_dtype) / denom))

        return self._extract_patch_batch_from_maps(prior_maps, center_x, center_y, size)

    def _extract_patch_batch_from_maps(
        self,
        maps: torch.Tensor,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        batch_size = int(center_x.shape[0])
        half = size // 2
        offsets = torch.arange(-half, half + 1, device=self.device, dtype=torch.int64)

        if maps.dim() == 2:
            maps = maps.unsqueeze(0).expand(batch_size, -1, -1)
        elif maps.dim() == 3 and maps.shape[0] != batch_size:
            raise ValueError("Batch map shape mismatch during patch extraction.")

        x_idx = center_x.to(device=self.device, dtype=torch.int64).view(
            batch_size, 1, 1
        )
        y_idx = center_y.to(device=self.device, dtype=torch.int64).view(
            batch_size, 1, 1
        )
        x_idx = (x_idx + offsets.view(1, -1, 1)).clamp(0, maps.shape[-2] - 1)
        y_idx = (y_idx + offsets.view(1, 1, -1)).clamp(0, maps.shape[-1] - 1)
        batch_idx = torch.arange(
            batch_size, device=self.device, dtype=torch.int64
        ).view(batch_size, 1, 1)
        return maps[batch_idx, x_idx, y_idx]

    def _build_patch_batch_from_meta(
        self, samples: List[Dict], key_state: str, size: int
    ) -> torch.Tensor:
        metas = torch.stack([s[key_state]["patch_meta"] for s in samples], dim=0)
        center_x = metas[:, 0]
        center_y = metas[:, 1]
        goal_x = metas[:, 2]
        goal_y = metas[:, 3]
        step_total = metas[:, 4]

        terrain = self.map_cache["terrain"]
        u_norm = self.map_cache["u_norm"]
        v_norm = self.map_cache["v_norm"]
        beacon = self.map_cache["beacon"]

        terrain_patch = self._extract_patch_batch_from_maps(
            terrain, center_x, center_y, size
        )
        u_patch = self._extract_patch_batch_from_maps(u_norm, center_x, center_y, size)
        v_patch = self._extract_patch_batch_from_maps(v_norm, center_x, center_y, size)
        beacon_patch = self._extract_patch_batch_from_maps(
            beacon, center_x, center_y, size
        )
        goal_patch = self._goal_prior_batch_from_meta(
            goal_x=goal_x,
            goal_y=goal_y,
            step_total=step_total,
            center_x=center_x,
            center_y=center_y,
            size=size,
        )
        return torch.stack(
            [terrain_patch, u_patch, v_patch, beacon_patch, goal_patch],
            dim=1,
        )

    def _build_patch(self, meta: torch.Tensor, size: int) -> torch.Tensor:
        x = int(meta[0].item())
        y = int(meta[1].item())
        gx = int(meta[2].item())
        gy = int(meta[3].item())
        step_total = int(meta[4].item())

        terrain = self.map_cache["terrain"]
        u_norm = self.map_cache["u_norm"]
        v_norm = self.map_cache["v_norm"]
        beacon = self.map_cache["beacon"]
        goal_prior = self._goal_prior_map(
            width=int(self.map_cache["width"]),
            height=int(self.map_cache["height"]),
            gx=gx,
            gy=gy,
            denom=step_total,
        )
        return torch.stack(
            [
                self._extract_patch(terrain, x, y, size),
                self._extract_patch(u_norm, x, y, size),
                self._extract_patch(v_norm, x, y, size),
                self._extract_patch(beacon, x, y, size),
                self._extract_patch(goal_prior, x, y, size),
            ],
            dim=0,
        )

    def _build_patch_batch(
        self, samples: List[Dict], key_state: str, size: int
    ) -> torch.Tensor:
        if self.use_batch_patch_indexing:
            return self._build_patch_batch_from_meta(samples, key_state, size)

        patches = [self._build_patch(s[key_state]["patch_meta"], size) for s in samples]
        return torch.stack(patches, dim=0)

    def _clone_state(self, state: Dict) -> Dict[str, torch.Tensor]:
        cloned = {
            "scalar": state["scalar"]
            .detach()
            .to(self.device, dtype=self.replay_dtype)
            .clone(),
            "action_feat": state["action_feat"]
            .detach()
            .to(self.device, dtype=self.replay_dtype)
            .clone(),
        }
        if self.reconstruct_patch_on_sample or self.meta_only:
            if "patch_meta" not in state:
                raise KeyError(
                    "State is missing `patch_meta` required for patch reconstruction."
                )
            cloned["patch_meta"] = (
                state["patch_meta"].detach().to(self.device, dtype=torch.int32).clone()
            )
        else:
            cloned["patch_s"] = (
                state["patch_s"]
                .detach()
                .to(self.device, dtype=self.replay_dtype)
                .clone()
            )
            cloned["patch_l"] = (
                state["patch_l"]
                .detach()
                .to(self.device, dtype=self.replay_dtype)
                .clone()
            )
        return cloned

    def _build_nstep_transition(self):
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
        if next_state is None:
            next_state = state
        gamma_pow = self.gamma**steps_used
        return (
            self._clone_state(state),
            int(action),
            float(reward_sum),
            self._clone_state(next_state),
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
        scaled = torch.pow(priorities + 1e-8, self.alpha)
        probs = scaled / (scaled.sum() + 1e-12)

        replace = size < batch_size
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=replace)
        samples = [self.storage[int(i.item())] for i in indices]

        weights = torch.pow(size * probs[indices] + 1e-8, -float(beta))
        weights = weights / (weights.max() + 1e-8)

        if self.reconstruct_patch_on_sample or self.meta_only:
            if not self.map_cache:
                raise RuntimeError(
                    "Patch reconstruction is enabled but map cache is empty."
                )
            state_patch_s = self._build_patch_batch(samples, "state", self.patch_small)
            state_patch_l = self._build_patch_batch(samples, "state", self.patch_large)
            next_patch_s = self._build_patch_batch(
                samples, "next_state", self.patch_small
            )
            next_patch_l = self._build_patch_batch(
                samples, "next_state", self.patch_large
            )
        else:
            state_patch_s = torch.stack([s["state"]["patch_s"] for s in samples], dim=0)
            state_patch_l = torch.stack([s["state"]["patch_l"] for s in samples], dim=0)
            next_patch_s = torch.stack(
                [s["next_state"]["patch_s"] for s in samples], dim=0
            )
            next_patch_l = torch.stack(
                [s["next_state"]["patch_l"] for s in samples], dim=0
            )

        batch = {
            "state_scalar": torch.stack([s["state"]["scalar"] for s in samples], dim=0),
            "state_patch_s": state_patch_s.to(dtype=self.replay_dtype),
            "state_patch_l": state_patch_l.to(dtype=self.replay_dtype),
            "state_action_feat": torch.stack(
                [s["state"]["action_feat"] for s in samples], dim=0
            ),
            "next_scalar": torch.stack(
                [s["next_state"]["scalar"] for s in samples], dim=0
            ),
            "next_patch_s": next_patch_s.to(dtype=self.replay_dtype),
            "next_patch_l": next_patch_l.to(dtype=self.replay_dtype),
            "next_action_feat": torch.stack(
                [s["next_state"]["action_feat"] for s in samples], dim=0
            ),
            "action": torch.tensor(
                [s["action"] for s in samples], dtype=torch.int64, device=self.device
            ),
            "reward": torch.tensor(
                [s["reward"] for s in samples], dtype=torch.float32, device=self.device
            ),
            "done": torch.tensor(
                [s["done"] for s in samples], dtype=torch.float32, device=self.device
            ),
            "gamma_pow": torch.tensor(
                [s["gamma_pow"] for s in samples],
                dtype=torch.float32,
                device=self.device,
            ),
            "weights": weights.to(dtype=torch.float32),
            "indices": indices.to(dtype=torch.int64),
        }
        return batch

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        else:
            indices = indices.to(device=self.device, dtype=torch.int64)

        if not torch.is_tensor(td_errors):
            td_errors = torch.as_tensor(
                td_errors, dtype=torch.float32, device=self.device
            )
        else:
            td_errors = td_errors.to(device=self.device, dtype=torch.float32)

        abs_err = td_errors.abs() + 1e-6
        self.priorities[indices] = abs_err
        self.max_priority = max(self.max_priority, float(abs_err.max().item()))

    def _state_dict_cpu(self, state: Dict) -> Dict:
        out = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                out[key] = value.detach().to("cpu")
            else:
                out[key] = value
        return out

    def _state_dict_device(self, state: Dict) -> Dict:
        out = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                dtype = torch.int32 if key == "patch_meta" else self.replay_dtype
                out[key] = value.to(device=self.device, dtype=dtype)
            else:
                out[key] = value
        return out

    def export_state(self, include_storage: bool = True) -> Dict:
        state = {
            "capacity": int(self.capacity),
            "alpha": float(self.alpha),
            "n_step": int(self.n_step),
            "gamma": float(self.gamma),
            "patch_small": int(self.patch_small),
            "patch_large": int(self.patch_large),
            "reconstruct_patch_on_sample": bool(self.reconstruct_patch_on_sample),
            "replay_store_mode": str(self.replay_store_mode),
            "replay_dtype": (
                "float16" if self.replay_dtype == torch.float16 else "float32"
            ),
            "use_batch_patch_indexing": bool(self.use_batch_patch_indexing),
            "cache_maps_on_gpu": bool(self.cache_maps_on_gpu),
            "goal_prior_cache_size": int(self.goal_prior_cache_size),
            "pos": int(self.pos),
            "max_priority": float(self.max_priority),
            "priorities": self.priorities.detach().to("cpu"),
            "length": int(len(self.storage)),
        }

        queue_cpu = []
        for s, a, r, ns, d in list(self.nstep_queue):
            queue_cpu.append(
                (
                    self._state_dict_cpu(s),
                    int(a),
                    float(r),
                    self._state_dict_cpu(ns),
                    bool(d),
                )
            )
        state["nstep_queue"] = queue_cpu

        if include_storage:
            storage_cpu = []
            for item in self.storage:
                storage_cpu.append(
                    {
                        "state": self._state_dict_cpu(item["state"]),
                        "action": int(item["action"]),
                        "reward": float(item["reward"]),
                        "next_state": self._state_dict_cpu(item["next_state"]),
                        "done": float(item["done"]),
                        "gamma_pow": float(item["gamma_pow"]),
                    }
                )
            state["storage"] = storage_cpu

        return state

    def import_state(self, state: Dict, with_storage: bool = True):
        if int(state.get("capacity", self.capacity)) != int(self.capacity):
            raise ValueError("Replay buffer capacity mismatch during restore.")
        if int(state.get("n_step", self.n_step)) != int(self.n_step):
            raise ValueError("Replay buffer n_step mismatch during restore.")

        if abs(float(state.get("gamma", self.gamma)) - float(self.gamma)) > 1e-9:
            raise ValueError("Replay buffer gamma mismatch during restore.")

        if int(state.get("patch_small", self.patch_small)) != int(self.patch_small):
            raise ValueError("Replay buffer patch_small mismatch during restore.")

        if int(state.get("patch_large", self.patch_large)) != int(self.patch_large):
            raise ValueError("Replay buffer patch_large mismatch during restore.")

        saved_pri = state.get("priorities")
        if saved_pri is not None:
            if not torch.is_tensor(saved_pri):
                saved_pri = torch.as_tensor(saved_pri, dtype=torch.float32)
            saved_pri = saved_pri.to(device=self.device, dtype=torch.float32)
            if saved_pri.shape[0] != self.priorities.shape[0]:
                raise ValueError(
                    "Replay buffer priorities shape mismatch during restore."
                )
            self.priorities.copy_(saved_pri)

        self.pos = int(state.get("pos", 0)) % max(1, self.capacity)
        self.max_priority = float(state.get("max_priority", 1.0))

        self.nstep_queue.clear()
        for s, a, r, ns, d in state.get("nstep_queue", []):
            self.nstep_queue.append(
                (
                    self._state_dict_device(s),
                    int(a),
                    float(r),
                    self._state_dict_device(ns),
                    bool(d),
                )
            )

        if with_storage:
            storage = []
            for item in state.get("storage", []):
                storage.append(
                    {
                        "state": self._state_dict_device(item["state"]),
                        "action": int(item["action"]),
                        "reward": float(item["reward"]),
                        "next_state": self._state_dict_device(item["next_state"]),
                        "done": float(item["done"]),
                        "gamma_pow": float(item["gamma_pow"]),
                    }
                )
            self.storage = storage
