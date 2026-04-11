from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # [性能优化] AMP 导入

from models.dqn_model import MultiScaleDuelingQNetwork


class DoubleDQNAgent:
    def __init__(
        self,
        scalar_dim: int,
        patch_channels: int,
        action_feat_dim: int,
        action_dim: int,
        gamma: float,
        lr: float,
        tau: float,
        hidden_dim: int = 128,
        device: str = "cpu",
        use_amp: bool = False,  # [性能优化] AMP 开关
    ):
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(device)

        self.online = MultiScaleDuelingQNetwork(
            scalar_dim=scalar_dim,
            patch_channels=patch_channels,
            action_feat_dim=action_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target = MultiScaleDuelingQNetwork(
            scalar_dim=scalar_dim,
            patch_channels=patch_channels,
            action_feat_dim=action_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        # [性能优化] AMP 初始化
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler(enabled=self.use_amp)

    def state_to_device(self, state: Dict) -> Dict[str, torch.Tensor]:
        out = {}
        for key in ("scalar", "patch_s", "patch_l", "action_feat"):
            value = state[key]
            if torch.is_tensor(value):
                out[key] = value.to(device=self.device, dtype=torch.float32)
            else:
                out[key] = torch.as_tensor(
                    value, dtype=torch.float32, device=self.device
                )
        if "patch_meta" in state:
            meta = state["patch_meta"]
            if torch.is_tensor(meta):
                out["patch_meta"] = meta.to(device=self.device, dtype=torch.int32)
            else:
                out["patch_meta"] = torch.as_tensor(
                    meta, dtype=torch.int32, device=self.device
                )
        return out

    def _to_tensor_state(self, state: Dict) -> Dict[str, torch.Tensor]:
        tensor_state = self.state_to_device(state)
        if tensor_state["scalar"].dim() == 1:
            tensor_state["scalar"] = tensor_state["scalar"].unsqueeze(0)
        if tensor_state["patch_s"].dim() == 3:
            tensor_state["patch_s"] = tensor_state["patch_s"].unsqueeze(0)
        if tensor_state["patch_l"].dim() == 3:
            tensor_state["patch_l"] = tensor_state["patch_l"].unsqueeze(0)
        if tensor_state["action_feat"].dim() == 1:
            tensor_state["action_feat"] = tensor_state["action_feat"].unsqueeze(0)
        return tensor_state

    def _batch_tensor(self, value, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def select_action(self, state: Dict, epsilon: float) -> int:
        if np.random.rand() < float(epsilon):
            return int(np.random.randint(0, self.action_dim))

        amp_ctx = autocast(dtype=torch.float16) if self.use_amp else nullcontext()
        with torch.no_grad():
            with amp_ctx:
                t_state = self._to_tensor_state(state)
                q = self.online(
                    t_state["scalar"],
                    t_state["patch_s"],
                    t_state["patch_l"],
                    t_state["action_feat"],
                )
            return int(torch.argmax(q, dim=1).item())

    def train_step(self, batch: Dict, grad_clip: float = 10.0):
        scalar = self._batch_tensor(batch["state_scalar"], torch.float32)
        patch_s = self._batch_tensor(batch["state_patch_s"], torch.float32)
        patch_l = self._batch_tensor(batch["state_patch_l"], torch.float32)
        action_feat = self._batch_tensor(batch["state_action_feat"], torch.float32)

        next_scalar = self._batch_tensor(batch["next_scalar"], torch.float32)
        next_patch_s = self._batch_tensor(batch["next_patch_s"], torch.float32)
        next_patch_l = self._batch_tensor(batch["next_patch_l"], torch.float32)
        next_action_feat = self._batch_tensor(batch["next_action_feat"], torch.float32)

        action = self._batch_tensor(batch["action"], torch.int64).view(-1, 1)
        reward = self._batch_tensor(batch["reward"], torch.float32).view(-1, 1)
        done = self._batch_tensor(batch["done"], torch.float32).view(-1, 1)
        gamma_pow = self._batch_tensor(batch["gamma_pow"], torch.float32).view(-1, 1)
        weights = self._batch_tensor(batch["weights"], torch.float32).view(-1, 1)

        amp_ctx = autocast(dtype=torch.float16) if self.use_amp else nullcontext()
        with amp_ctx:
            q = self.online(scalar, patch_s, patch_l, action_feat).gather(1, action)

            with torch.no_grad():
                next_online_q = self.online(
                    next_scalar,
                    next_patch_s,
                    next_patch_l,
                    next_action_feat,
                )
                next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
                next_target_q = self.target(
                    next_scalar,
                    next_patch_s,
                    next_patch_l,
                    next_action_feat,
                ).gather(1, next_actions)

                target = reward + (1.0 - done) * gamma_pow * next_target_q

            loss_per_item = F.smooth_l1_loss(q, target, reduction="none")
            loss = (weights * loss_per_item).mean()

        td_error = (target.float() - q.float()).detach().squeeze(1)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
            self.optimizer.step()

        self.soft_update()

        return float(loss.item()), td_error.detach()

    def soft_update(self):
        tau = self.tau
        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)

    def save(self, path: str):
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)
        return path

    def save_checkpoint(self, path: str, metadata: Dict = None):
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        return path

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])

    def load_checkpoint(self, path: str) -> Dict:
        payload = torch.load(path, map_location=self.device)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        meta = payload.get("metadata", {})
        if isinstance(meta, dict):
            return meta
        return {}
