import argparse
import csv
import math
import os
import tempfile
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console

repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import envs
from agents.double_dqn import DoubleDQNAgent
from buffers.prioritized_replay_gpu import PrioritizedNStepReplayBufferGPU
from envs.env import Env
from trainers.train_qlearning import compute_stage_schedule
from utils.rich_print import log


def _fmt_float5(value):
    return round(float(value), 5)


def make_output_dir(base="outputs"):
    today = datetime.now().strftime("%Y-%m-%d")
    t = datetime.now().strftime("%H-%M-%S")
    out_dir = os.path.join(base, today, t)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def three_stage_epsilon(cfg, eps_start, eps_end, total_episodes, episode_idx):
    stage_cfg = cfg.get("stage", {}) if hasattr(cfg, "get") else {}
    p1 = float(stage_cfg.get("ep_phase1_ratio", 0.2))
    p2 = float(stage_cfg.get("ep_phase2_ratio", 0.6))
    p1 = max(0.0, min(1.0, p1))
    p2 = max(p1, min(1.0, p2))

    if total_episodes <= 1:
        return "optimization", float(eps_end)

    progress = episode_idx / max(1.0, float(total_episodes - 1))
    progress = min(1.0, max(0.0, progress))

    if progress < p1:
        return "exploration", float(eps_start)
    if progress < p2:
        alpha = (progress - p1) / max(1e-9, (p2 - p1))
        eps = float(eps_start + (eps_end - eps_start) * alpha)
        return "transition", eps
    return "optimization", float(eps_end)


def _safe_get(arr: np.ndarray, x: int, y: int, default: float = 0.0) -> float:
    if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1]:
        return float(arr[x, y])
    return float(default)


def _extract_patch(arr: np.ndarray, cx: int, cy: int, size: int, fill: float = 0.0):
    half = size // 2
    patch = np.full((size, size), fill, dtype=np.float32)

    x0 = max(0, cx - half)
    x1 = min(arr.shape[0], cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(arr.shape[1], cy + half + 1)

    px0 = x0 - (cx - half)
    py0 = y0 - (cy - half)
    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)
    patch[px0:px1, py0:py1] = arr[x0:x1, y0:y1]
    return patch


def _goal_prior_map(width: int, height: int, gx: int, gy: int, denom: float):
    xs = np.arange(width, dtype=np.float32).reshape(-1, 1)
    ys = np.arange(height, dtype=np.float32).reshape(1, -1)
    dist_x = np.abs(xs - np.float32(gx))
    dist_y = np.abs(ys - np.float32(gy))
    dist = (dist_x + dist_y).astype(np.float32)
    scale = np.float32(max(1.0, float(denom)))
    return np.exp((np.float32(-1.0) * dist) / scale).astype(np.float32)


def _action_delta(action: int) -> Tuple[int, int]:
    if action == 0:
        return (0, -1)
    if action == 1:
        return (1, 0)
    if action == 2:
        return (0, 1)
    if action == 3:
        return (-1, 0)
    return (0, 0)


def encode_observation(
    env: Env, obs: Tuple[int, int, int, int, int], small=11, large=21
):
    x, y, dx, dy, ins_error = [int(v) for v in obs]
    width = int(env.map_width)
    height = int(env.map_height)

    x_norm = x / max(1.0, width - 1)
    y_norm = y / max(1.0, height - 1)
    dx_norm = dx / max(1.0, width - 1)
    dy_norm = dy / max(1.0, height - 1)
    ins_norm = ins_error / max(1.0, float(env.ins_error_threshold))

    manhattan = abs(dx) + abs(dy)
    manhattan_norm = manhattan / max(1.0, float(env.step_total))
    energy_ratio = float(env.robot.energy) / max(1.0, float(env.now_init_energy))

    angle = math.atan2(float(dy), float(dx)) if (dx != 0 or dy != 0) else 0.0
    goal_sin = math.sin(angle)
    goal_cos = math.cos(angle)

    if 0 <= x < width and 0 <= y < height:
        ux = float(env.u[x][y])
        uy = float(env.v[x][y])
        beacon_here = 1.0 if float(env.beacon_map[x][y]) > 0 else 0.0
        depth_reward_here = float(env.depth_reward_map[x][y])
    else:
        ux = 0.0
        uy = 0.0
        beacon_here = 0.0
        depth_reward_here = 0.0

    goal_vec_norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    current_to_goal = (ux * dx + uy * dy) / goal_vec_norm

    scalar = np.array(
        [
            x_norm,
            y_norm,
            dx_norm,
            dy_norm,
            ins_norm,
            np.clip(energy_ratio, -1.0, 2.0),
            np.clip(manhattan_norm, 0.0, 2.0),
            goal_sin,
            goal_cos,
            np.clip(current_to_goal, -2.0, 2.0),
            np.clip(depth_reward_here, -1.0, 1.0),
            beacon_here,
        ],
        dtype=np.float32,
    )

    cache = env.get_observation_cache()
    terrain = cache["terrain"]
    u_arr = cache["u_norm"]
    v_arr = cache["v_norm"]
    beacon = cache["beacon"]
    goal_prior = _goal_prior_map(width, height, env.goal_x, env.goal_y, env.step_total)

    def build_patch(size):
        return np.stack(
            [
                _extract_patch(terrain, x, y, size, fill=0.0),
                _extract_patch(u_arr, x, y, size, fill=0.0),
                _extract_patch(v_arr, x, y, size, fill=0.0),
                _extract_patch(beacon, x, y, size, fill=0.0),
                _extract_patch(goal_prior, x, y, size, fill=0.0),
            ],
            axis=0,
        ).astype(np.float32)

    patch_s = build_patch(small)
    patch_l = build_patch(large)

    action_features = []
    for action in range(4):
        ddx, ddy = _action_delta(action)
        nx = x + ddx
        ny = y + ddy
        out_of_bounds = 1.0 if not (0 <= nx < width and 0 <= ny < height) else 0.0

        cur_dist = abs(env.goal_x - x) + abs(env.goal_y - y)
        nxt_dist = abs(env.goal_x - nx) + abs(env.goal_y - ny)
        dist_delta = (nxt_dist - cur_dist) / max(1.0, float(env.step_total))

        ux_n = _safe_get(u_arr, nx, ny, default=0.0)
        uy_n = _safe_get(v_arr, nx, ny, default=0.0)
        move_norm = max(1e-6, (ddx * ddx + ddy * ddy) ** 0.5)
        alignment = (ux_n * ddx + uy_n * ddy) / move_norm
        reverse_risk = max(0.0, -alignment)

        depth_next = _safe_get(env.depth_reward_map, nx, ny, default=-1.0)
        depth_risk = (1.0 - float(np.clip(depth_next, -1.0, 1.0))) / 2.0

        beacon_next = 1.0 if _safe_get(beacon, nx, ny, 0.0) > 0 else 0.0

        action_features.extend(
            [
                out_of_bounds,
                float(np.clip(dist_delta, -1.0, 1.0)),
                float(np.clip(reverse_risk, 0.0, 2.0)),
                float(np.clip(depth_risk, 0.0, 1.0)),
                beacon_next,
            ]
        )

    action_feat = np.array(action_features, dtype=np.float32)

    return {
        "scalar": scalar,
        "patch_s": patch_s,
        "patch_l": patch_l,
        "action_feat": action_feat,
        # Keep compact metadata so replay buffer can rebuild patches lazily.
        "patch_meta": np.array(
            [x, y, int(env.goal_x), int(env.goal_y), int(env.step_total)],
            dtype=np.int32,
        ),
    }


def train_double_dqn_with_cfg(
    cfg,
    out_dir,
    console=None,
    smoke_episodes=None,
    skip_env_generation=False,
):
    if console is None:
        console = Console()

    log(console, "INFO", f"Double DQN 训练输出目录: {out_dir}")

    if not skip_env_generation:
        envs.generate_terrain(cfg, out_dir, console)
        envs.generate_ocean_current(cfg, out_dir, console)
        envs.generate_beacon_area(cfg, out_dir, console)

    env = Env(cfg, run_dir=out_dir, console=console)

    dqn_cfg = cfg.get("dqn", {}) if hasattr(cfg, "get") else {}
    use_stage_schedule = bool(dqn_cfg.get("use_stage_schedule", False))
    stage_name = str(dqn_cfg.get("fixed_stage", "optimization"))

    if smoke_episodes is None:
        smoke_episodes = int(cfg.train.max_episodes)

    max_steps = int(cfg.train.max_steps_per_episode)
    action_dim = int(cfg.env.action_dim)

    eps_start = float(cfg.agent.get("epsilon_start", 1.0))
    eps_end = float(cfg.agent.get("epsilon_end", 0.01))

    patch_small = int(dqn_cfg.get("patch_small", 11))
    patch_large = int(dqn_cfg.get("patch_large", 21))
    reconstruct_patch_on_sample = bool(
        dqn_cfg.get("reconstruct_patch_on_sample", False)
    )
    cache_maps_on_gpu = bool(dqn_cfg.get("cache_maps_on_gpu", True))
    goal_prior_cache_size = int(dqn_cfg.get("goal_prior_cache_size", 64))

    obs = env.reset()
    first_state = encode_observation(env, obs, small=patch_small, large=patch_large)

    scalar_dim = int(first_state["scalar"].shape[0])
    patch_channels = int(first_state["patch_s"].shape[0])
    action_feat_dim = int(first_state["action_feat"].shape[0])

    raw_device = str(dqn_cfg.get("device", "cuda")).strip().lower()
    if raw_device == "auto":
        raw_device = "cuda"

    if not raw_device.startswith("cuda"):
        raise ValueError(
            "Double DQN 已配置为仅支持 GPU 训练/推理，请将 dqn.device 设为 cuda 或 cuda:N。"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用 CUDA 设备。请确认已安装支持 CUDA 的 PyTorch，并在 GPU 环境下运行 Double DQN。"
        )

    device = raw_device
    gpu_name = torch.cuda.get_device_name(torch.device(device))
    log(
        console,
        "INFO",
        f"DQN device={device}, cuda_available={torch.cuda.is_available()}, gpu={gpu_name}",
    )

    agent = DoubleDQNAgent(
        scalar_dim=scalar_dim,
        patch_channels=patch_channels,
        action_feat_dim=action_feat_dim,
        action_dim=action_dim,
        gamma=float(cfg.agent.gamma),
        lr=float(cfg.agent.lr),
        tau=float(dqn_cfg.get("tau", 0.005)),
        hidden_dim=int(dqn_cfg.get("hidden_dim", 128)),
        device=device,
    )

    reward_cfg = cfg.get("reward", {}) if hasattr(cfg, "get") else {}
    approach_reward_weight = float(reward_cfg.get("approach_reward_weight", 1.0))
    terrain_reward_weight = float(reward_cfg.get("terrain_reward_weight", 1.0))
    current_reward_weight = float(reward_cfg.get("current_reward_weight", 1.2))

    buffer = PrioritizedNStepReplayBufferGPU(
        capacity=int(cfg.buffer.capacity),
        alpha=float(dqn_cfg.get("per_alpha", 0.6)),
        n_step=int(dqn_cfg.get("n_step", 3)),
        gamma=float(cfg.agent.gamma),
        device=device,
        observation_cache=env.get_observation_cache(),
        patch_small=patch_small,
        patch_large=patch_large,
        reconstruct_patch_on_sample=reconstruct_patch_on_sample,
        cache_maps_on_gpu=cache_maps_on_gpu,
        goal_prior_cache_size=goal_prior_cache_size,
    )

    batch_size = int(cfg.train.batch_size)
    warmup_steps = int(dqn_cfg.get("warmup_steps", 2000))
    updates_per_step = int(dqn_cfg.get("updates_per_step", 1))
    grad_clip = float(dqn_cfg.get("grad_clip", 10.0))

    train_mode = str(cfg.train.get("mode", "normal")).strip().lower()
    if train_mode not in {"normal", "resume"}:
        raise ValueError("train.mode 仅支持 normal 或 resume")

    resume_checkpoint_path = str(cfg.train.get("resume_checkpoint_path", "")).strip()

    ckpt_cfg = cfg.get("checkpoint", {}) if hasattr(cfg, "get") else {}
    periodic_interval = int(ckpt_cfg.get("periodic_interval", 100))
    periodic_filename = str(ckpt_cfg.get("periodic_filename", "dqn_model_periodic.pt"))
    save_buffer = bool(ckpt_cfg.get("save_buffer", True))

    beta_start = float(dqn_cfg.get("per_beta_start", 0.4))
    beta_end = float(dqn_cfg.get("per_beta_end", 1.0))

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    periodic_ckpt_path = os.path.join(ckpt_dir, periodic_filename)

    start_episode = 0
    global_step = 0
    best_reward = -1e18

    if train_mode == "resume":
        if not resume_checkpoint_path:
            raise ValueError(
                "train.mode=resume 时必须提供 train.resume_checkpoint_path"
            )
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(
                f"resume checkpoint 不存在: {resume_checkpoint_path}"
            )

        log(console, "INFO", f"[DQN] 断点续训模式，加载: {resume_checkpoint_path}")
        metadata = agent.load_checkpoint(resume_checkpoint_path)

        start_episode = int(metadata.get("episode", -1)) + 1
        global_step = int(metadata.get("global_step", 0))
        best_reward = float(metadata.get("best_reward", -1e18))

        if save_buffer:
            buf_state = metadata.get("buffer_state")
            if buf_state is None:
                log(
                    console,
                    "WARN",
                    "[DQN] 配置要求恢复buffer，但checkpoint中未找到buffer_state，将冷启动buffer。",
                )
            else:
                try:
                    buffer.import_state(buf_state, with_storage=True)
                    log(
                        console,
                        "INFO",
                        f"[DQN] 已恢复 replay buffer，当前长度={len(buffer)}",
                    )
                except Exception as err:
                    raise RuntimeError(f"恢复 replay buffer 失败: {err}") from err

        np_rng_state = metadata.get("numpy_rng_state")
        torch_rng_state = metadata.get("torch_rng_state")
        if np_rng_state is not None:
            try:
                np.random.set_state(np_rng_state)
            except Exception:
                log(console, "WARN", "[DQN] numpy RNG state 恢复失败，已忽略。")
        if torch_rng_state is not None:
            try:
                if torch.is_tensor(torch_rng_state):
                    torch.set_rng_state(
                        torch_rng_state.to(dtype=torch.uint8, device="cpu")
                    )
                else:
                    torch.set_rng_state(
                        torch.as_tensor(torch_rng_state, dtype=torch.uint8)
                    )
            except Exception:
                log(console, "WARN", "[DQN] torch RNG state 恢复失败，已忽略。")

        if start_episode >= int(smoke_episodes):
            log(
                console,
                "WARN",
                f"[DQN] resume起始轮次({start_episode})已超过或等于max_episodes({int(smoke_episodes)})，将不再训练。",
            )

    csv_path = os.path.join(out_dir, "train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "episode",
                "total_reward",
                "steps",
                "epsilon",
                "stage",
                "progress_reward_scale",
                "terrain_reward_weight",
                "current_reward_weight",
                "energy_reward_weight",
                "success",
                "termination_reason",
                "energy_remaining",
                "loss_mean",
                "buffer_size",
            ]
        )

        device_checked = False

        for ep in range(start_episode, int(smoke_episodes)):
            if use_stage_schedule:
                stage_name, stage_weights = compute_stage_schedule(
                    cfg, ep, smoke_episodes
                )
            else:
                stage_weights = {
                    "progress_reward_scale": float(cfg.reward.progress_reward_scale),
                    "approach_reward_weight": approach_reward_weight,
                    "terrain_reward_weight": terrain_reward_weight,
                    "current_reward_weight": current_reward_weight,
                    "energy_reward_weight": float(cfg.reward.energy_reward_weight),
                }

            stage_weights["approach_reward_weight"] = approach_reward_weight
            stage_weights["terrain_reward_weight"] = terrain_reward_weight
            stage_weights["current_reward_weight"] = current_reward_weight

            env.set_stage(stage_name)
            env.set_reward_weights(**stage_weights)

            obs = env.reset()
            state = agent.state_to_device(
                encode_observation(env, obs, small=patch_small, large=patch_large)
            )

            if not device_checked:
                for k, v in state.items():
                    if not (torch.is_tensor(v) and v.device.type == "cuda"):
                        raise RuntimeError(f"DQN state tensor `{k}` 未位于 CUDA 设备。")

            epsilon_stage, epsilon = three_stage_epsilon(
                cfg,
                eps_start,
                eps_end,
                smoke_episodes,
                ep,
            )
            beta = beta_start + (beta_end - beta_start) * (
                ep / max(1, smoke_episodes - 1)
            )

            ep_reward = 0.0
            ep_steps = 0
            losses = []

            for _ in range(max_steps):
                action = agent.select_action(state, epsilon)
                next_obs, reward, done, _ = env.step(action)
                next_state = agent.state_to_device(
                    encode_observation(
                        env, next_obs, small=patch_small, large=patch_large
                    )
                )

                buffer.add(state, action, float(reward), next_state, bool(done))
                state = next_state
                ep_reward += float(reward)
                ep_steps += 1
                global_step += 1

                if len(buffer) >= max(batch_size, warmup_steps):
                    for _ in range(max(1, updates_per_step)):
                        batch = buffer.sample(batch_size, beta=beta)
                        if not device_checked:
                            required_keys = [
                                "state_scalar",
                                "state_patch_s",
                                "state_patch_l",
                                "state_action_feat",
                                "next_scalar",
                                "next_patch_s",
                                "next_patch_l",
                                "next_action_feat",
                                "action",
                                "reward",
                                "done",
                                "gamma_pow",
                                "weights",
                                "indices",
                            ]
                            for key in required_keys:
                                value = batch[key]
                                if not (
                                    torch.is_tensor(value)
                                    and value.device.type == "cuda"
                                ):
                                    raise RuntimeError(
                                        f"DQN batch tensor `{key}` 未位于 CUDA 设备。"
                                    )
                            device_checked = True
                        loss, td_errors = agent.train_step(batch, grad_clip=grad_clip)
                        buffer.update_priorities(batch["indices"], td_errors)
                        losses.append(loss)

                if done:
                    break

            termination_reason = getattr(env, "last_termination_reason", "unknown")
            success = 1 if termination_reason == "goal_reached" else 0
            loss_mean = float(np.mean(losses)) if losses else 0.0

            writer.writerow(
                [
                    ep,
                    _fmt_float5(ep_reward),
                    ep_steps,
                    _fmt_float5(epsilon),
                    stage_name,
                    _fmt_float5(stage_weights["progress_reward_scale"]),
                    _fmt_float5(stage_weights["terrain_reward_weight"]),
                    _fmt_float5(stage_weights["current_reward_weight"]),
                    _fmt_float5(stage_weights["energy_reward_weight"]),
                    success,
                    termination_reason,
                    _fmt_float5(env.robot.energy),
                    _fmt_float5(loss_mean),
                    len(buffer),
                ]
            )
            csvfile.flush()

            log(
                console,
                "INFO",
                f"[DQN] Ep {ep}: reward={ep_reward:.3f}, steps={ep_steps}, eps={epsilon:.4f}, eps_stage={epsilon_stage}, "
                f"term={termination_reason}, loss={loss_mean:.5f}, buffer={len(buffer)}",
            )

            if ep_reward > best_reward:
                best_reward = ep_reward
                model_path = os.path.join(out_dir, "dqn_model_best.pt")
                agent.save(model_path)
                log(console, "INFO", f"Saved best DQN model: {model_path}")

            if periodic_interval > 0 and ((ep + 1) % periodic_interval == 0):
                ckpt_meta = {
                    "episode": int(ep),
                    "global_step": int(global_step),
                    "best_reward": float(best_reward),
                    "epsilon": float(epsilon),
                    "beta": float(beta),
                    "save_buffer": bool(save_buffer),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state().to("cpu"),
                }
                if save_buffer:
                    ckpt_meta["buffer_state"] = buffer.export_state(
                        include_storage=True
                    )

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".tmp", delete=False, dir=ckpt_dir
                ) as tmpf:
                    tmp_path = tmpf.name
                try:
                    agent.save_checkpoint(tmp_path, metadata=ckpt_meta)
                    os.replace(tmp_path, periodic_ckpt_path)
                    log(
                        console,
                        "INFO",
                        f"[DQN] 周期checkpoint已更新(覆盖): ep={ep + 1}, path={periodic_ckpt_path}",
                    )
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

    env.close()
    log(console, "INFO", f"Double DQN训练完成，日志保存在 {csv_path}")
    return out_dir


def train(cfg_path, smoke_episodes=50):
    console = Console()
    cfg = OmegaConf.load(cfg_path)

    out_dir = make_output_dir()
    return train_double_dqn_with_cfg(
        cfg,
        out_dir,
        console=console,
        smoke_episodes=smoke_episodes,
        skip_env_generation=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config1.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    train(args.config, smoke_episodes=args.episodes)
