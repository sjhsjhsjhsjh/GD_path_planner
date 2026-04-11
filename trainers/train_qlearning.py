import os
import csv
import argparse
import math
from datetime import datetime

import sys
from omegaconf import OmegaConf
from rich.console import Console

# Ensure repository root is on sys.path so local packages can be imported
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import envs
from envs.env import Env
from agents.q_learning import QLearningAgent
from utils.rich_print import log


def _fmt_float5(value):
    return round(float(value), 5)


def make_output_dir(base="outputs"):
    today = datetime.now().strftime("%Y-%m-%d")
    t = datetime.now().strftime("%H-%M-%S")
    out_dir = os.path.join(base, today, t)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def exponential_epsilon(eps_start, eps_end, decay_steps, episode):
    if decay_steps <= 0:
        return eps_end
    return eps_end + (eps_start - eps_end) * math.exp(-episode / decay_steps)


def compute_stage_schedule(cfg, ep, total_eps):
    stage_cfg = cfg.get("stage", {}) if hasattr(cfg, "get") else {}
    p1 = float(stage_cfg.get("ep_phase1_ratio", 0.2))
    p2 = float(stage_cfg.get("ep_phase2_ratio", 0.6))
    p1 = max(0.0, min(1.0, p1))
    p2 = max(p1, min(1.0, p2))
    progress = ep / max(1, total_eps - 1)

    reward_cfg = cfg.get("reward", {}) if hasattr(cfg, "get") else {}
    base = {
        "progress_reward_scale": float(reward_cfg.get("progress_reward_scale", 12.0)),
        "terrain_reward_weight": float(reward_cfg.get("terrain_reward_weight", 1.5)),
        "current_reward_weight": float(reward_cfg.get("current_reward_weight", 1.2)),
        "energy_reward_weight": float(reward_cfg.get("energy_reward_weight", 0.05)),
    }

    if progress < p1:
        mult = {
            "progress_reward_scale": 1.00,
            "terrain_reward_weight": 0.20,
            "current_reward_weight": 0.15,
            "energy_reward_weight": 0.20,
        }
        stage = "exploration"
    elif progress < p2:
        alpha = (progress - p1) / max(1e-9, (p2 - p1))
        mult = {
            "progress_reward_scale": 1.0 - 0.33 * alpha,
            "terrain_reward_weight": 0.20 + 0.80 * alpha,
            "current_reward_weight": 0.15 + 0.68 * alpha,
            "energy_reward_weight": 0.20 + 0.80 * alpha,
        }
        stage = "transition"
    else:
        mult = {
            "progress_reward_scale": 0.67,
            "terrain_reward_weight": 1.00,
            "current_reward_weight": 0.83,
            "energy_reward_weight": 1.00,
        }
        stage = "optimization"

    weights = {k: base[k] * mult[k] for k in base.keys()}
    return stage, weights


def train_with_cfg(
    cfg,
    out_dir,
    console=None,
    smoke_episodes=None,
    skip_env_generation=False,
):
    if console is None:
        console = Console()

    log(console, "INFO", f"训练输出目录: {out_dir}")

    if not skip_env_generation:
        envs.generate_terrain(cfg, out_dir, console)
        envs.generate_ocean_current(cfg, out_dir, console)
        envs.generate_beacon_area(cfg, out_dir, console)

    # 环境读写统一使用本次训练输出目录，避免读取旧地图或把日志写到工作目录
    env = Env(cfg, run_dir=out_dir, console=console)

    width = env.map_width
    height = env.map_height
    n_actions = cfg.env.action_dim if hasattr(cfg.env, "action_dim") else 4
    ins_error_threshold = (
        cfg.env.ins_error_threshold if hasattr(cfg.env, "ins_error_threshold") else 10
    )

    alpha = cfg.agent.alpha if hasattr(cfg.agent, "alpha") else 0.01
    gamma = cfg.agent.gamma if hasattr(cfg.agent, "gamma") else 0.99
    eps_start = cfg.agent.epsilon_start if hasattr(cfg.agent, "epsilon_start") else 1.0
    eps_end = cfg.agent.epsilon_end if hasattr(cfg.agent, "epsilon_end") else 0.01
    if smoke_episodes is None:
        smoke_episodes = (
            cfg.train.max_episodes if hasattr(cfg.train, "max_episodes") else 50
        )
    decay_steps = (
        cfg.agent.epsilon_decay_steps
        if hasattr(cfg.agent, "epsilon_decay_steps")
        else int(0.8 * smoke_episodes)
    )

    agent = QLearningAgent(
        width,
        height,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        ins_error_threshold=ins_error_threshold,
    )

    max_steps = (
        cfg.train.max_steps_per_episode
        if hasattr(cfg.train, "max_steps_per_episode")
        else 500
    )

    # 日志 CSV
    csv_path = os.path.join(out_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
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
            ]
        )

        best_reward = -1e9

        for ep in range(smoke_episodes):
            stage_name, stage_weights = compute_stage_schedule(cfg, ep, smoke_episodes)
            env.set_stage(stage_name)
            env.set_reward_weights(**stage_weights)

            obs = env.reset()
            # obs is (x,y,dx,dy,ins_error)
            state = (obs[0], obs[1], obs[2], obs[3], obs[4])
            total_reward = 0.0
            steps = 0
            start_pos = (obs[0], obs[1])
            goal_pos = (env.goal_x, env.goal_y)

            epsilon = exponential_epsilon(eps_start, eps_end, decay_steps, ep)

            for t in range(max_steps):
                action = agent.select_action(state, epsilon)
                next_obs, reward, done, info = env.step(action)
                # next_obs is (x,y,dx,dy,ins_error)
                next_state = (
                    next_obs[0],
                    next_obs[1],
                    next_obs[2],
                    next_obs[3],
                    next_obs[4],
                )
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

            success = (
                1
                if getattr(env, "last_termination_reason", "") == "goal_reached"
                else 0
            )
            termination_reason = getattr(env, "last_termination_reason", "unknown")
            writer.writerow(
                [
                    ep,
                    _fmt_float5(total_reward),
                    steps,
                    _fmt_float5(epsilon),
                    stage_name,
                    _fmt_float5(stage_weights["progress_reward_scale"]),
                    _fmt_float5(stage_weights["terrain_reward_weight"]),
                    _fmt_float5(stage_weights["current_reward_weight"]),
                    _fmt_float5(stage_weights["energy_reward_weight"]),
                    success,
                    termination_reason,
                    _fmt_float5(env.robot.energy),
                ]
            )
            csvfile.flush()
            log(
                console,
                "INFO",
                f"Ep {ep}: reward={total_reward:.3f}, steps={steps}, eps={epsilon:.4f}, stage={stage_name}, term={termination_reason}",
            )

            if total_reward > best_reward:
                best_reward = total_reward
                q_path = os.path.join(out_dir, "q_table_best.npz")
                saved_path = agent.save(q_path)
                log(console, "INFO", f"Saved best Q table: {saved_path}")

    env.close()
    log(console, "INFO", f"训练完成，日志保存在 {csv_path}")
    return out_dir


def train(cfg_path, smoke_episodes=50):
    console = Console()
    cfg = OmegaConf.load(cfg_path)

    out_dir = make_output_dir()
    return train_with_cfg(
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
