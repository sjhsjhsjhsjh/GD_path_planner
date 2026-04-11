"""
run_comparison.py — 对比算法独立运行入口

与 trainers/train_qlearning.py 完全独立，互不影响。
使用与 Q-learning 相同的 Env 接口（env.step()）计算每步代价，
保证对比实验公平性。

用法示例：
    python trainers/run_comparison.py \
        --src_run_dir outputs/2026-03-15/15-58-35 \
        --algo random_rollout \
        --episodes 100 \
        --max_steps 500 \
        --config configs/config1.yaml \
        --seed 42
"""

import argparse
import csv
import os
import shutil
import sys

from omegaconf import OmegaConf
from rich.console import Console
from typing import Optional

# 确保项目根目录在 sys.path 中
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from envs.env import Env
from utils.rich_print import log


# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------


def _latest_time_key(name: str):
    prefix = name[:8]
    parts = prefix.split("-")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return (-1, -1, -1)


def resolve_latest_run_dir(base: str) -> str:
    if not os.path.isdir(base):
        raise FileNotFoundError(f"输出目录不存在: {base}")

    date_dirs = [
        name
        for name in os.listdir(base)
        if os.path.isdir(os.path.join(base, name)) and len(name) == 10
    ]
    if not date_dirs:
        raise FileNotFoundError(f"{base} 下没有日期目录")

    latest_date = max(date_dirs)
    latest_date_dir = os.path.join(base, latest_date)
    run_dirs = [
        name
        for name in os.listdir(latest_date_dir)
        if os.path.isdir(os.path.join(latest_date_dir, name))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"{latest_date_dir} 下没有运行目录")

    latest_run = max(run_dirs, key=lambda name: (_latest_time_key(name), name))
    return os.path.join(latest_date_dir, latest_run)


# ---------------------------------------------------------------------------
# Agent 工厂
# ---------------------------------------------------------------------------


def build_agent(algo: str, cfg, seed: int):
    if algo == "random_rollout":
        from other_algorithm.random_rollout import RandomRolloutAgent

        n_actions = cfg.env.action_dim if hasattr(cfg.env, "action_dim") else 4
        return RandomRolloutAgent(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"未知对比算法: {algo}，目前支持: random_rollout")


# ---------------------------------------------------------------------------
# 主运行逻辑
# ---------------------------------------------------------------------------


def run_comparison(
    cfg,
    src_run_dir: str,
    out_dir: str,
    algo: str,
    episodes: int,
    max_steps: int,
    seed: int,
    log_prefix: str,
    console: Optional[Console] = None,
):
    if console is None:
        console = Console()

    log(console, "INFO", f"对比算法: {algo}")
    log(console, "INFO", f"源环境目录: {src_run_dir}")
    log(console, "INFO", f"输出目录: {out_dir}")

    step_rewards_filename = f"{log_prefix}_step_rewards.csv"
    train_log_filename = f"{log_prefix}_train_log.csv"

    # 若源目录与输出目录不同，则复制环境网格；相同时直接复用
    src_grid = os.path.join(src_run_dir, "environment_grid.csv")
    if not os.path.exists(src_grid):
        raise FileNotFoundError(
            f"找不到环境网格文件: {src_grid}\n"
            f"请确认 --src_run_dir 指向含有 environment_grid.csv 的 Q-learning 输出目录。"
        )
    dst_grid = os.path.join(out_dir, "environment_grid.csv")
    if os.path.abspath(src_run_dir) != os.path.abspath(out_dir):
        shutil.copy2(src_grid, dst_grid)
        log(console, "INFO", f"已复制环境网格: {src_grid} -> {dst_grid}")
    else:
        log(console, "INFO", f"复用当前运行目录中的环境网格: {dst_grid}")

    # 初始化环境（跳过地图生成，直接读取复制过来的网格）
    env = Env(cfg, run_dir=out_dir, console=console)
    env.set_step_rewards_filename(step_rewards_filename)

    step_rewards_path = os.path.join(out_dir, step_rewards_filename)
    if os.path.exists(step_rewards_path):
        os.remove(step_rewards_path)

    # 构建对比 agent
    agent = build_agent(algo, cfg, seed)

    # 获取 optimization 阶段的固定奖励权重（与 Q-learning 最终阶段保持一致，保证对比公平）
    reward_cfg = cfg.get("reward", {}) if hasattr(cfg, "get") else {}
    optimization_weights = {
        "progress_reward_scale": float(reward_cfg.get("progress_reward_scale", 12.0))
        * 0.67,
        "terrain_reward_weight": float(reward_cfg.get("terrain_reward_weight", 1.5))
        * 1.00,
        "current_reward_weight": float(reward_cfg.get("current_reward_weight", 1.2))
        * 0.83,
        "energy_reward_weight": float(reward_cfg.get("energy_reward_weight", 0.3))
        * 1.00,
    }

    # 初始化 comparison train_log.csv
    train_log_path = os.path.join(out_dir, train_log_filename)
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "total_reward",
                "steps",
                "algo",
                "success",
                "termination_reason",
                "energy_remaining",
            ]
        )

    best_reward = -1e9
    success_count = 0

    with open(train_log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)

        for ep in range(episodes):
            env.set_stage("optimization")
            env.set_reward_weights(**optimization_weights)

            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            for _ in range(max_steps):
                action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                if done:
                    break

            termination_reason = getattr(env, "last_termination_reason", "unknown")
            if not done:
                termination_reason = "max_steps_reached"
            success = 1 if termination_reason == "goal_reached" else 0
            energy_remaining = float(env.robot.energy)

            writer.writerow(
                [
                    ep,
                    total_reward,
                    steps,
                    algo,
                    success,
                    termination_reason,
                    energy_remaining,
                ]
            )
            log_file.flush()

            if success:
                success_count += 1
            if total_reward > best_reward:
                best_reward = total_reward

            log(
                console,
                "INFO",
                f"Ep {ep:03d}: reward={total_reward:.3f}, steps={steps}, "
                f"term={termination_reason}, energy={energy_remaining:.2f}",
            )

    # 汇总打印
    log(console, "INFO", "=" * 60)
    log(console, "INFO", f"对比实验完成 | 算法: {algo}")
    log(console, "INFO", f"  总 episode 数 : {episodes}")
    log(
        console,
        "INFO",
        f"  成功率        : {success_count}/{episodes} ({100*success_count/episodes:.1f}%)",
    )
    log(console, "INFO", f"  最优 episode reward: {best_reward:.4f}")
    log(console, "INFO", f"  {step_rewards_filename} -> {step_rewards_path}")
    log(console, "INFO", f"  {train_log_filename} -> {train_log_path}")
    log(console, "INFO", "=" * 60)
    log(
        console,
        "INFO",
        f"可视化命令：python scripts/build_trajectory_dashboard.py --csv {step_rewards_path}",
    )

    return out_dir


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比算法独立运行入口")
    parser.add_argument(
        "--src_run_dir",
        type=str,
        default=None,
        help="含有 environment_grid.csv 的 Q-learning 输出目录（用于复用相同地图）",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="random_rollout",
        choices=["random_rollout"],
        help="对比算法名称（默认: random_rollout）",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="运行的 episode 数量"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="每个 episode 最大步数"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config1.yaml", help="配置文件路径"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    parser.add_argument("--output_base", type=str, default="outputs", help="输出根目录")
    parser.add_argument(
        "--log_prefix",
        type=str,
        default="comparison1",
        help="对比实验日志前缀",
    )
    args = parser.parse_args()

    _console = Console()
    _cfg = OmegaConf.load(args.config)
    _resolved_run_dir = args.src_run_dir or resolve_latest_run_dir(args.output_base)

    run_comparison(
        cfg=_cfg,
        src_run_dir=_resolved_run_dir,
        out_dir=_resolved_run_dir,
        algo=args.algo,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        log_prefix=args.log_prefix,
        console=_console,
    )
