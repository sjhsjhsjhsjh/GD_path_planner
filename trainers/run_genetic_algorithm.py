import argparse
import csv
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from envs.env import Env
from other_algorithm.genetic_algorithm import GeneticAlgorithmPlanner
from utils.rich_print import log


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


def _optimization_weights(cfg):
    reward_cfg = cfg.get("reward", {}) if hasattr(cfg, "get") else {}
    return {
        "progress_reward_scale": float(reward_cfg.get("progress_reward_scale", 12.0))
        * 0.67,
        "terrain_reward_weight": float(reward_cfg.get("terrain_reward_weight", 1.5))
        * 1.00,
        "current_reward_weight": float(reward_cfg.get("current_reward_weight", 1.2))
        * 0.83,
        "energy_reward_weight": float(reward_cfg.get("energy_reward_weight", 0.3))
        * 1.00,
    }


def rollout_once(env: Env, chromosome, max_steps: int):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    for action in chromosome[:max_steps]:
        obs, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if done:
            break

    termination_reason = getattr(env, "last_termination_reason", "unknown")
    if not done:
        termination_reason = "max_steps_reached"
    success = 1 if termination_reason == "goal_reached" else 0

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "termination_reason": termination_reason,
        "energy_remaining": float(env.robot.energy),
    }


def run_ga(
    cfg,
    run_dir: str,
    generations: int,
    population_size: int,
    elite_count: int,
    mutation_rate: float,
    max_steps: int,
    seed: int,
    log_prefix: str,
    console: Console,
):
    step_rewards_filename = f"{log_prefix}_step_rewards.csv"
    train_log_filename = f"{log_prefix}_train_log.csv"
    generation_log_filename = f"{log_prefix}_generation_log.csv"

    n_actions = cfg.env.action_dim if hasattr(cfg.env, "action_dim") else 4

    planner = GeneticAlgorithmPlanner(
        n_actions=n_actions,
        population_size=population_size,
        elite_count=elite_count,
        mutation_rate=mutation_rate,
        max_steps=max_steps,
        seed=seed,
    )

    # 评估环境：使用同一 Env.step 接口，但关闭 step 级日志
    eval_env = Env(cfg, run_dir=run_dir, console=console)
    eval_env.set_stage("optimization")
    eval_env.set_reward_weights(**_optimization_weights(cfg))
    eval_env.set_step_logging_enabled(False)

    # 回放环境：每代仅记录最优个体轨迹
    replay_env = Env(cfg, run_dir=run_dir, console=console)
    replay_env.set_stage("optimization")
    replay_env.set_reward_weights(**_optimization_weights(cfg))
    replay_env.set_step_rewards_filename(step_rewards_filename)
    replay_env.set_step_logging_enabled(True)

    step_rewards_path = os.path.join(run_dir, step_rewards_filename)
    if os.path.exists(step_rewards_path):
        os.remove(step_rewards_path)

    train_log_path = os.path.join(run_dir, train_log_filename)
    generation_log_path = os.path.join(run_dir, generation_log_filename)

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "generation",
                "total_reward",
                "steps",
                "algo",
                "success",
                "termination_reason",
                "energy_remaining",
            ]
        )

    with open(generation_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "best_reward",
                "mean_reward",
                "best_steps",
                "best_termination_reason",
                "success_count",
                "population_size",
            ]
        )

    population = planner.init_population()

    with open(train_log_path, "a", newline="", encoding="utf-8") as train_f, open(
        generation_log_path, "a", newline="", encoding="utf-8"
    ) as gen_f:
        train_writer = csv.writer(train_f)
        gen_writer = csv.writer(gen_f)

        best_overall_reward = -1e18

        for gen in range(generations):
            fitness = np.zeros(population_size, dtype=np.float64)
            steps_buf = np.zeros(population_size, dtype=np.int32)
            success_buf = np.zeros(population_size, dtype=np.int32)
            term_buf = ["unknown"] * population_size
            energy_buf = np.zeros(population_size, dtype=np.float64)

            for i in range(population_size):
                result = rollout_once(eval_env, population[i], max_steps=max_steps)
                fitness[i] = result["total_reward"]
                steps_buf[i] = result["steps"]
                success_buf[i] = result["success"]
                term_buf[i] = result["termination_reason"]
                energy_buf[i] = result["energy_remaining"]

            best_idx = int(np.argmax(fitness))
            best_reward = float(fitness[best_idx])
            mean_reward = float(np.mean(fitness))
            success_count = int(np.sum(success_buf))

            # 每代回放最优个体，生成 step 级日志（用于 dashboard）
            replay_result = rollout_once(
                replay_env,
                population[best_idx],
                max_steps=max_steps,
            )

            train_writer.writerow(
                [
                    gen,
                    gen,
                    replay_result["total_reward"],
                    replay_result["steps"],
                    "genetic_algorithm",
                    replay_result["success"],
                    replay_result["termination_reason"],
                    replay_result["energy_remaining"],
                ]
            )
            train_f.flush()

            gen_writer.writerow(
                [
                    gen,
                    best_reward,
                    mean_reward,
                    int(steps_buf[best_idx]),
                    term_buf[best_idx],
                    success_count,
                    population_size,
                ]
            )
            gen_f.flush()

            if best_reward > best_overall_reward:
                best_overall_reward = best_reward

            log(
                console,
                "INFO",
                f"Gen {gen:04d}: best={best_reward:.3f}, mean={mean_reward:.3f}, "
                f"success={success_count}/{population_size}, replay_term={replay_result['termination_reason']}",
            )

            population = planner.next_generation(population, fitness)

    log(console, "INFO", "=" * 60)
    log(console, "INFO", "遗传算法对比实验完成")
    log(console, "INFO", f"  目录: {run_dir}")
    log(console, "INFO", f"  最优 reward: {best_overall_reward:.4f}")
    log(console, "INFO", f"  {step_rewards_filename}: {step_rewards_path}")
    log(console, "INFO", f"  {train_log_filename}: {train_log_path}")
    log(console, "INFO", f"  {generation_log_filename}: {generation_log_path}")
    log(
        console,
        "INFO",
        f"可视化命令：python scripts/build_trajectory_dashboard.py --csv {step_rewards_path}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遗传算法对比实验一键入口")
    parser.add_argument("--config", type=str, default="configs/config1.yaml")
    parser.add_argument("--output_base", type=str, default="outputs")
    parser.add_argument("--src_run_dir", type=str, default=None)
    parser.add_argument("--log_prefix", type=str, default="comparison2")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--elite_count", type=int, default=6)
    parser.add_argument("--mutation_rate", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console = Console()
    cfg = OmegaConf.load(args.config)
    run_dir = args.src_run_dir or resolve_latest_run_dir(args.output_base)

    run_ga(
        cfg=cfg,
        run_dir=run_dir,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        mutation_rate=args.mutation_rate,
        max_steps=args.max_steps,
        seed=args.seed,
        log_prefix=args.log_prefix,
        console=console,
    )
