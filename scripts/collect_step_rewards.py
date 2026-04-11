"""
读取训练过程中生成的 step_rewards.csv，并将每一步 reward 的子项可视化。

用法示例:
    python scripts/collect_step_rewards.py --csv outputs/2026-03-10/17-31-30/step_rewards.csv
    python scripts/collect_step_rewards.py --csv outputs/2026-03-10/17-31-30/step_rewards.csv --episode 12

输出内容:
    - reward_plots/episode_{ep}_reward_breakdown.png: 每步 reward 分解折线图
    - reward_plots/episode_component_sums.png: 每个 episode 的各奖励项总和
    - reward_plots/episode_total_rewards.png: 每个 episode 的总 reward
    - reward_plots/step_reward_episode_summary.csv: 每个 episode 的统计汇总
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMPONENTS = [
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
    "beacon_reward",
]

LEGACY_COMPONENTS = [
    "approach_reward",
    "terrain_reward",
    "current_reward",
    "beacon_reward",
]

COMPONENT_LABELS = {
    "step_penalty": "StepCost",
    "boundary_penalty": "Boundary",
    "energy_penalty": "Energy",
    "ins_penalty": "INS",
    "revisit_penalty": "Revisit",
    "goal_reward": "Goal",
    "approach_reward": "Approach",
    "terrain_reward": "Terrain",
    "current_reward": "Current",
    "energy_reward": "EnergyStep",
    "beacon_reward": "Beacon",
}


def safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def safe_int(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default


def load_rows(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 中没有数据")

    return rows


def detect_components(fieldnames):
    if all(component in fieldnames for component in DEFAULT_COMPONENTS):
        return DEFAULT_COMPONENTS
    available = [
        component for component in DEFAULT_COMPONENTS if component in fieldnames
    ]
    if available:
        return available
    return [component for component in LEGACY_COMPONENTS if component in fieldnames]


def normalize_rows(rows, components):
    normalized = []
    for row in rows:
        item = {
            "episode": safe_int(row, "episode"),
            "step": safe_int(row, "step"),
            "step_reward": safe_float(row, "step_reward"),
            "terminated": safe_int(row, "terminated"),
            "termination_reason": row.get("termination_reason", "unknown"),
            "pos_x": safe_int(row, "pos_x", safe_int(row, "last_pos_x")),
            "pos_y": safe_int(row, "pos_y", safe_int(row, "last_pos_y")),
        }
        for component in components:
            item[component] = safe_float(row, component)
        normalized.append(item)

    normalized.sort(key=lambda item: (item["episode"], item["step"]))
    return normalized


def group_by_episode(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def compute_episode_summary(episodes, components):
    summary_rows = []
    for episode, rows in episodes.items():
        totals = {
            component: sum(row[component] for row in rows) for component in components
        }
        means = {component: (totals[component] / len(rows)) for component in components}
        total_reward = sum(row["step_reward"] for row in rows)
        termination_counts = Counter(
            row["termination_reason"] for row in rows if row.get("terminated", 0) == 1
        )
        dominant_reason = "running"
        if termination_counts:
            dominant_reason = termination_counts.most_common(1)[0][0]

        summary = {
            "episode": episode,
            "steps": len(rows),
            "total_reward": total_reward,
            "mean_step_reward": total_reward / len(rows),
            "terminated_steps": sum(row.get("terminated", 0) for row in rows),
            "dominant_termination_reason": dominant_reason,
        }
        for component in components:
            summary[f"sum_{component}"] = totals[component]
            summary[f"mean_{component}"] = means[component]
        summary_rows.append(summary)

    return summary_rows


def write_summary_csv(summary_rows, output_path):
    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_episode_breakdown(episode, rows, components, output_path):
    steps = [row["step"] for row in rows]
    total_rewards = [row["step_reward"] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(steps, total_rewards, color="black", linewidth=2, label="Step reward")
    for component in components:
        values = [row[component] for row in rows]
        axes[0].plot(
            steps,
            values,
            linewidth=1.2,
            label=COMPONENT_LABELS.get(component, component),
        )
    axes[0].set_title(f"Episode {episode} step reward breakdown")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=5, fontsize=9)

    cumulative_total = np.cumsum(total_rewards)
    axes[1].plot(
        steps, cumulative_total, color="black", linewidth=2, label="Cumulative total"
    )
    for component in components:
        cumulative_values = np.cumsum([row[component] for row in rows])
        axes[1].plot(
            steps,
            cumulative_values,
            linewidth=1.2,
            label=f"Cum {COMPONENT_LABELS.get(component, component)}",
        )
    axes[1].set_title(f"Episode {episode} cumulative reward components")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].grid(True, alpha=0.25)

    component_sums = [sum(row[component] for row in rows) for component in components]
    axes[2].bar(
        [COMPONENT_LABELS.get(component, component) for component in components],
        component_sums,
        color="steelblue",
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title(f"Episode {episode} component sums")
    axes[2].set_ylabel("Sum over episode")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].set_xlabel("Component")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_episode_component_sums(summary_rows, components, output_path):
    if not summary_rows:
        return

    episodes = [row["episode"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(16, 8))
    for component in components:
        ax.plot(
            episodes,
            [row[f"sum_{component}"] for row in summary_rows],
            marker="o",
            linewidth=1.5,
            label=COMPONENT_LABELS.get(component, component),
        )
    ax.set_title("Episode reward component sums")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward sum")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_episode_total_rewards(summary_rows, output_path):
    if not summary_rows:
        return

    episodes = [row["episode"] for row in summary_rows]
    rewards = [row["total_reward"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(episodes, rewards, color="black", linewidth=1.8)
    ax.set_title("Episode total rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_terminal_summary(summary_rows, components):
    print("分析完成。Episode 汇总如下:")
    for row in summary_rows[:10]:
        component_desc = ", ".join(
            f"{COMPONENT_LABELS.get(component, component)}={row[f'sum_{component}']:.3f}"
            for component in components
        )
        print(
            f"Episode {row['episode']}: total={row['total_reward']:.3f}, "
            f"steps={row['steps']}, term={row['dominant_termination_reason']}, {component_desc}"
        )

    if len(summary_rows) > 10:
        print(f"... 共 {len(summary_rows)} 个 episode，终端仅显示前 10 个。")


def analyze(csv_path, output_dir=None, episode=None):
    rows = load_rows(csv_path)
    components = detect_components(rows[0].keys())
    normalized_rows = normalize_rows(rows, components)
    episodes = group_by_episode(normalized_rows)

    if episode is not None:
        if episode not in episodes:
            raise ValueError(
                f"Episode {episode} 不存在，可选范围: {list(episodes.keys())[:10]}..."
            )
        episodes = {episode: episodes[episode]}

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), "reward_plots")
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = compute_episode_summary(episodes, components)
    write_summary_csv(
        summary_rows,
        os.path.join(output_dir, "step_reward_episode_summary.csv"),
    )

    # for ep, ep_rows in episodes.items():
    #     plot_episode_breakdown(
    #         ep,
    #         ep_rows,
    #         components,
    #         os.path.join(output_dir, f"episode_{ep}_reward_breakdown.png"),
    #     )

    # plot_episode_component_sums(
    #     summary_rows,
    #     components,
    #     os.path.join(output_dir, "episode_component_sums.png"),
    # )
    # plot_episode_total_rewards(
    #     summary_rows,
    #     os.path.join(output_dir, "episode_total_rewards.png"),
    # )

    print_terminal_summary(summary_rows, components)
    print(f"图像与汇总 CSV 已输出到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="step_rewards.csv 的路径")
    parser.add_argument("--output-dir", help="图像与汇总输出目录")
    parser.add_argument("--episode", type=int, help="仅分析指定 episode")
    args = parser.parse_args()
    analyze(args.csv, output_dir=args.output_dir, episode=args.episode)
