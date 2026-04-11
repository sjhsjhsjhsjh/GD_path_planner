import csv
import sys
from pathlib import Path


def main(log_path):
    rows = list(csv.DictReader(open(log_path)))
    rewards = [float(r["total_reward"]) for r in rows]
    steps = [int(r["steps"]) for r in rows]
    episodes = len(rewards)
    avg_reward = sum(rewards) / episodes
    var = sum((x - avg_reward) ** 2 for x in rewards) / episodes
    std_reward = var**0.5
    avg_steps = sum(steps) / episodes
    successes = sum(1 for x in rewards if x > 0)
    print(f"episodes: {episodes}")
    print(f"avg_reward: {avg_reward:.4f}")
    print(f"std_reward: {std_reward:.4f}")
    print(f"avg_steps: {avg_steps:.2f}")
    print(f"successes: {successes}")


if __name__ == "__main__":
    log = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "outputs/2026-03-10/16-44-54/train_log.csv"
    )
    if not Path(log).exists():
        print("log not found:", log)
        sys.exit(1)
    main(log)
