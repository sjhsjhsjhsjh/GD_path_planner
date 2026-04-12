import hydra
from omegaconf import DictConfig
import envs
from hydra.core.hydra_config import HydraConfig
from rich.console import Console
from utils import log as log
import os
import shutil
from trainers.train_qlearning import train_with_cfg
from trainers.train_double_dqn import train_double_dqn_with_cfg
from scripts.collect_step_rewards import analyze as analyze_step_rewards
from scripts.collect_summary import main as collect_train_summary

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ！！！！！！！注意：运行后会清除之前的输出文件夹，请提前备份有价值的实验结果！！！！！！！
# 默认只保留最近的2次运行结果，删除更早的输出文件夹
def clean_old_output(console, workspace_dir=None):
    if workspace_dir is None:
        workspace_dir = DEFAULT_OUTPUTS_DIR
    if not os.path.exists(workspace_dir):
        return

    experiment_dirs = []
    for day_dir in os.listdir(workspace_dir):
        day_dir_path = os.path.join(workspace_dir, day_dir)
        if not os.path.isdir(day_dir_path):
            continue
        for run_dir in os.listdir(day_dir_path):
            run_dir_path = os.path.join(day_dir_path, run_dir)
            if os.path.isdir(run_dir_path):
                experiment_dirs.append(run_dir_path)

    if len(experiment_dirs) <= 2:
        return

    experiment_dirs.sort(key=os.path.getmtime, reverse=True)
    stale_dirs = experiment_dirs[2:]

    for stale_dir in stale_dirs:
        shutil.rmtree(stale_dir)
        log(console, "WARN", f"已删除旧的输出文件夹: {stale_dir}")

    for day_dir in os.listdir(workspace_dir):
        day_dir_path = os.path.join(workspace_dir, day_dir)
        if os.path.isdir(day_dir_path) and not os.listdir(day_dir_path):
            os.rmdir(day_dir_path)
            log(console, "WARN", f"已删除空的日期目录: {day_dir_path}")


def postprocess_outputs(run_dir, console):
    step_reward_csv = os.path.join(run_dir, "step_rewards.csv")
    train_log_csv = os.path.join(run_dir, "train_log.csv")

    if os.path.exists(step_reward_csv):
        analyze_step_rewards(step_reward_csv)
        log(
            console,
            "INFO",
            f"reward 分析图已输出到: {os.path.join(run_dir, 'reward_plots')}",
        )
    else:
        log(console, "WARN", f"未找到 step reward 日志: {step_reward_csv}")

    if os.path.exists(train_log_csv):
        collect_train_summary(train_log_csv)
    else:
        log(console, "WARN", f"未找到训练日志: {train_log_csv}")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config1",
)
def main(cfg: DictConfig):
    # 运行库初始化
    run_dir = HydraConfig.get().runtime.output_dir
    console = Console()
    log(console, "INFO", "当前运行目录: " + run_dir)

    # 清理旧的输出文件夹
    clean_old_output(console)

    # 生成环境信息：地形和洋流
    envs.generate_terrain(cfg, run_dir, console)
    envs.generate_ocean_current(cfg, run_dir, console)

    # 生成信标信息
    envs.generate_beacon_area(cfg, run_dir, console)

    # 直接实例化 Agent 并开始训练
    algorithm = str(cfg.train.get("algorithm", "q_learning")).lower()
    log(console, "INFO", f"当前算法: {algorithm}")

    if algorithm == "q_learning":
        train_with_cfg(
            cfg,
            run_dir,
            console=console,
            smoke_episodes=cfg.train.total_steps,
            skip_env_generation=True,
        )
    elif algorithm == "double_dqn":
        train_double_dqn_with_cfg(
            cfg,
            run_dir,
            console=console,
            smoke_episodes=cfg.train.total_steps,
            skip_env_generation=True,
        )
    else:
        raise ValueError(
            f"不支持的算法: {algorithm}，可选值为 q_learning 或 double_dqn"
        )

    # 训练完成后，自动生成分析图像和汇总
    postprocess_outputs(run_dir, console)


if __name__ == "__main__":
    main()
