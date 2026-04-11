import sys
import os

# Ensure repository root is on sys.path so local packages can be imported
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from omegaconf import OmegaConf
from rich.console import Console
from utils.rich_print import log
import envs
from envs.env import Env


def main(config_path="configs/config1.yaml"):
    console = Console()
    cfg = OmegaConf.load(config_path)
    out_dir = "outputs/sanity_check"
    os.makedirs(out_dir, exist_ok=True)
    envs.generate_terrain(cfg, out_dir, console)
    envs.generate_ocean_current(cfg, out_dir, console)
    envs.generate_beacon_area(cfg, out_dir, console)

    # 对于本脚本，显式传入我们刚创建的 out_dir
    env = Env(cfg, run_dir=out_dir, console=console)
    obs = env.reset()
    log(console, "INFO", f"reset obs: {obs}")
    step_res = env.step(0)
    log(console, "INFO", f"step(0) -> {step_res}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/config1.yaml"
    main(cfg)
