import os
from typing import Dict, Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from rich_print import log

try:
    from hydra.utils import get_original_cwd
    from hydra.core.hydra_config import HydraConfig
    import hydra

    HYDRA_AVAILABLE = True
except Exception:
    HYDRA_AVAILABLE = False


class TensorBoardLogger:
    """
    与 Hydra 运行目录兼容的 TensorBoard 日志类

    设计目标：
        1. 自动写入 Hydra outputs 目录
        2. 支持自定义 runs 子目录
        3. 强化学习常用日志接口统一
    """

    def __init__(
        self,
        console,
        run_dir_,
        sub_dir: str = "tensorboard",
        use_hydra_run_dir: bool = True,
    ):
        """
        init TensorBoardLogger with Hydra run directory

        :param console : rich Console 对象，用于打印日志
        :param run_dir_ : hydra run_dir
        :param sub_dir : TensorBoard 子目录名
        :param use_hydra_run_dir : True -> 写入当前 hydra run.dir
                                    False -> 写入原始工程目录
        """

        log_dir = self._resolve_logdir(console, run_dir_, sub_dir, use_hydra_run_dir)
        log(
            console,
            "INFO",
            f"TensorBoardLogger 使用 Hydra 运行目录: {log_dir}",
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logdir = {log_dir}")

    # ==========================================================
    # Hydra 目录解析
    # ==========================================================
    def _resolve_logdir(self, console, run_dir_, sub_dir, use_hydra_run_dir):
        if HYDRA_AVAILABLE:
            try:
                if use_hydra_run_dir:
                    run_dir = run_dir_
                    return os.path.join(run_dir, sub_dir)
            except Exception:
                raise RuntimeError(
                    "Hydra 目录解析失败，请确保在 Hydra 运行环境中使用 TensorBoardLogger，或设置 use_hydra_run_dir=False"
                )

        # fallback
        return os.path.join(os.getcwd(), "runs", sub_dir)

    # ==========================================================
    # 标量
    # ==========================================================
    def log_scalar(self, name: str, value: float, step: int):
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, name: str, values: Dict[str, float], step: int):
        self.writer.add_scalars(name, values, step)

    # ==========================================================
    # 图像
    # ==========================================================
    def log_image(self, name: str, img: np.ndarray, step: int, normalize: bool = True):
        if normalize:
            img = self._normalize(img)

        if img.ndim == 2:
            self.writer.add_image(name, img, step, dataformats="HW")
        elif img.ndim == 3:
            if img.shape[0] in [1, 3]:
                self.writer.add_image(name, img, step, dataformats="CHW")
            else:
                self.writer.add_image(name, img, step, dataformats="HWC")
        else:
            raise ValueError("Unsupported image shape")

    def log_figure(self, name: str, fig, step: int, close: bool = True):
        self.writer.add_figure(name, fig, step)
        if close:
            plt.close(fig)

    # ==========================================================
    # histogram
    # ==========================================================
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        self.writer.add_histogram(name, values, step)

    # ==========================================================
    # 文本
    # ==========================================================
    def log_text(self, name: str, text: str, step: int):
        self.writer.add_text(name, text, step)

    # ==========================================================
    # RL 专用
    # ==========================================================
    def log_map_overlay(
        self, name: str, base_map: np.ndarray, overlay: Optional[np.ndarray], step: int
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(base_map, cmap="viridis", origin="lower")
        if overlay is not None:
            ax.imshow(overlay, cmap="hot", alpha=0.4, origin="lower")
        ax.set_title(name)
        self.log_figure(name, fig, step)

    def log_q_values(self, q_values: np.ndarray, step: int):
        self.log_histogram("Q_values", q_values, step)

    # ==========================================================
    # 工具
    # ==========================================================
    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    @staticmethod
    def _normalize(arr: np.ndarray):
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
