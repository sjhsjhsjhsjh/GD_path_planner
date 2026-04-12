import os
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.beacon_layout import load_beacon_layout, resolve_beacon_config_path
from envs.dynamic_ins_threshold import compute_dynamic_ins_threshold
from envs.environment_grid import environment_grid_path, load_environment_grid


TESTER_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "configs", "ins_threshold_visual_tester.yaml"
)


def _resolve_project_path(path_value: str, fallback: Optional[str] = None) -> str:
    if path_value:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_value))
    if fallback:
        return os.path.normpath(os.path.join(PROJECT_ROOT, fallback))
    return PROJECT_ROOT


def _load_cfg(cfg_path: str):
    return OmegaConf.load(cfg_path)


def _load_tester_cfg():
    if not os.path.exists(TESTER_CONFIG_PATH):
        raise FileNotFoundError(f"Tester config file not found: {TESTER_CONFIG_PATH}")
    return _load_cfg(TESTER_CONFIG_PATH)


def _load_terrain(rows: int, cols: int, terrain_run_dir: Optional[str]) -> np.ndarray:
    if terrain_run_dir:
        csv_path = environment_grid_path(terrain_run_dir)
        if os.path.exists(csv_path):
            try:
                layers = load_environment_grid(terrain_run_dir)
                terrain = np.asarray(layers.get("seafloor"), dtype=np.float32)
                if terrain.shape == (rows, cols):
                    return terrain
            except Exception:
                pass
    return np.zeros((rows, cols), dtype=np.float32)


def _resolve_beacon_override(base_cfg, tester_cfg):
    beacon_override = str(tester_cfg.tester.get("beacon_config_path", "")).strip()
    if not beacon_override:
        return base_cfg

    cfg_copy = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg_copy.beacon.beacon_config_path = beacon_override
    return cfg_copy


class InsThresholdVisualTester:
    def __init__(
        self,
        rows: int,
        cols: int,
        terrain: np.ndarray,
        beacons: List[Dict[str, int]],
        default_threshold: int,
        dynamic_enabled: bool,
        corridor_width: float,
        safety_factor: float,
        min_threshold: int,
        max_threshold: int,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.terrain = terrain
        self.beacons = beacons

        self.default_threshold = int(default_threshold)
        self.dynamic_enabled = bool(dynamic_enabled)
        self.corridor_width = float(corridor_width)
        self.safety_factor = float(safety_factor)
        self.min_threshold = int(min_threshold)
        self.max_threshold = int(max_threshold)

        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.last_threshold: Optional[int] = None
        self.last_details: Optional[Dict] = None

        self.status_artist = None
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw("Left click to pick start and goal. Right click clears both.")

    @staticmethod
    def _wrap_lines(lines: List[str], width: int = 58, max_lines: int = 16) -> str:
        wrapped: List[str] = []
        for line in lines:
            parts = textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if not parts:
                wrapped.append("")
            else:
                wrapped.extend(parts)
            if len(wrapped) >= max_lines:
                wrapped = wrapped[: max_lines - 1] + ["..."]
                break
        return "\n".join(wrapped)

    def _compute(self):
        if self.start is None or self.goal is None:
            self.last_threshold = None
            self.last_details = None
            return

        step_total = int(
            abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1])
        )
        result = compute_dynamic_ins_threshold(
            start=self.start,
            goal=self.goal,
            step_total=step_total,
            beacon_layout=self.beacons,
            dynamic_enabled=self.dynamic_enabled,
            default_threshold=self.default_threshold,
            corridor_width=self.corridor_width,
            safety_factor=self.safety_factor,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold,
            return_details=True,
        )
        if isinstance(result, tuple):
            threshold, details = result
        else:
            threshold = int(result)
            details = {}
        self.last_threshold = int(threshold)
        self.last_details = details

    def _draw_corridor(self):
        if self.start is None or self.goal is None:
            return
        sx, sy = float(self.start[0]), float(self.start[1])
        gx, gy = float(self.goal[0]), float(self.goal[1])
        vx = gx - sx
        vy = gy - sy
        seg_len = float(np.hypot(vx, vy))
        if seg_len <= 1e-6:
            return

        nx = -vy / seg_len
        ny = vx / seg_len
        w = max(0.0, float(self.corridor_width))
        pts = np.array(
            [
                [sx + nx * w, sy + ny * w],
                [gx + nx * w, gy + ny * w],
                [gx - nx * w, gy - ny * w],
                [sx - nx * w, sy - ny * w],
            ]
        )
        patch = Polygon(
            pts,
            closed=True,
            facecolor="#00bcd4",
            edgecolor="#0097a7",
            alpha=0.16,
            linewidth=1.0,
        )
        self.ax.add_patch(patch)

    def _draw_beacons(self):
        candidate_set = set()
        if self.last_details is not None:
            candidate_set = set(self.last_details.get("candidate_beacon_indices", []))

        for idx, b in enumerate(self.beacons):
            color = "#ffeb3b" if idx in candidate_set else "#ffffff"
            edge = "#f57f17" if idx in candidate_set else "#111111"
            self.ax.scatter(
                b["x"],
                b["y"],
                c=color,
                s=42,
                edgecolors=edge,
                linewidths=0.9,
                zorder=3,
            )
            self.ax.add_patch(
                Circle(
                    (b["x"], b["y"]),
                    b["radius"],
                    fill=False,
                    color=edge,
                    linewidth=1.1,
                    alpha=0.85,
                )
            )

    def _draw_start_goal(self):
        if self.start is not None:
            self.ax.scatter(
                self.start[0],
                self.start[1],
                marker="s",
                c="#00e676",
                s=90,
                edgecolors="#1b5e20",
                linewidths=1.2,
                zorder=5,
            )
        if self.goal is not None:
            self.ax.scatter(
                self.goal[0],
                self.goal[1],
                marker="*",
                c="#ff5252",
                s=130,
                edgecolors="#b71c1c",
                linewidths=1.0,
                zorder=5,
            )
        if self.start is not None and self.goal is not None:
            self.ax.plot(
                [self.start[0], self.goal[0]],
                [self.start[1], self.goal[1]],
                color="#f50057",
                linewidth=1.4,
                alpha=0.9,
                zorder=4,
            )

    def _build_status_text(self, message: str = "") -> str:
        lines = [f"start={self.start}", f"goal={self.goal}"]
        lines.append(
            "params: "
            + f"default={self.default_threshold}, dynamic={self.dynamic_enabled}, "
            + f"corridor={self.corridor_width}, safety={self.safety_factor}, "
            + f"clip=[{self.min_threshold},{self.max_threshold}]"
        )

        if self.last_details is None:
            lines.append("threshold=NA")
        else:
            lines.append(
                f"step_total={self.last_details['step_total']} | last_threshold={self.last_threshold} | raw={self.last_details['raw_threshold']}"
            )
            lines.append(
                f"candidates={len(self.last_details.get('candidate_beacon_indices', []))} | max_gap={self.last_details.get('max_gap', 0)}"
            )
            for gap in self.last_details.get("gaps", [])[:6]:
                frm = gap["from"]
                to = gap["to"]
                lines.append(
                    f"gap={gap['gap_steps']:>3d} from {frm['kind']}@{frm['t']:.2f} -> {to['kind']}@{to['t']:.2f}"
                )

        if message:
            lines.append(message)
        return "\n".join(lines)

    def redraw(self, message: str = ""):
        self.ax.clear()
        self.ax.imshow(self.terrain, cmap="viridis", origin="lower")
        self._draw_corridor()
        self._draw_beacons()
        self._draw_start_goal()

        self.ax.set_xlim(-0.5, self.rows - 0.5)
        self.ax.set_ylim(-0.5, self.cols - 0.5)
        self.ax.set_title(
            "INS Threshold Visual Tester | Left: pick start/goal | Right: clear | C: clear all"
        )
        self.ax.set_xlabel(
            "Left click: first point=start, second point=goal, third click starts a new pair.\n"
            "Right click or key C: clear both points and reset."
        )

        status_text = self._build_status_text(message)
        status_text = self._wrap_lines(status_text.split("\n"), width=58, max_lines=16)
        self.fig.subplots_adjust(right=0.74, bottom=0.18)
        if self.status_artist is None:
            self.status_artist = self.fig.text(
                0.76,
                0.96,
                "",
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#666666"},
            )
        self.status_artist.set_text(status_text)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return

        if event.button == 3:
            self.start = None
            self.goal = None
            self._compute()
            self.redraw("Cleared start and goal.")
            return

        if event.button != 1:
            return

        if self.start is None or (self.start is not None and self.goal is not None):
            self.start = (x, y)
            self.goal = None
            self._compute()
            self.redraw("Start point set.")
            return

        self.goal = (x, y)
        self._compute()
        self.redraw("Goal point set. Threshold computed.")

    def on_key(self, event):
        if str(event.key).lower() == "c":
            self.start = None
            self.goal = None
            self._compute()
            self.redraw("Reset complete.")


def main():
    tester_cfg = _load_tester_cfg()

    base_cfg_path = _resolve_project_path(
        str(tester_cfg.tester.get("base_config_path", "configs/config1.yaml"))
    )
    base_cfg = _load_cfg(base_cfg_path)
    base_cfg = _resolve_beacon_override(base_cfg, tester_cfg)

    rows = int(tester_cfg.tester.get("rows", base_cfg.env.rows))
    cols = int(tester_cfg.tester.get("cols", base_cfg.env.cols))
    terrain_run_dir = str(tester_cfg.tester.get("terrain_run_dir", "")).strip()
    terrain = _load_terrain(
        rows, cols, _resolve_project_path(terrain_run_dir) if terrain_run_dir else None
    )

    beacons, _ = load_beacon_layout(base_cfg, rows, cols)

    dynamic_enabled = bool(
        tester_cfg.tester.get(
            "dynamic_enabled", base_cfg.env.get("dynamic_ins_error_threshold", True)
        )
    )
    default_threshold = int(
        tester_cfg.tester.get(
            "default_threshold", base_cfg.env.get("ins_error_threshold", 10)
        )
    )
    corridor_width = float(
        tester_cfg.tester.get(
            "corridor_width", base_cfg.env.get("dynamic_ins_corridor_width", 4.0)
        )
    )
    safety_factor = float(
        tester_cfg.tester.get(
            "safety_factor", base_cfg.env.get("dynamic_ins_safety_factor", 0.9)
        )
    )
    min_threshold = int(
        tester_cfg.tester.get(
            "min_threshold", base_cfg.env.get("dynamic_ins_min", default_threshold)
        )
    )
    max_threshold = int(
        tester_cfg.tester.get(
            "max_threshold", base_cfg.env.get("dynamic_ins_max", default_threshold)
        )
    )

    app = InsThresholdVisualTester(
        rows=rows,
        cols=cols,
        terrain=terrain,
        beacons=beacons,
        default_threshold=default_threshold,
        dynamic_enabled=dynamic_enabled,
        corridor_width=corridor_width,
        safety_factor=safety_factor,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    plt.show()


if __name__ == "__main__":
    main()
