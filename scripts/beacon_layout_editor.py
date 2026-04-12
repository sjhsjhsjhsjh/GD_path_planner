import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.beacon_layout import load_beacon_layout, resolve_beacon_config_path
from envs.environment_grid import environment_grid_path, load_environment_grid


EDITOR_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "beacon_layout_editor.yaml")


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


def _load_editor_cfg():
    if not os.path.exists(EDITOR_CONFIG_PATH):
        raise FileNotFoundError(f"编辑器配置文件不存在: {EDITOR_CONFIG_PATH}")
    return _load_cfg(EDITOR_CONFIG_PATH)


def _resolve_output_path(base_cfg, editor_cfg) -> str:
    output_path = str(editor_cfg.editor.get("output_beacon_config_path", "")).strip()
    if output_path:
        return _resolve_project_path(output_path)

    rel_path = str(base_cfg.beacon.get("beacon_config_path", "")).strip()
    resolved = resolve_beacon_config_path(rel_path)
    if resolved:
        return resolved
    return _resolve_project_path("configs/beacons/default_beacons.json")


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


def _save_beacons(
    path: str, rows: int, cols: int, beacons: List[Dict[str, int]]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": 1,
        "map": {"rows": rows, "cols": cols},
        "distance_metric": "manhattan",
        "beacons": [
            {"x": int(item["x"]), "y": int(item["y"]), "radius": int(item["radius"])}
            for item in beacons
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class BeaconEditor:
    def __init__(
        self,
        rows: int,
        cols: int,
        terrain: np.ndarray,
        beacons: List[Dict[str, int]],
        output_path: str,
        default_radius: int,
        nearby_display_count: int = 5,
        nearby_gap_limit: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.terrain = terrain
        self.beacons = [
            {"x": int(item["x"]), "y": int(item["y"]), "radius": int(item["radius"])}
            for item in beacons
        ]
        self.output_path = output_path
        self.default_radius = max(1, int(default_radius))
        self.nearby_display_count = max(1, int(nearby_display_count))
        self.nearby_gap_limit = None
        if nearby_gap_limit is not None and int(nearby_gap_limit) > 0:
            self.nearby_gap_limit = int(nearby_gap_limit)

        self.selected_idx: Optional[int] = None
        self.history: List[List[Dict[str, int]]] = []
        self.status_artist = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw("Editor ready")

    def _push_history(self):
        snapshot = [dict(item) for item in self.beacons]
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def _find_beacon(self, x: float, y: float) -> Optional[int]:
        if x is None or y is None:
            return None
        best_idx = None
        best_dist = 1e9
        for idx, item in enumerate(self.beacons):
            d = abs(item["x"] - x) + abs(item["y"] - y)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is not None and best_dist <= 1.0:
            return best_idx
        return None

    def _beacon_distance_gap(self, left_idx: int, right_idx: int):
        left = self.beacons[left_idx]
        right = self.beacons[right_idx]
        center_distance = abs(left["x"] - right["x"]) + abs(left["y"] - right["y"])
        edge_gap = max(0, center_distance - left["radius"] - right["radius"])
        return center_distance, edge_gap

    def _build_neighbor_lines(self, anchor_idx: int) -> List[str]:
        lines = []
        anchor = self.beacons[anchor_idx]
        candidates = []
        for idx in range(len(self.beacons)):
            if idx == anchor_idx:
                continue
            center_distance, edge_gap = self._beacon_distance_gap(anchor_idx, idx)
            if self.nearby_gap_limit is not None and edge_gap > self.nearby_gap_limit:
                continue
            candidates.append((edge_gap, center_distance, idx))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for edge_gap, center_distance, idx in candidates[: self.nearby_display_count]:
            other = self.beacons[idx]
            lines.append(
                f"#{idx:02d} ({other['x']},{other['y']}) r={other['radius']} | center={center_distance} | gap={edge_gap}"
            )

        if not lines:
            lines.append("(no nearby beacons under current filter)")
        return lines

    def _build_status_text(self) -> str:
        if self.selected_idx is None or self.selected_idx >= len(self.beacons):
            return "selected=None"

        b = self.beacons[self.selected_idx]
        lines = [f"selected=#{self.selected_idx} ({b['x']},{b['y']}) r={b['radius']}"]
        lines.append(
            f"nearby range: top {self.nearby_display_count}"
            + (
                f", gap <= {self.nearby_gap_limit}"
                if self.nearby_gap_limit is not None
                else ""
            )
        )
        lines.extend(self._build_neighbor_lines(self.selected_idx))
        return "\n".join(lines)

    def redraw(self, message: str = ""):
        self.ax.clear()
        self.ax.imshow(self.terrain, cmap="viridis", origin="lower")
        for idx, item in enumerate(self.beacons):
            color = "#ff3333" if idx == self.selected_idx else "#ffffff"
            self.ax.scatter(
                item["x"],
                item["y"],
                c=color,
                s=45,
                edgecolors="#111111",
                linewidths=0.8,
            )
            circle = Circle(
                (item["x"], item["y"]),
                item["radius"],
                fill=False,
                color=color,
                linewidth=1.3,
                alpha=0.9,
            )
            self.ax.add_patch(circle)
        self.ax.set_xlim(-0.5, self.rows - 0.5)
        self.ax.set_ylim(-0.5, self.cols - 0.5)
        self.ax.set_title(
            "Beacon Layout Editor | Left: add/select, Right: delete | [ ]: radius +/- | Del: remove | u: undo | s: save"
        )
        status_text = self._build_status_text()
        if message:
            status_text = f"{status_text}\n{message}"
        self.ax.set_xlabel(
            "Selected beacon details and nearby distances are shown in the right panel"
        )
        self.fig.subplots_adjust(right=0.72)
        if self.status_artist is None:
            self.status_artist = self.fig.text(
                0.74,
                0.95,
                "",
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666"},
            )
        self.status_artist.set_text("Nearby distances\n" + status_text)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return

        idx = self._find_beacon(x, y)
        if event.button == 1:
            if idx is None:
                self._push_history()
                self.beacons.append({"x": x, "y": y, "radius": self.default_radius})
                self.selected_idx = len(self.beacons) - 1
                self.redraw("added")
            else:
                self.selected_idx = idx
                self.redraw("selected")
        elif event.button == 3:
            if idx is not None:
                self._push_history()
                self.beacons.pop(idx)
                self.selected_idx = None
                self.redraw("deleted")

    def on_key(self, event):
        if event.key == "s":
            _save_beacons(self.output_path, self.rows, self.cols, self.beacons)
            self.redraw(f"saved -> {self.output_path}")
            return

        if event.key == "u":
            if self.history:
                self.beacons = self.history.pop()
                self.selected_idx = None
                self.redraw("undo")
            return

        if event.key == "delete":
            if self.selected_idx is not None and 0 <= self.selected_idx < len(
                self.beacons
            ):
                self._push_history()
                self.beacons.pop(self.selected_idx)
                self.selected_idx = None
                self.redraw("deleted")
            return

        if self.selected_idx is None or self.selected_idx >= len(self.beacons):
            return

        if event.key == "[":
            self._push_history()
            self.beacons[self.selected_idx]["radius"] = max(
                1, self.beacons[self.selected_idx]["radius"] - 1
            )
            self.redraw("radius -1")
        elif event.key == "]":
            self._push_history()
            self.beacons[self.selected_idx]["radius"] = (
                self.beacons[self.selected_idx]["radius"] + 1
            )
            self.redraw("radius +1")


def main():
    editor_cfg = _load_editor_cfg()
    base_cfg_path = _resolve_project_path(
        str(editor_cfg.editor.get("base_config_path", "configs/config1.yaml"))
    )
    base_cfg = _load_cfg(base_cfg_path)

    rows = int(editor_cfg.editor.get("rows", base_cfg.env.rows))
    cols = int(editor_cfg.editor.get("cols", base_cfg.env.cols))
    default_radius = int(
        editor_cfg.editor.get(
            "default_radius", base_cfg.beacon.get("beacon_signal_area", 5)
        )
    )
    output_path = _resolve_output_path(base_cfg, editor_cfg)

    beacons, _ = load_beacon_layout(base_cfg, rows, cols)
    terrain_run_dir = str(editor_cfg.editor.get("terrain_run_dir", "")).strip()
    terrain = _load_terrain(
        rows, cols, _resolve_project_path(terrain_run_dir) if terrain_run_dir else None
    )

    editor = BeaconEditor(
        rows=rows,
        cols=cols,
        terrain=terrain,
        beacons=beacons,
        output_path=output_path,
        default_radius=default_radius,
        nearby_display_count=int(editor_cfg.editor.get("nearby_display_count", 5)),
        nearby_gap_limit=editor_cfg.editor.get("nearby_gap_limit", None),
    )
    plt.show()


if __name__ == "__main__":
    main()
