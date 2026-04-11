import csv
import os

import numpy as np


ENVIRONMENT_GRID_COLUMNS = [
    "x",
    "y",
    "seafloor",
    "beacon_strength",
    "beacon_id",
    "u_flow",
    "v_flow",
]

GRID_LAYER_SPECS = {
    "seafloor": {"dtype": float, "default": 0.0},
    "beacon_strength": {"dtype": float, "default": 0.0},
    "beacon_id": {"dtype": float, "default": 0.0},
    "u_flow": {"dtype": float, "default": 0.0},
    "v_flow": {"dtype": float, "default": 0.0},
}


def environment_grid_path(run_dir: str) -> str:
    return os.path.join(run_dir, "environment_grid.csv")


def _normalize_array(name, value):
    if value is None:
        return None
    dtype = GRID_LAYER_SPECS[name]["dtype"]
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，当前维度: {array.ndim}")
    return array


def _resolve_shape(layers):
    shape = None
    for value in layers.values():
        if value is None:
            continue
        if shape is None:
            shape = value.shape
            continue
        if value.shape != shape:
            raise ValueError(f"环境层数组尺寸不一致: {shape} vs {value.shape}")
    return shape


def load_environment_grid(run_dir: str):
    csv_path = environment_grid_path(run_dir)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} 不存在")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} 中没有数据")

    max_x = max(int(float(row["x"])) for row in rows)
    max_y = max(int(float(row["y"])) for row in rows)
    shape = (max_x + 1, max_y + 1)
    layers = {
        name: np.full(shape, spec["default"], dtype=spec["dtype"])
        for name, spec in GRID_LAYER_SPECS.items()
    }

    for row in rows:
        x = int(float(row["x"]))
        y = int(float(row["y"]))
        for name, spec in GRID_LAYER_SPECS.items():
            raw = row.get(name, "")
            if raw in ("", None):
                continue
            layers[name][x, y] = spec["dtype"](float(raw))

    return layers


def update_environment_grid(
    run_dir: str,
    *,
    seafloor=None,
    beacon_strength=None,
    beacon_id=None,
    u_flow=None,
    v_flow=None,
):
    updates = {
        "seafloor": _normalize_array("seafloor", seafloor),
        "beacon_strength": _normalize_array("beacon_strength", beacon_strength),
        "beacon_id": _normalize_array("beacon_id", beacon_id),
        "u_flow": _normalize_array("u_flow", u_flow),
        "v_flow": _normalize_array("v_flow", v_flow),
    }
    update_shape = _resolve_shape(updates)
    existing = None
    if os.path.exists(environment_grid_path(run_dir)):
        existing = load_environment_grid(run_dir)

    existing_shape = _resolve_shape(existing or {})
    shape = update_shape or existing_shape
    if shape is None:
        raise ValueError("没有可写入的环境层数据")
    if (
        update_shape is not None
        and existing_shape is not None
        and update_shape != existing_shape
    ):
        raise ValueError(
            f"environment_grid.csv 现有尺寸 {existing_shape} 与更新尺寸 {update_shape} 不一致"
        )

    merged = {}
    for name, spec in GRID_LAYER_SPECS.items():
        if updates[name] is not None:
            merged[name] = updates[name]
        elif existing is not None:
            merged[name] = existing[name]
        else:
            merged[name] = np.full(shape, spec["default"], dtype=spec["dtype"])

    csv_path = environment_grid_path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENVIRONMENT_GRID_COLUMNS)
        writer.writeheader()
        for x in range(shape[0]):
            for y in range(shape[1]):
                writer.writerow(
                    {
                        "x": x,
                        "y": y,
                        "seafloor": f"{float(merged['seafloor'][x, y]):.10g}",
                        "beacon_strength": f"{float(merged['beacon_strength'][x, y]):.10g}",
                        "beacon_id": int(round(float(merged["beacon_id"][x, y]))),
                        "u_flow": f"{float(merged['u_flow'][x, y]):.10g}",
                        "v_flow": f"{float(merged['v_flow'][x, y]):.10g}",
                    }
                )

    return csv_path
