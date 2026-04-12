import json
import os
from typing import Dict, List, Optional, Tuple


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def resolve_beacon_config_path(path_value: str) -> Optional[str]:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value

    # Hydra may switch cwd to outputs; original cwd points to repo root.
    try:
        from hydra.utils import get_original_cwd

        base_dir = get_original_cwd()
    except Exception:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(base_dir, path_value))


def _normalize_from_json(
    data: Dict,
    rows: int,
    cols: int,
    default_radius: int,
) -> List[Dict[str, int]]:
    result: List[Dict[str, int]] = []
    for item in data.get("beacons", []):
        x = _safe_int(item.get("x", 0), 0)
        y = _safe_int(item.get("y", 0), 0)
        radius = _safe_int(item.get("radius", default_radius), default_radius)
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue
        radius = max(1, radius)
        result.append({"x": x, "y": y, "radius": radius})
    return result


def _normalize_from_cfg(
    cfg,
    rows: int,
    cols: int,
    default_radius: int,
) -> List[Dict[str, int]]:
    result: List[Dict[str, int]] = []
    raw_locations = cfg.beacon.get("beacon_locations", [])
    for item in raw_locations:
        x = None
        y = None
        radius = default_radius

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x = _safe_int(item[0], 0)
            y = _safe_int(item[1], 0)
            if len(item) >= 3:
                radius = _safe_int(item[2], default_radius)
        elif isinstance(item, dict):
            x = _safe_int(item.get("x", 0), 0)
            y = _safe_int(item.get("y", 0), 0)
            radius = _safe_int(item.get("radius", default_radius), default_radius)

        if x is None or y is None:
            continue
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue
        radius = max(1, radius)
        result.append({"x": x, "y": y, "radius": radius})
    return result


def load_beacon_layout(
    cfg,
    rows: int,
    cols: int,
) -> Tuple[List[Dict[str, int]], Optional[str]]:
    default_radius = _safe_int(cfg.beacon.get("beacon_signal_area", 5), 5)
    beacon_path = str(cfg.beacon.get("beacon_config_path", "")).strip()
    resolved_path = resolve_beacon_config_path(beacon_path)

    if resolved_path and os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _normalize_from_json(data, rows, cols, default_radius), resolved_path

    # Fallback for backward compatibility: read from YAML beacon_locations
    return _normalize_from_cfg(cfg, rows, cols, default_radius), resolved_path
