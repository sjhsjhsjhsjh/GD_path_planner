import math
from typing import Dict, List, Optional, Tuple


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_dynamic_ins_threshold(
    *,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    step_total: int,
    beacon_layout: List[Dict[str, int]],
    dynamic_enabled: bool,
    default_threshold: int,
    corridor_width: float,
    safety_factor: float,
    min_threshold: int,
    max_threshold: int,
    return_details: bool = False,
):
    details = {
        "start": (int(start[0]), int(start[1])),
        "goal": (int(goal[0]), int(goal[1])),
        "step_total": int(step_total),
        "dynamic_enabled": bool(dynamic_enabled),
        "default_threshold": int(default_threshold),
        "corridor_width": float(corridor_width),
        "safety_factor": float(safety_factor),
        "candidate_beacon_indices": [],
        "anchors": [],
        "gaps": [],
        "max_gap": 1,
        "raw_threshold": int(default_threshold),
        "threshold": int(default_threshold),
        "clamp_range": (
            int(min(min_threshold, max_threshold)),
            int(max(min_threshold, max_threshold)),
        ),
    }

    if not dynamic_enabled or int(step_total) <= 1:
        details["reason"] = "dynamic_disabled_or_short_path"
        if return_details:
            return int(default_threshold), details
        return int(default_threshold)

    sx = float(start[0])
    sy = float(start[1])
    gx = float(goal[0])
    gy = float(goal[1])
    vx = gx - sx
    vy = gy - sy
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        details["reason"] = "zero_segment"
        if return_details:
            return int(default_threshold), details
        return int(default_threshold)

    corridor = max(0.0, float(corridor_width))
    details["corridor_width"] = corridor

    anchors = [
        {"t": 0.0, "kind": "start", "beacon_index": None},
        {"t": 1.0, "kind": "goal", "beacon_index": None},
    ]

    for idx, beacon in enumerate(beacon_layout):
        bx = float(beacon["x"])
        by = float(beacon["y"])
        radius = float(beacon.get("radius", 0))

        wx = bx - sx
        wy = by - sy
        t = (wx * vx + wy * vy) / seg_len_sq
        t_clamped = _clamp(t, 0.0, 1.0)
        proj_x = sx + t_clamped * vx
        proj_y = sy + t_clamped * vy
        dist = math.sqrt((bx - proj_x) ** 2 + (by - proj_y) ** 2)

        if dist <= corridor + radius:
            details["candidate_beacon_indices"].append(idx)
            anchors.append(
                {
                    "t": float(t_clamped),
                    "kind": "beacon",
                    "beacon_index": int(idx),
                    "distance_to_line": float(dist),
                }
            )

    anchors = sorted(anchors, key=lambda item: item["t"])
    details["anchors"] = anchors

    max_gap = 1
    gaps = []
    total_steps = max(1, int(step_total))
    for i in range(1, len(anchors)):
        prev_t = float(anchors[i - 1]["t"])
        curr_t = float(anchors[i]["t"])
        gap_steps = int(math.ceil((curr_t - prev_t) * total_steps))
        max_gap = max(max_gap, gap_steps)
        gaps.append(
            {
                "from": anchors[i - 1],
                "to": anchors[i],
                "gap_steps": int(gap_steps),
            }
        )

    details["gaps"] = gaps
    details["max_gap"] = int(max_gap)

    raw_threshold = int(math.ceil(max_gap * max(0.1, float(safety_factor))))
    lower = min(int(min_threshold), int(max_threshold))
    upper = max(int(min_threshold), int(max_threshold))
    threshold = int(max(lower, min(upper, raw_threshold)))

    details["raw_threshold"] = int(raw_threshold)
    details["threshold"] = int(threshold)
    details["reason"] = "ok"

    if return_details:
        return threshold, details
    return threshold