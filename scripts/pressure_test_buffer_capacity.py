import math
import os
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from buffers.prioritized_replay_gpu import PrioritizedNStepReplayBufferGPU


TEST_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "buffer_pressure_test.yaml")


def _resolve_project_path(path_value: str, fallback: Optional[str] = None) -> str:
    if path_value:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_value))
    if fallback:
        return os.path.normpath(os.path.join(PROJECT_ROOT, fallback))
    return PROJECT_ROOT


def _load_cfg(path: str):
    return OmegaConf.load(path)


def _load_test_cfg():
    if not os.path.exists(TEST_CONFIG_PATH):
        raise FileNotFoundError(f"Pressure test config not found: {TEST_CONFIG_PATH}")
    return _load_cfg(TEST_CONFIG_PATH)


def _resolve_store_mode(base_cfg, test_cfg) -> str:
    mode = str(test_cfg.pressure_test.get("test_mode", "from_config")).strip().lower()
    if mode == "force_meta_only":
        return "meta_only"
    if mode == "force_full_patch":
        return "full_patch"
    return str(base_cfg.dqn.get("replay_store_mode", "meta_only")).strip().lower()


def _build_observation_cache(rows: int, cols: int):
    shape = (rows, cols)
    terrain = np.random.rand(*shape).astype(np.float32)
    u_norm = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    v_norm = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    beacon = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    depth_reward = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return {
        "terrain": terrain,
        "u_norm": u_norm,
        "v_norm": v_norm,
        "beacon": beacon,
        "depth_reward": depth_reward,
        # Keep this None in pressure test to avoid unrelated precompute overhead.
        "goal_distance_library": None,
    }


def _gpu_stats(device: torch.device) -> Dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = float(torch.cuda.memory_allocated(device))
    reserved = float(torch.cuda.memory_reserved(device))
    used = float(total_bytes - free_bytes)
    return {
        "total_gb": total_bytes / (1024**3),
        "free_gb": free_bytes / (1024**3),
        "used_gb": used / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "used_ratio": used / max(1.0, float(total_bytes)),
        "allocated_ratio": allocated / max(1.0, float(total_bytes)),
        "reserved_ratio": reserved / max(1.0, float(total_bytes)),
    }


def _make_state(device: torch.device, action_dim: int, mode: str, dqn_cfg):
    patch_small = int(dqn_cfg.get("patch_small", 11))
    patch_large = int(dqn_cfg.get("patch_large", 21))
    replay_dtype_name = str(dqn_cfg.get("replay_dtype", "float16")).lower()
    replay_dtype = (
        torch.float16
        if replay_dtype_name in {"float16", "fp16", "half"}
        else torch.float32
    )

    scalar = torch.rand((12,), dtype=replay_dtype, device=device)
    action_feat = torch.rand((action_dim * 5,), dtype=replay_dtype, device=device)

    if mode == "meta_only":
        x = random.randint(0, 49)
        y = random.randint(0, 49)
        gx = random.randint(0, 49)
        gy = random.randint(0, 49)
        step_total = abs(gx - x) + abs(gy - y)
        patch_meta = torch.tensor(
            [x, y, gx, gy, step_total], dtype=torch.int32, device=device
        )
        return {
            "scalar": scalar,
            "action_feat": action_feat,
            "patch_meta": patch_meta,
        }

    channels = 5
    patch_s = torch.rand(
        (channels, patch_small, patch_small), dtype=replay_dtype, device=device
    )
    patch_l = torch.rand(
        (channels, patch_large, patch_large), dtype=replay_dtype, device=device
    )
    return {
        "scalar": scalar,
        "action_feat": action_feat,
        "patch_s": patch_s,
        "patch_l": patch_l,
    }


def _fill_buffer_until_limit(
    buffer, mode: str, base_cfg, test_cfg, device: torch.device
):
    action_dim = int(base_cfg.env.get("action_dim", 4))
    stop_ratio = float(test_cfg.pressure_test.get("stop_gpu_usage_ratio", 0.90))
    write_batch_size = int(test_cfg.pressure_test.get("write_batch_size", 1000))
    max_total_writes = int(test_cfg.pressure_test.get("max_total_writes", 5_000_000))
    report_every = max(
        1, int(test_cfg.pressure_test.get("report_interval_batches", 10))
    )

    writes = 0
    batches = 0
    reached_limit = False
    oom_message = None

    while writes < max_total_writes and len(buffer) < buffer.capacity:
        try:
            for _ in range(write_batch_size):
                if writes >= max_total_writes or len(buffer) >= buffer.capacity:
                    break
                state = _make_state(device, action_dim, mode, base_cfg.dqn)
                next_state = _make_state(device, action_dim, mode, base_cfg.dqn)
                action = random.randint(0, action_dim - 1)
                reward = random.uniform(-5.0, 5.0)
                done = True  # one add -> one finalized transition
                buffer.add(state, action, reward, next_state, done)
                writes += 1
            batches += 1
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                oom_message = str(err)
                break
            raise

        stats = _gpu_stats(device)
        if batches % report_every == 0:
            print(
                f"[progress] cap={buffer.capacity:,} len={len(buffer):,} writes={writes:,} "
                f"used={stats['used_gb']:.2f}GB({stats['used_ratio'] * 100:.1f}%) "
                f"reserved={stats['reserved_gb']:.2f}GB({stats['reserved_ratio'] * 100:.1f}%)"
            )

        if stats["used_ratio"] >= stop_ratio:
            reached_limit = True
            break

    final_stats = _gpu_stats(device)
    return {
        "writes": int(writes),
        "length": int(len(buffer)),
        "reached_limit": bool(reached_limit),
        "oom_message": oom_message,
        "stats": final_stats,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pressure test.")

    test_cfg = _load_test_cfg()
    base_cfg_path = _resolve_project_path(
        str(test_cfg.pressure_test.get("base_config_path", "configs/config1.yaml"))
    )
    base_cfg = _load_cfg(base_cfg_path)

    seed = int(test_cfg.pressure_test.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = str(base_cfg.dqn.get("device", "cuda"))
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("Pressure test must run on CUDA device.")

    mode = _resolve_store_mode(base_cfg, test_cfg)
    if mode not in {"meta_only", "full_patch"}:
        raise ValueError(f"Unsupported replay mode for pressure test: {mode}")

    start_capacity = int(test_cfg.pressure_test.get("start_capacity", 50_000))
    step_capacity = int(test_cfg.pressure_test.get("capacity_step", 50_000))
    max_capacity = int(test_cfg.pressure_test.get("max_capacity", 2_000_000))
    stop_ratio = float(test_cfg.pressure_test.get("stop_gpu_usage_ratio", 0.90))

    rows = int(base_cfg.env.get("rows", 50))
    cols = int(base_cfg.env.get("cols", 50))
    observation_cache = _build_observation_cache(rows, cols)

    print("=== Buffer Pressure Test (GPU VRAM) ===")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"mode={mode}, replay_dtype={base_cfg.dqn.get('replay_dtype', 'float16')}")
    print(
        f"capacity sweep: start={start_capacity:,}, step={step_capacity:,}, max={max_capacity:,}"
    )
    print(f"stop threshold: used_ratio >= {stop_ratio:.2f}")

    start_time = time.time()
    records = []
    recommended_capacity = None

    capacity = start_capacity
    while capacity <= max_capacity:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        buffer = PrioritizedNStepReplayBufferGPU(
            capacity=capacity,
            alpha=float(base_cfg.dqn.get("per_alpha", 0.6)),
            n_step=int(base_cfg.dqn.get("n_step", 3)),
            gamma=float(base_cfg.agent.get("gamma", 0.99)),
            device=str(device),
            observation_cache=observation_cache,
            patch_small=int(base_cfg.dqn.get("patch_small", 11)),
            patch_large=int(base_cfg.dqn.get("patch_large", 21)),
            reconstruct_patch_on_sample=bool(
                base_cfg.dqn.get("reconstruct_patch_on_sample", False)
            ),
            cache_maps_on_gpu=bool(base_cfg.dqn.get("cache_maps_on_gpu", True)),
            goal_prior_cache_size=int(base_cfg.dqn.get("goal_prior_cache_size", 64)),
            replay_store_mode=mode,
            replay_dtype=str(base_cfg.dqn.get("replay_dtype", "float16")),
            use_batch_patch_indexing=bool(
                base_cfg.dqn.get("use_batch_patch_indexing", True)
            ),
        )

        print(f"\n[capacity-test] target_capacity={capacity:,}")
        result = _fill_buffer_until_limit(buffer, mode, base_cfg, test_cfg, device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)

        record = {
            "capacity": capacity,
            "length": result["length"],
            "writes": result["writes"],
            "reached_limit": result["reached_limit"],
            "oom": bool(result["oom_message"]),
            "used_ratio": result["stats"]["used_ratio"],
            "used_gb": result["stats"]["used_gb"],
            "reserved_gb": result["stats"]["reserved_gb"],
            "allocated_gb": result["stats"]["allocated_gb"],
            "free_gb": result["stats"]["free_gb"],
            "peak_allocated_gb": peak_allocated,
            "oom_message": result["oom_message"],
        }
        records.append(record)

        print(
            "[result] "
            + f"len={record['length']:,}, writes={record['writes']:,}, "
            + f"used={record['used_gb']:.2f}GB({record['used_ratio'] * 100:.1f}%), "
            + f"allocated={record['allocated_gb']:.2f}GB, reserved={record['reserved_gb']:.2f}GB, "
            + f"peak_allocated={record['peak_allocated_gb']:.2f}GB"
        )

        if record["reached_limit"] or record["oom"]:
            # Keep safety margin to leave room for model/optimizer tensors in real training.
            recommended_capacity = max(start_capacity, int(capacity * 0.85))
            break

        capacity += step_capacity

    elapsed = time.time() - start_time
    print("\n=== Analysis Report ===")
    print(f"elapsed={elapsed:.1f}s, tested_cases={len(records)}")

    for idx, r in enumerate(records, start=1):
        reason = "ok"
        if r["reached_limit"]:
            reason = "stop@90%"
        if r["oom"]:
            reason = "oom"
        print(
            f"{idx}. cap={r['capacity']:,}, stored={r['length']:,}, used={r['used_gb']:.2f}GB "
            f"({r['used_ratio'] * 100:.1f}%), peak_alloc={r['peak_allocated_gb']:.2f}GB, reason={reason}"
        )

    if recommended_capacity is None:
        recommended_capacity = records[-1]["capacity"] if records else start_capacity

    print("\nRecommendation:")
    print(f"- Suggested safe buffer.capacity: {recommended_capacity:,}")
    print("- Keep 10%-15% VRAM headroom for model forward/backward and batch sampling.")

    if records and records[-1]["oom_message"]:
        print("\nOOM snippet:")
        print(records[-1]["oom_message"])


if __name__ == "__main__":
    main()
