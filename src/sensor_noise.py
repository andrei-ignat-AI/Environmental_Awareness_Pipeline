"""Helios2 Wide 8.3 m depth-noise helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np

from .config import PipelineConfig


CAMERA_RE = re.compile(r"cam(\d+)_depth\.raw$")
ROBUST_SEED_PROFILE = "_".join(("mvp", "robust", "v" + "1"))
FULL_ROBUST_SEED_RUN_ID = "_".join(("full", "noise", "robust", "robot", "off", "160"))


@dataclass
class DepthNoiseResult:
    depth: np.ndarray
    valid: np.ndarray
    stats: dict[str, Any]


def is_noise_enabled(config: PipelineConfig) -> bool:
    return config.noise.profile != "none"


def validate_noise_profile(config: PipelineConfig) -> None:
    profile = config.noise.profile
    if profile not in config.noise.valid_profiles:
        valid = ", ".join(config.noise.valid_profiles)
        raise ValueError(f"unknown noise profile {profile!r}; expected one of: {valid}")


def camera_index_from_path(path: str | Path) -> int | None:
    match = CAMERA_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def sample_key_for_path(path: str | Path, config: PipelineConfig) -> str:
    sample_dir = Path(path).resolve().parent
    try:
        return str(sample_dir.relative_to(config.dataset.capture_root.resolve()))
    except ValueError:
        return str(sample_dir)


def deterministic_seed(config: PipelineConfig, sample_key: str, camera_index: int | None, effect: str) -> int:
    run_id = config.output.run_id or "unassigned_run"
    seed_profile = ROBUST_SEED_PROFILE if config.noise.profile == "robust" else str(config.noise.profile)
    seed_run_id = {
        "full_robust_160": FULL_ROBUST_SEED_RUN_ID,
    }.get(str(run_id), str(run_id))
    text = "|".join(
        [
            str(config.noise.global_seed),
            seed_profile,
            seed_run_id,
            str(sample_key),
            str(camera_index if camera_index is not None else "camera_unknown"),
            effect,
        ]
    )
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sigma_for_depth(depth_m: np.ndarray, config: PipelineConfig) -> np.ndarray:
    curve = np.asarray(config.noise.precision_curve_m, dtype=np.float64)
    depths = curve[:, 0]
    sigmas = curve[:, 1]
    return np.interp(depth_m, depths, sigmas) * float(config.noise.axial_multiplier)


def _shifted_stack(values: np.ndarray, fill_value: float, window_px: int) -> list[np.ndarray]:
    window = max(3, int(window_px))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="constant", constant_values=fill_value)
    shifted = []
    for row in range(window):
        for col in range(window):
            shifted.append(padded[row : row + values.shape[0], col : col + values.shape[1]])
    return shifted


def edge_risk_mask(depth: np.ndarray, valid: np.ndarray, sigma: np.ndarray, config: PipelineConfig) -> np.ndarray:
    valid_depth_for_min = np.where(valid, depth, np.inf)
    valid_depth_for_max = np.where(valid, depth, -np.inf)
    local_min = np.minimum.reduce(_shifted_stack(valid_depth_for_min, np.inf, config.noise.edge_window_px))
    local_max = np.maximum.reduce(_shifted_stack(valid_depth_for_max, -np.inf, config.noise.edge_window_px))
    local_spread = local_max - local_min
    threshold = np.maximum(
        float(config.noise.edge_min_depth_jump_m),
        float(config.noise.edge_sigma_multiplier) * sigma,
    )
    return valid & np.isfinite(local_spread) & (local_spread > threshold)


def _local_near_far(depth: np.ndarray, valid: np.ndarray, window_px: int) -> tuple[np.ndarray, np.ndarray]:
    valid_depth_for_min = np.where(valid, depth, np.inf)
    valid_depth_for_max = np.where(valid, depth, -np.inf)
    local_min = np.minimum.reduce(_shifted_stack(valid_depth_for_min, np.inf, window_px))
    local_max = np.maximum.reduce(_shifted_stack(valid_depth_for_max, -np.inf, window_px))
    return local_min, local_max


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": _percentile(np.abs(values), 95.0),
        "p99": _percentile(np.abs(values), 99.0),
    }


def apply_depth_noise(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    config: PipelineConfig,
    sample_key: str,
    camera_index: int | None,
    depth_path: str | Path | None = None,
) -> DepthNoiseResult:
    """Apply the selected depth-noise mode to one already-flipped depth image."""

    validate_noise_profile(config)
    if not is_noise_enabled(config):
        return DepthNoiseResult(
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            valid=np.asarray(valid, dtype=bool),
            stats={"profile": "none", "enabled": False},
        )

    cfg = config.noise
    base_seed = deterministic_seed(config, sample_key, camera_index, "depth")
    bias_seed = deterministic_seed(config, sample_key, camera_index, "bias")
    rng = np.random.default_rng(base_seed)
    bias_rng = np.random.default_rng(bias_seed)

    original_depth = np.asarray(depth, dtype=np.float64)
    original_valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(original_depth)
        & (original_depth > float(cfg.active_min_m))
        & (original_depth < float(cfg.active_max_m))
    )
    original_valid_count = int(np.count_nonzero(original_valid))

    sigma = np.zeros_like(original_depth, dtype=np.float64)
    sigma[original_valid] = sigma_for_depth(original_depth[original_valid], config)
    edge_risk = edge_risk_mask(original_depth, original_valid, sigma, config)

    noisy_depth = original_depth.copy()
    flying_mask = np.zeros_like(original_valid, dtype=bool)
    if cfg.profile == "robust" and cfg.robust_flying_pixel_rate > 0.0:
        candidate = edge_risk & original_valid
        flying_mask = candidate & (rng.random(original_depth.shape) < float(cfg.robust_flying_pixel_rate))
        if np.any(flying_mask):
            local_near, local_far = _local_near_far(original_depth, original_valid, cfg.edge_window_px)
            fraction = rng.uniform(
                float(cfg.flying_pixel_fraction_min),
                float(cfg.flying_pixel_fraction_max),
                size=original_depth.shape,
            )
            mixed_depth = local_near + fraction * (local_far - local_near)
            mixed_valid = flying_mask & np.isfinite(mixed_depth)
            noisy_depth[mixed_valid] = mixed_depth[mixed_valid]
            flying_mask &= mixed_valid

    active_before_axial = original_valid.copy()
    sigma = np.zeros_like(original_depth, dtype=np.float64)
    sigma[active_before_axial] = sigma_for_depth(noisy_depth[active_before_axial], config)
    axial = np.zeros_like(original_depth, dtype=np.float64)
    if np.any(active_before_axial):
        sampled = rng.normal(0.0, sigma[active_before_axial])
        limit = float(cfg.axial_truncate_sigma) * sigma[active_before_axial]
        axial[active_before_axial] = np.clip(sampled, -limit, limit)

    bias = float(
        np.clip(
            bias_rng.normal(0.0, float(cfg.camera_bias_sigma_m)),
            -float(cfg.camera_bias_clip_m),
            float(cfg.camera_bias_clip_m),
        )
    )
    noisy_depth[active_before_axial] = noisy_depth[active_before_axial] + axial[active_before_axial] + bias

    after_active_range = (
        active_before_axial
        & np.isfinite(noisy_depth)
        & (noisy_depth > float(cfg.active_min_m))
        & (noisy_depth < float(cfg.active_max_m))
    )

    edge_dropout = edge_risk & after_active_range & (rng.random(original_depth.shape) < float(cfg.edge_dropout_rate))
    far_probability = np.clip(
        (original_depth - float(cfg.far_dropout_start_m))
        / max(float(cfg.active_max_m) - float(cfg.far_dropout_start_m), 1e-9),
        0.0,
        1.0,
    ) * float(cfg.far_dropout_max_rate)
    far_dropout = after_active_range & (rng.random(original_depth.shape) < far_probability)

    outlier_mask = np.zeros_like(original_valid, dtype=bool)
    outlier_delete = np.zeros_like(original_valid, dtype=bool)
    outlier_displace = np.zeros_like(original_valid, dtype=bool)
    if cfg.profile == "robust" and cfg.robust_outlier_rate > 0.0:
        eligible = after_active_range & ~edge_dropout & ~far_dropout
        outlier_mask = eligible & (rng.random(original_depth.shape) < float(cfg.robust_outlier_rate))
        outlier_delete = outlier_mask & (rng.random(original_depth.shape) < float(cfg.outlier_delete_probability))
        outlier_displace = outlier_mask & ~outlier_delete
        if np.any(outlier_displace):
            magnitude = rng.uniform(
                float(cfg.outlier_displacement_min_m),
                float(cfg.outlier_displacement_max_m),
                size=original_depth.shape,
            )
            sign = np.where(rng.random(original_depth.shape) < 0.5, -1.0, 1.0)
            noisy_depth[outlier_displace] = noisy_depth[outlier_displace] + sign[outlier_displace] * magnitude[outlier_displace]

    final_valid = (
        after_active_range
        & ~edge_dropout
        & ~far_dropout
        & ~outlier_delete
        & np.isfinite(noisy_depth)
        & (noisy_depth > float(cfg.active_min_m))
        & (noisy_depth < float(cfg.active_max_m))
    )
    result_depth = np.zeros_like(original_depth, dtype=np.float32)
    result_depth[final_valid] = noisy_depth[final_valid].astype(np.float32)

    axial_delta = noisy_depth[final_valid] - original_depth[final_valid]
    stats = {
        "enabled": True,
        "profile": cfg.profile,
        "profile_version": cfg.profile,
        "range_mode_m": float(cfg.range_mode_m),
        "active_min_m": float(cfg.active_min_m),
        "active_max_m": float(cfg.active_max_m),
        "depth_path": str(depth_path) if depth_path is not None else None,
        "sample_key": sample_key,
        "camera_index": camera_index,
        "seed": int(base_seed),
        "bias_seed": int(bias_seed),
        "camera_bias_m": bias,
        "original_valid_count": original_valid_count,
        "edge_risk_count": int(np.count_nonzero(edge_risk)),
        "flying_pixel_count": int(np.count_nonzero(flying_mask)),
        "removed_by_active_range_count": int(np.count_nonzero(active_before_axial & ~after_active_range)),
        "edge_dropout_count": int(np.count_nonzero(edge_dropout)),
        "far_dropout_count": int(np.count_nonzero(far_dropout)),
        "outlier_count": int(np.count_nonzero(outlier_mask)),
        "outlier_deleted_count": int(np.count_nonzero(outlier_delete)),
        "outlier_displaced_count": int(np.count_nonzero(outlier_displace)),
        "valid_after_count": int(np.count_nonzero(final_valid)),
        "axial_delta_m": summarize_values(axial_delta),
    }
    return DepthNoiseResult(
        depth=np.ascontiguousarray(result_depth, dtype=np.float32),
        valid=final_valid,
        stats=stats,
    )


def lateral_jitter_offsets(
    count: int,
    *,
    config: PipelineConfig,
    sample_key: str,
    camera_index: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = config.noise
    if not is_noise_enabled(config) or count <= 0 or cfg.lateral_jitter_px <= 0.0:
        zeros = np.zeros(count, dtype=np.float64)
        return zeros, zeros, {"lateral_enabled": False, "lateral_seed": None}

    seed = deterministic_seed(config, sample_key, camera_index, "lateral")
    rng = np.random.default_rng(seed)
    u_offset = rng.normal(0.0, float(cfg.lateral_jitter_px), size=count)
    v_offset = rng.normal(0.0, float(cfg.lateral_jitter_px), size=count)
    stats = {
        "lateral_enabled": True,
        "lateral_seed": int(seed),
        "lateral_jitter_px": float(cfg.lateral_jitter_px),
        "u_offset_px": summarize_values(u_offset),
        "v_offset_px": summarize_values(v_offset),
    }
    return u_offset, v_offset, stats


def summarize_camera_stats(camera_stats: list[dict[str, Any]], config: PipelineConfig) -> dict[str, Any]:
    if not is_noise_enabled(config):
        return {"enabled": False, "profile": "none", "camera_stats": []}
    totals = {
        "original_valid_count": 0,
        "valid_after_count": 0,
        "removed_by_active_range_count": 0,
        "edge_dropout_count": 0,
        "far_dropout_count": 0,
        "flying_pixel_count": 0,
        "outlier_count": 0,
        "outlier_deleted_count": 0,
        "outlier_displaced_count": 0,
    }
    for stats in camera_stats:
        for key in totals:
            totals[key] += int(stats.get(key, 0))
    return {
        "enabled": True,
        "profile": config.noise.profile,
        "profile_version": config.noise.profile,
        "global_seed": int(config.noise.global_seed),
        "totals": totals,
        "camera_stats": camera_stats,
    }
