"""Thin wrapper around the current camera-rig metric script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import source_camera_rig_metrics as source
from .config import PipelineConfig
from . import sensor_noise


_ORIGINAL_LOAD_CAMERAS = source.load_cameras


def _sample_indices_arg(config: PipelineConfig) -> str | None:
    if config.dataset.sample_indices is None:
        return None
    return ",".join(str(index) for index in config.dataset.sample_indices)


def run(config: PipelineConfig, run_dir: Path, audit_only: bool = False) -> dict[str, Any]:
    cfg = config.metrics_camera_rig
    results_dir = run_dir / "camera_rig_metrics"
    args = argparse.Namespace(
        capture_root=str(config.dataset.capture_root),
        results_dir=str(results_dir),
        audit_only=bool(audit_only),
        sample_indices=_sample_indices_arg(config),
        skip_figures=bool(cfg.skip_figures),
        depth_stride=int(cfg.depth_stride),
        metric_scope=cfg.metric_scope,
        distance_thresholds_cm=",".join(f"{value:g}" for value in cfg.distance_thresholds_cm),
        headline_threshold_cm=float(cfg.headline_threshold_cm),
        plot_style=cfg.plot_style,
    )
    if sensor_noise.is_noise_enabled(config):
        source.load_cameras = _noisy_load_cameras(config)
    else:
        source.load_cameras = _ORIGINAL_LOAD_CAMERAS
    try:
        source.run(args)
    finally:
        source.load_cameras = _ORIGINAL_LOAD_CAMERAS
    return {
        "results_dir": str(results_dir),
        "noise_profile": config.noise.profile,
        "audit": str(results_dir / "SUMMARY_integrity_audit.md"),
        "headline_csv": str(results_dir / "SUMMARY_camera_rig_visibility.csv"),
        "layout_csv": str(results_dir / "SUMMARY_layout_robustness.csv"),
        "free_space_csv": str(results_dir / "SUMMARY_free_space_certainty.csv"),
    }


def _noisy_load_cameras(config: PipelineConfig):
    def load_cameras_with_noise(sample):
        metadata, cameras = _ORIGINAL_LOAD_CAMERAS(sample)
        for camera in cameras:
            result = sensor_noise.apply_depth_noise(
                camera.depth,
                camera.valid_depth,
                config=config,
                sample_key=sample.relative_path,
                camera_index=camera.index,
                depth_path=camera.depth_path,
            )
            camera.depth = result.depth
            camera.valid_depth = source.valid_depth_mask(camera.depth, camera.near_clip, camera.far_clip)
        return metadata, cameras

    return load_cameras_with_noise
