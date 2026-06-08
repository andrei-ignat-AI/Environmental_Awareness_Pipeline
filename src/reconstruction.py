"""Reconstruction-stage wrappers with an optional per-camera noise hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from . import azurion_dataset

from . import source_robert2_decomposition as source
from .config import PipelineConfig
from . import sensor_noise


class IdentityPointNoise:
    """Placeholder for future redesigned point-cloud noise."""

    name = "identity"

    def apply(self, world_points, *, camera_index: int | None = None):
        return world_points


_ORIGINAL_DRAFT_ADD_DEPTH_IMAGE_SAFE = source.draft_add_depth_image_safe
_ACTIVE_CONFIG: PipelineConfig | None = None
_CAMERA_STATS: list[dict] = []


def configure_source_projection(config: PipelineConfig) -> None:
    """Install or remove the noise-aware safe projection wrapper."""

    global _ACTIVE_CONFIG
    sensor_noise.validate_noise_profile(config)
    _ACTIVE_CONFIG = config
    reset_noise_stats()
    if sensor_noise.is_noise_enabled(config):
        source.draft_add_depth_image_safe = add_depth_image_safe
    else:
        source.draft_add_depth_image_safe = _ORIGINAL_DRAFT_ADD_DEPTH_IMAGE_SAFE


def reset_noise_stats() -> None:
    _CAMERA_STATS.clear()


def current_noise_summary() -> dict:
    if _ACTIVE_CONFIG is None:
        return {"enabled": False, "profile": "none", "camera_stats": []}
    return sensor_noise.summarize_camera_stats(list(_CAMERA_STATS), _ACTIVE_CONFIG)


def add_depth_image_safe(point_cloud_holder, image_path, intrinsic, extrinsic, add_noise=False, noise_hook=None):
    """Robert2 safe projection with MVP Helios2 Wide noise before merge."""

    if _ACTIVE_CONFIG is None or not sensor_noise.is_noise_enabled(_ACTIVE_CONFIG):
        return _ORIGINAL_DRAFT_ADD_DEPTH_IMAGE_SAFE(
            point_cloud_holder,
            image_path,
            intrinsic,
            extrinsic,
            add_noise=False,
        )

    width, height = intrinsic.width, intrinsic.height
    depth_data = np.fromfile(image_path, dtype=np.float32).reshape((height, width))

    near_clip, far_clip = azurion_dataset.depth_clip_for_raw_path(image_path)
    print("Near clip:", near_clip, "Far clip:", far_clip)
    depth_data[~azurion_dataset.valid_depth_mask(depth_data, near_clip, far_clip)] = 0.0
    depth_data = np.ascontiguousarray(np.flipud(depth_data))

    camera_index = sensor_noise.camera_index_from_path(image_path)
    sample_key = sensor_noise.sample_key_for_path(image_path, _ACTIVE_CONFIG)
    valid_full = azurion_dataset.valid_depth_mask(depth_data, near_clip, far_clip)
    noise_result = sensor_noise.apply_depth_noise(
        depth_data,
        valid_full,
        config=_ACTIVE_CONFIG,
        sample_key=sample_key,
        camera_index=camera_index,
        depth_path=image_path,
    )
    depth_data = noise_result.depth

    ext = np.array(extrinsic, dtype=np.float64)
    ext[1, :] *= -1
    ext[2, :] *= -1
    ext[:, 2] *= -1
    camera_to_world = np.linalg.inv(np.ascontiguousarray(ext, dtype=np.float64))

    stride = max(1, int(source.DRAFT_DEPTH_PIXEL_STRIDE))
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth_data[::stride, ::stride]

    valid = azurion_dataset.valid_depth_mask(sampled_depth, near_clip, far_clip)
    if not np.any(valid):
        stats = dict(noise_result.stats)
        stats["projected_point_count"] = 0
        _CAMERA_STATS.append(stats)
        return

    z = sampled_depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    u_offset, v_offset, lateral_stats = sensor_noise.lateral_jitter_offsets(
        len(z),
        config=_ACTIVE_CONFIG,
        sample_key=sample_key,
        camera_index=camera_index,
    )
    u = u + u_offset
    v = v + v_offset
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
    world_points = (camera_to_world @ camera_points).T[:, :3]
    world_points = world_points[np.all(np.isfinite(world_points), axis=1)]
    world_points = np.ascontiguousarray(world_points, dtype=np.float64)

    stats = dict(noise_result.stats)
    stats.update(lateral_stats)
    stats["projected_point_count"] = int(len(world_points))
    _CAMERA_STATS.append(stats)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    point_cloud_holder.point_cloud += pcd
