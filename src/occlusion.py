"""Occlusion and blind-spot wrappers from robert2."""

from __future__ import annotations

from . import source_robert2_decomposition as source


def compute_occlusion_grid(depth_image_paths, intrinsics, extrinsics, resolution, room_dimensions):
    return source.compute_occlusion_grid(
        depth_image_paths,
        intrinsics,
        extrinsics,
        resolution=resolution,
        room_dimensions=room_dimensions,
    )


def compute_blind_spot_grid(depth_image_paths, intrinsics, extrinsics, resolution, room_dimensions):
    return source.compute_blind_spot_grid(
        depth_image_paths,
        intrinsics,
        extrinsics,
        resolution=resolution,
        room_dimensions=room_dimensions,
    )


def find_occlusion_zone_components(visible_grid, occlusion_grid, min_voxels, connectivity, include_detached):
    return source.draft_find_occlusion_zone_components(
        visible_grid,
        occlusion_grid,
        min_voxels,
        connectivity,
        include_detached,
    )

