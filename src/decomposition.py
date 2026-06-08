"""Configured access to the pinned robert2 decomposition source."""

from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from . import source_robert2_decomposition as source


def apply_config(config: PipelineConfig) -> None:
    from . import reconstruction

    cfg = config.decomposition
    source.default_voxel_size = cfg.voxel_size
    source.default_room_dimensions = cfg.room_dimensions
    source.REMOVE_TABLE_IN_DECOMPOSITION = cfg.remove_table
    source.INCLUDE_TABLE_IN_DECOMPOSITION = not cfg.remove_table
    source.DRAFT_USE_SAFE_DEPTH_PROJECTION = cfg.safe_depth_projection
    source.REQUIRE_STABLE_OPEN3D_RUNTIME = cfg.require_stable_open3d_runtime
    source.DRAFT_DEPTH_PIXEL_STRIDE = cfg.depth_pixel_stride
    source.SPLIT_OBJECTS_INTO_SMALLER_BOXES = cfg.split_objects_into_smaller_boxes
    source.SPLIT_MAX_BOX_EDGE_M = cfg.split_max_box_edge_m
    source.SPLIT_MIN_VOXELS_PER_BOX = cfg.split_min_voxels_per_box
    source.SPLIT_MAX_RECURSION_DEPTH = cfg.split_max_recursion_depth
    source.SPLIT_MAX_BOXES_PER_OBJECT = cfg.split_max_boxes_per_object
    source.MIN_COMPONENT_VOXELS = cfg.min_component_voxels
    source.COMPONENT_CONNECTIVITY = cfg.component_connectivity
    source.OBJECT_BOX_MARGIN_M = cfg.object_box_margin_m
    source.CONVEX_HULL_MARGIN_M = cfg.component_convex_hull_margin_m
    source.REMOVE_ROOM_BOUNDARY_VOXELS = cfg.remove_room_boundary_voxels
    source.ROOM_BOUNDARY_MARGIN_M = cfg.room_boundary_margin_m
    source.FLOOR_REMOVAL_HEIGHT_M = cfg.floor_removal_height_m
    source.CEILING_REMOVAL_MARGIN_M = cfg.ceiling_removal_margin_m
    source.TABLE_REMOVAL_MARGIN_M = cfg.table_removal_margin_m
    source.REMOVE_TABLE_EDGE_COMPONENTS = cfg.remove_table_edge_components
    source.TABLE_EDGE_COMPONENT_MARGIN_M = cfg.table_edge_component_margin_m
    source.TABLE_EDGE_COMPONENT_MAX_VOXELS = cfg.table_edge_component_max_voxels
    source.REMOVE_TABLE_LOW_SUPPORT_CLEANUP = cfg.remove_table_low_support_cleanup
    source.TABLE_LOW_SUPPORT_MARGIN_M = cfg.table_low_support_margin_m
    source.TABLE_LOW_SUPPORT_HEIGHT_M = cfg.table_low_support_height_m
    source.TABLE_FIXTURE_EXPAND_VOXELS = cfg.table_fixture_expand_voxels
    source.TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS = cfg.table_fixture_protect_props_expand_voxels
    source.MERGE_OCCLUSIONS_WITH_OBJECTS = cfg.merge_occlusions_with_objects
    source.INCLUDE_DETACHED_OCCLUSION_ZONES = cfg.include_detached_occlusion_zones
    source.BUILD_OCCLUSION_ZONE_BOXES = cfg.build_occlusion_zone_boxes
    source.SHOW_OCCLUSION_ZONE_BOXES = False
    source.SHOW_BLIND_SPOTS = cfg.show_blind_spots
    source.OCCLUSION_BOX_MARGIN_M = cfg.occlusion_box_margin_m
    source.OCCLUSION_ATTACHMENT_RADIUS_VOXELS = cfg.occlusion_attachment_radius_voxels
    source.OCCLUSION_MIN_COMPONENT_VOXELS = cfg.occlusion_min_component_voxels
    source.OCCLUSION_MAX_COMPONENT_VOXELS = cfg.occlusion_max_component_voxels
    source.OCCLUSION_SURFACE_CLEARANCE_VOXELS = cfg.occlusion_surface_clearance_voxels
    source.OCCLUSION_MAX_BOXES_PER_ZONE = cfg.occlusion_max_boxes_per_zone
    source.FILL_ONE_VOXEL_GAPS = cfg.fill_one_voxel_gaps
    source.GAP_FILL_ITERATIONS = cfg.gap_fill_iterations
    source.OPEN_INTERACTIVE_VIEW = False
    source.DISPLAY_OCCLUSION_GRID = False
    source.SHOW_SIDE_RGB_IMAGE = False
    source.SHOW_ROBOT_HULL_OVERLAY = False
    source.SHOW_FULL_VOXEL_GRID = False
    source.SHOW_PHANTOM_FREE_SPACE_DIAGNOSTIC = False
    reconstruction.configure_source_projection(config)


def process_sample(dataset, sample_index: int, config: PipelineConfig) -> dict:
    from . import reconstruction

    apply_config(config)
    reconstruction.reset_noise_stats()
    result = source.draft_process_sample(dataset, sample_index)
    result["noise"] = reconstruction.current_noise_summary()
    return result


def key_counts(result: dict) -> dict:
    stats = result.get("occlusion_zone_stats", {}) or {}
    counts = {
        "raw_occupied_count": int(result.get("raw_occupied_count", 0)),
        "decomposition_occupied_count": int(result.get("decomposition_occupied_count", 0)),
        "occlusion_not_free": int(result.get("occlusion_not_free", 0)),
        "component_count": len(result.get("components", [])),
        "entity_hull_count": len(result.get("hull_records", [])),
        "occlusion_hull_count": len(result.get("occlusion_hull_records", [])),
        "blind_spot_hull_count": len(result.get("blind_spot_hull_records", [])),
        "occlusion_selected_components": int(stats.get("selected_components", 0)),
        "occlusion_selected_voxels": int(stats.get("selected_voxels", 0)),
    }
    noise = result.get("noise", {}) or {}
    if noise.get("enabled"):
        totals = noise.get("totals", {}) or {}
        counts.update(
            {
                "noise_profile": noise.get("profile", ""),
                "noise_valid_before": int(totals.get("original_valid_count", 0)),
                "noise_valid_after": int(totals.get("valid_after_count", 0)),
                "noise_edge_dropouts": int(totals.get("edge_dropout_count", 0)),
                "noise_far_dropouts": int(totals.get("far_dropout_count", 0)),
                "noise_flying_pixels": int(totals.get("flying_pixel_count", 0)),
                "noise_outliers": int(totals.get("outlier_count", 0)),
            }
        )
    return counts


def empty_bool_like(shape):
    return np.zeros(shape, dtype=bool)
