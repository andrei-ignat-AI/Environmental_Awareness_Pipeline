"""MuJoCo export orchestration around the pinned robert2 exporter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from . import decomposition
from . import reconstruction
from . import source_robert2_mujoco_export as source


def apply_config(config: PipelineConfig, export_root: Path) -> None:
    source.decomposition = decomposition.source
    source.DATASET_FOLDER = str(config.dataset.capture_root)
    source.EXPORT_FOLDER = str(export_root)
    cfg = config.mujoco_export
    source.EXPORT_SINGLE_BOX_PER_OBJECT = cfg.export_single_box_per_object
    source.REMOVE_TABLE_IN_MUJOCO_EXPORT = cfg.remove_table
    source.INCLUDE_TABLE_IN_MUJOCO_EXPORT = not cfg.remove_table
    source.MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT = cfg.merge_occlusions_with_objects
    source.EXPORT_OCCLUSION_ZONE_BOXES = cfg.export_occlusion_zone_boxes
    source.EXPORT_BLIND_SPOTS = cfg.export_blind_spots
    source.EXPORT_GEOM_MODE = cfg.export_geom_mode
    source.CONVEX_HULL_MARGIN_M = cfg.convex_hull_margin_m
    source.MANUAL_DIAGNOSTIC_SHIFT = np.asarray(cfg.manual_diagnostic_shift, dtype=np.float64)
    source.INCLUDE_ROBOT = cfg.include_robot
    source.ROBOT_HULLS_FOLDER = cfg.robot_hulls_folder
    source.VALIDATE_AXIS_CORRESPONDENCE = cfg.validate_axis_correspondence
    source.AXIS_CHECK_TOLERANCE_M = cfg.axis_check_tolerance_m
    source.EXPORT_ALL = False
    source.SHOW_MUJOCO_VIEWER = False
    source.SHOW_OPEN3D_DECOMPOSITION = False
    source.SHOW_SIDE_RGB_IMAGE = False
    source.configure_decomposition_for_export()
    decomposition.apply_config(config)
    decomposition.source.CONVEX_HULL_MARGIN_M = cfg.convex_hull_margin_m


def export_sample(dataset, sample_index: int, config: PipelineConfig, export_root: Path):
    apply_config(config, export_root)
    reconstruction.reset_noise_stats()
    result, xml_path, json_path, model, data = source.load_sample_for_export(dataset, sample_index)
    result["noise"] = reconstruction.current_noise_summary()
    sample_path = Path(result["sample_path"])
    sample_name = sample_path.name
    sample_rel = Path(xml_path).parent.relative_to(export_root)
    export_dir = export_root / sample_rel
    voxel_indices_path = None
    voxel_ply_path = None
    if config.mujoco_export.export_voxels:
        source.export_voxels_from_mujoco(
            model,
            data,
            sample_name,
            voxel_size=config.decomposition.voxel_size,
            export_dir=str(export_dir),
        )
        voxel_indices_path = export_dir / f"{sample_name}_voxel_indices.npy"
        voxel_ply_path = export_dir / f"{sample_name}_voxels.ply"
    return result, Path(xml_path), Path(json_path), voxel_indices_path, voxel_ply_path, model, data
