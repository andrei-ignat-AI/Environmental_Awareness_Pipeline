"""Typed settings consumed by the pipeline internals.

Users should edit the root-level ``config.py``. This module converts those
simple constants into the dataclass shape used by the copied computation code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_CONFIG = import_module("config")


def _user_path(name: str, default: Path) -> Path:
    value = getattr(USER_CONFIG, name, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dataset_root() -> Path:
    override = getattr(USER_CONFIG, "CAPTURE_ROOT", None)
    if override:
        path = Path(override)
        return path if path.is_absolute() else PROJECT_ROOT / path
    mode = str(getattr(USER_CONFIG, "DATASET_MODE", "demo")).lower()
    if mode == "demo":
        return PROJECT_ROOT / "DepthCaptures_demo"
    if mode == "full":
        return PROJECT_ROOT / "DepthCaptures"
    raise ValueError("DATASET_MODE must be 'demo' or 'full'")


@dataclass
class SourceRefs:
    robert2_commit: str = "ff3bf25"
    camera_rig_metrics_source: str = "copied from current camera-rig metric script"
    planning_source: str = "copied from current Ruben-style informed RRT* script"


@dataclass
class DatasetConfig:
    capture_root: Path = field(default_factory=_dataset_root)
    rig_id: str | None = getattr(USER_CONFIG, "RIG_ID", None)
    layout: str | None = getattr(USER_CONFIG, "LAYOUT", None)
    sample_indices: tuple[int, ...] | None = getattr(USER_CONFIG, "SAMPLE_INDICES", None)


@dataclass
class DecompositionConfig:
    voxel_size: float = getattr(USER_CONFIG, "VOXEL_SIZE_M", 0.05)
    room_dimensions: tuple[float, float, float] = getattr(USER_CONFIG, "ROOM_DIMENSIONS_M", (9.9, 2.7, 5.9))
    remove_table: bool = getattr(USER_CONFIG, "REMOVE_TABLE", True)
    safe_depth_projection: bool = getattr(USER_CONFIG, "SAFE_DEPTH_PROJECTION", True)
    require_stable_open3d_runtime: bool = getattr(USER_CONFIG, "REQUIRE_STABLE_OPEN3D_RUNTIME", True)
    depth_pixel_stride: int = getattr(USER_CONFIG, "DEPTH_PIXEL_STRIDE", 1)
    split_objects_into_smaller_boxes: bool = getattr(USER_CONFIG, "SPLIT_OBJECTS_INTO_SMALLER_BOXES", True)
    split_max_box_edge_m: float = getattr(USER_CONFIG, "SPLIT_MAX_BOX_EDGE_M", 0.25)
    split_min_voxels_per_box: int = getattr(USER_CONFIG, "SPLIT_MIN_VOXELS_PER_BOX", 12)
    split_max_recursion_depth: int = getattr(USER_CONFIG, "SPLIT_MAX_RECURSION_DEPTH", 10)
    split_max_boxes_per_object: int = getattr(USER_CONFIG, "SPLIT_MAX_BOXES_PER_OBJECT", 25)
    min_component_voxels: int = getattr(USER_CONFIG, "MIN_COMPONENT_VOXELS", 25)
    component_connectivity: int = getattr(USER_CONFIG, "COMPONENT_CONNECTIVITY", 6)
    object_box_margin_m: float = getattr(USER_CONFIG, "OBJECT_BOX_MARGIN_M", 0.10)
    component_convex_hull_margin_m: float = getattr(USER_CONFIG, "COMPONENT_CONVEX_HULL_MARGIN_M", 0.05)
    remove_room_boundary_voxels: bool = getattr(USER_CONFIG, "REMOVE_ROOM_BOUNDARY_VOXELS", True)
    room_boundary_margin_m: float = getattr(USER_CONFIG, "ROOM_BOUNDARY_MARGIN_M", 0.16)
    floor_removal_height_m: float = getattr(USER_CONFIG, "FLOOR_REMOVAL_HEIGHT_M", 0.03)
    ceiling_removal_margin_m: float = getattr(USER_CONFIG, "CEILING_REMOVAL_MARGIN_M", 0.03)
    table_removal_margin_m: float = getattr(USER_CONFIG, "TABLE_REMOVAL_MARGIN_M", 0.05)
    remove_table_edge_components: bool = getattr(USER_CONFIG, "REMOVE_TABLE_EDGE_COMPONENTS", True)
    table_edge_component_margin_m: float = getattr(USER_CONFIG, "TABLE_EDGE_COMPONENT_MARGIN_M", 0.13)
    table_edge_component_max_voxels: int = getattr(USER_CONFIG, "TABLE_EDGE_COMPONENT_MAX_VOXELS", 400)
    remove_table_low_support_cleanup: bool = getattr(USER_CONFIG, "REMOVE_TABLE_LOW_SUPPORT_CLEANUP", True)
    table_low_support_margin_m: float = getattr(USER_CONFIG, "TABLE_LOW_SUPPORT_MARGIN_M", 0.08)
    table_low_support_height_m: float = getattr(USER_CONFIG, "TABLE_LOW_SUPPORT_HEIGHT_M", 0.18)
    table_fixture_expand_voxels: int = getattr(USER_CONFIG, "TABLE_FIXTURE_EXPAND_VOXELS", 1)
    table_fixture_protect_props_expand_voxels: int = getattr(USER_CONFIG, "TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS", 1)
    merge_occlusions_with_objects: bool = getattr(USER_CONFIG, "MERGE_OCCLUSIONS_WITH_OBJECTS", False)
    include_detached_occlusion_zones: bool = getattr(USER_CONFIG, "INCLUDE_DETACHED_OCCLUSION_ZONES", True)
    build_occlusion_zone_boxes: bool = getattr(USER_CONFIG, "BUILD_OCCLUSION_ZONE_BOXES", True)
    show_blind_spots: bool = getattr(USER_CONFIG, "SHOW_BLIND_SPOTS", True)
    occlusion_box_margin_m: float = getattr(USER_CONFIG, "OCCLUSION_BOX_MARGIN_M", 0.05)
    occlusion_attachment_radius_voxels: int = getattr(USER_CONFIG, "OCCLUSION_ATTACHMENT_RADIUS_VOXELS", 1)
    occlusion_min_component_voxels: int = getattr(USER_CONFIG, "OCCLUSION_MIN_COMPONENT_VOXELS", 25)
    occlusion_max_component_voxels: int = getattr(USER_CONFIG, "OCCLUSION_MAX_COMPONENT_VOXELS", 100000)
    occlusion_surface_clearance_voxels: int = getattr(USER_CONFIG, "OCCLUSION_SURFACE_CLEARANCE_VOXELS", 1)
    occlusion_max_boxes_per_zone: int = getattr(USER_CONFIG, "OCCLUSION_MAX_BOXES_PER_ZONE", 4)
    fill_one_voxel_gaps: bool = getattr(USER_CONFIG, "FILL_ONE_VOXEL_GAPS", True)
    gap_fill_iterations: int = getattr(USER_CONFIG, "GAP_FILL_ITERATIONS", 1)


@dataclass
class MujocoExportConfig:
    export_single_box_per_object: bool = getattr(USER_CONFIG, "EXPORT_SINGLE_BOX_PER_OBJECT", False)
    remove_table: bool = getattr(USER_CONFIG, "REMOVE_TABLE_FROM_MUJOCO", True)
    merge_occlusions_with_objects: bool = getattr(USER_CONFIG, "MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT", False)
    export_occlusion_zone_boxes: bool = getattr(USER_CONFIG, "EXPORT_OCCLUSION_ZONE_BOXES", True)
    export_blind_spots: bool = getattr(USER_CONFIG, "EXPORT_BLIND_SPOTS", True)
    export_geom_mode: str = getattr(USER_CONFIG, "EXPORT_GEOM_MODE", "convex_hull")
    convex_hull_margin_m: float = getattr(USER_CONFIG, "MUJOCO_CONVEX_HULL_MARGIN_M", 0.10)
    manual_diagnostic_shift: tuple[float, float, float] = getattr(USER_CONFIG, "MANUAL_DIAGNOSTIC_SHIFT_M", (0.0, 0.0, 0.0))
    include_robot: bool = getattr(USER_CONFIG, "INCLUDE_ROBOT", False)
    robot_hulls_folder: str = str(_user_path("ROBOT_HULLS_FOLDER", Path("assets/Robotarm/hulls")))
    validate_axis_correspondence: bool = getattr(USER_CONFIG, "VALIDATE_AXIS_CORRESPONDENCE", True)
    axis_check_tolerance_m: float = getattr(USER_CONFIG, "AXIS_CHECK_TOLERANCE_M", 0.02)
    export_voxels: bool = getattr(USER_CONFIG, "EXPORT_VOXELS", True)


@dataclass
class CameraRigMetricsConfig:
    enabled: bool = getattr(USER_CONFIG, "RUN_CAMERA_RIG_METRICS", True)
    metric_scope: str = getattr(USER_CONFIG, "METRIC_SCOPE", "roi")
    depth_stride: int = getattr(USER_CONFIG, "METRIC_DEPTH_STRIDE", 1)
    skip_figures: bool = getattr(USER_CONFIG, "SKIP_METRIC_FIGURES", False)
    distance_thresholds_cm: tuple[float, ...] = getattr(USER_CONFIG, "DISTANCE_THRESHOLDS_CM", (5.0, 10.0, 15.0))
    headline_threshold_cm: float = getattr(USER_CONFIG, "HEADLINE_THRESHOLD_CM", 10.0)
    plot_style: str | None = getattr(USER_CONFIG, "PLOT_STYLE", None)


@dataclass
class NoiseConfig:
    profile: str = getattr(USER_CONFIG, "NOISE_MODE", "robust")
    valid_profiles: tuple[str, ...] = ("none", "robust")
    global_seed: int = getattr(USER_CONFIG, "NOISE_GLOBAL_SEED", 20260602)
    range_mode_m: float = 8.3
    active_min_m: float = 0.30
    active_max_m: float = 8.30
    axial_multiplier: float = 1.0
    axial_truncate_sigma: float = 3.0
    precision_curve_m: tuple[tuple[float, float], ...] = (
        (0.5, 0.00092),
        (1.0, 0.00081),
        (1.5, 0.00134),
        (2.0, 0.00202),
        (3.0, 0.00549),
        (4.0, 0.00657),
        (5.0, 0.01003),
        (6.0, 0.01294),
        (7.0, 0.02068),
        (8.0, 0.02548),
        (8.3, 0.03000),
    )
    camera_bias_sigma_m: float = 0.002
    camera_bias_clip_m: float = 0.005
    lateral_jitter_px: float = 0.03
    edge_window_px: int = 3
    edge_min_depth_jump_m: float = 0.05
    edge_sigma_multiplier: float = 4.0
    edge_dropout_rate: float = 0.03
    far_dropout_start_m: float = 7.0
    far_dropout_max_rate: float = 0.03
    robust_flying_pixel_rate: float = 0.005
    flying_pixel_fraction_min: float = 0.20
    flying_pixel_fraction_max: float = 0.80
    robust_outlier_rate: float = 0.0002
    outlier_delete_probability: float = 0.80
    outlier_displacement_min_m: float = 0.05
    outlier_displacement_max_m: float = 0.10


@dataclass
class PlanningConfig:
    enabled: bool = getattr(USER_CONFIG, "RUN_PLANNING", False)
    goal_poi: tuple[float, float, float] = getattr(USER_CONFIG, "PLANNING_TARGET_M", (3.0, 2.0, 1.0))
    max_iterations: int = getattr(USER_CONFIG, "PLANNING_MAX_ITERATIONS", 1500)
    step_size: float = getattr(USER_CONFIG, "PLANNING_STEP_SIZE", 0.20)
    goal_sample_rate: float = getattr(USER_CONFIG, "PLANNING_GOAL_SAMPLE_RATE", 0.12)
    goal_tolerance: float = getattr(USER_CONFIG, "PLANNING_GOAL_TOLERANCE", 0.08)
    rewire_radius: float = getattr(USER_CONFIG, "PLANNING_REWIRE_RADIUS", 0.55)
    collision_check_step: float = getattr(USER_CONFIG, "PLANNING_COLLISION_CHECK_STEP", 0.08)
    ik_random_restarts: int = getattr(USER_CONFIG, "PLANNING_IK_RANDOM_RESTARTS", 80)
    max_goal_candidates: int = getattr(USER_CONFIG, "PLANNING_MAX_GOAL_CANDIDATES", 12)


@dataclass
class OutputConfig:
    root: Path = field(default_factory=lambda: _user_path("OUTPUT_ROOT", Path("outputs")))
    run_id: str | None = getattr(USER_CONFIG, "RUN_ID", None)
    continue_on_sample_failure: bool = getattr(USER_CONFIG, "CONTINUE_ON_SAMPLE_FAILURE", True)


@dataclass
class PipelineConfig:
    source_refs: SourceRefs = field(default_factory=SourceRefs)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    mujoco_export: MujocoExportConfig = field(default_factory=MujocoExportConfig)
    metrics_camera_rig: CameraRigMetricsConfig = field(default_factory=CameraRigMetricsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return normalize(asdict(self))


def default_config() -> PipelineConfig:
    return PipelineConfig()
