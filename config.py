"""Master configuration for the Environmental Awareness Pipeline.

Edit this file for normal use. Command-line flags can override the common
sample, rig, noise, and run-id options without changing this file.
"""

from pathlib import Path


# =============================================================================
# Project Paths
# These paths are resolved from the repository root unless absolute paths are
# provided. Put the full Unity export dataset in DepthCaptures/ for full runs.
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_MODE = "demo"  # "demo" uses DepthCaptures_demo/, "full" uses DepthCaptures/
CAPTURE_ROOT = None  # Optional explicit path; leave None for DATASET_MODE routing.
OUTPUT_ROOT = "outputs"
ROBOT_HULLS_FOLDER = "assets/Robotarm/hulls"


# =============================================================================
# Run Selection
# These defaults affect run-demo, run-full, and recompute-temp unless the CLI
# provides --rig, --sample, --layout, or --run-id.
# =============================================================================

RUN_ID = None
RIG_ID = None
LAYOUT = None
SAMPLE_INDICES = None
CONTINUE_ON_SAMPLE_FAILURE = True


# =============================================================================
# Noise Model
# "robust" is the report default. "none" disables synthetic depth noise.
# The numeric robust-noise parameters are intentionally kept in src/config.py
# because they are part of the preserved computation path.
# =============================================================================

NOISE_MODE = "robust"  # "robust" or "none"
NOISE_GLOBAL_SEED = 20260602


# =============================================================================
# Reconstruction And Decomposition
# These values control point projection, voxelization, table cleanup, connected
# components, object hull margins, and occlusion-zone generation.
# =============================================================================

VOXEL_SIZE_M = 0.05
ROOM_DIMENSIONS_M = (9.9, 2.7, 5.9)
REMOVE_TABLE = True
SAFE_DEPTH_PROJECTION = True
REQUIRE_STABLE_OPEN3D_RUNTIME = True
DEPTH_PIXEL_STRIDE = 1

SPLIT_OBJECTS_INTO_SMALLER_BOXES = True
SPLIT_MAX_BOX_EDGE_M = 0.25
SPLIT_MIN_VOXELS_PER_BOX = 12
SPLIT_MAX_RECURSION_DEPTH = 10
SPLIT_MAX_BOXES_PER_OBJECT = 25

MIN_COMPONENT_VOXELS = 25
COMPONENT_CONNECTIVITY = 6
OBJECT_BOX_MARGIN_M = 0.10
COMPONENT_CONVEX_HULL_MARGIN_M = 0.05

REMOVE_ROOM_BOUNDARY_VOXELS = True
ROOM_BOUNDARY_MARGIN_M = 0.16
FLOOR_REMOVAL_HEIGHT_M = 0.03
CEILING_REMOVAL_MARGIN_M = 0.03
TABLE_REMOVAL_MARGIN_M = 0.05
REMOVE_TABLE_EDGE_COMPONENTS = True
TABLE_EDGE_COMPONENT_MARGIN_M = 0.13
TABLE_EDGE_COMPONENT_MAX_VOXELS = 400
REMOVE_TABLE_LOW_SUPPORT_CLEANUP = True
TABLE_LOW_SUPPORT_MARGIN_M = 0.08
TABLE_LOW_SUPPORT_HEIGHT_M = 0.18
TABLE_FIXTURE_EXPAND_VOXELS = 1
TABLE_FIXTURE_PROTECT_PROPS_EXPAND_VOXELS = 1

MERGE_OCCLUSIONS_WITH_OBJECTS = False
INCLUDE_DETACHED_OCCLUSION_ZONES = True
BUILD_OCCLUSION_ZONE_BOXES = True
SHOW_OCCLUSION_ZONE_BOXES = True
SHOW_BLIND_SPOTS = True
OCCLUSION_BOX_MARGIN_M = 0.05
OCCLUSION_ATTACHMENT_RADIUS_VOXELS = 1
OCCLUSION_MIN_COMPONENT_VOXELS = 25
OCCLUSION_MAX_COMPONENT_VOXELS = 100000
OCCLUSION_SURFACE_CLEARANCE_VOXELS = 1
OCCLUSION_MAX_BOXES_PER_ZONE = 4
FILL_ONE_VOXEL_GAPS = True
GAP_FILL_ITERATIONS = 1


# =============================================================================
# MuJoCo Export
# These options control XML/box/voxel export. Robot inclusion is off by default
# because the reported collision-geometry metrics use robot-off exports.
# =============================================================================

EXPORT_SINGLE_BOX_PER_OBJECT = False
REMOVE_TABLE_FROM_MUJOCO = True
MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT = False
EXPORT_OCCLUSION_ZONE_BOXES = True
EXPORT_BLIND_SPOTS = True
EXPORT_GEOM_MODE = "convex_hull"
MUJOCO_CONVEX_HULL_MARGIN_M = 0.10
MANUAL_DIAGNOSTIC_SHIFT_M = (0.0, 0.0, 0.0)
INCLUDE_ROBOT = False
VALIDATE_AXIS_CORRESPONDENCE = True
AXIS_CHECK_TOLERANCE_M = 0.02
EXPORT_VOXELS = True


# =============================================================================
# Metrics
# Camera-rig metrics produce visibility, trade-off, free-space, and figure
# outputs. Collision-geometry metrics are generated from existing exported PLYs.
# =============================================================================

RUN_CAMERA_RIG_METRICS = True
METRIC_SCOPE = "roi"
METRIC_DEPTH_STRIDE = 1
SKIP_METRIC_FIGURES = False
DISTANCE_THRESHOLDS_CM = (5.0, 10.0, 15.0)
HEADLINE_THRESHOLD_CM = 10.0
PLOT_STYLE = None


# =============================================================================
# Planning
# Planning is disabled during default report reproduction. Viewer commands can
# attempt planning against an exported MuJoCo sample on demand.
# =============================================================================

RUN_PLANNING = False
PLANNING_TARGET_M = (3.0, 2.0, 1.0)
PLANNING_MAX_ITERATIONS = 1500
PLANNING_STEP_SIZE = 0.20
PLANNING_GOAL_SAMPLE_RATE = 0.12
PLANNING_GOAL_TOLERANCE = 0.08
PLANNING_REWIRE_RADIUS = 0.55
PLANNING_COLLISION_CHECK_STEP = 0.08
PLANNING_IK_RANDOM_RESTARTS = 80
PLANNING_MAX_GOAL_CANDIDATES = 12
