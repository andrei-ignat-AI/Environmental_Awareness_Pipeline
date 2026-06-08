#!/usr/bin/env python3
"""Camera-rig metrics generator for Azurion Unity captures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


# +++++=====+++++ fixed experiment definitions +++++=====+++++

DEFAULT_CAPTURE_ROOT = Path("DepthCaptures")
DEFAULT_RESULTS_DIR = Path("camera_rig_metrics")
DEFAULT_PLOT_STYLE = Path("results") / "figures" / "styles" / "ieee_matlab.mplstyle"
CM = 1 / 2.54
IEEE_SINGLE = (8.89 * CM, 5.2 * CM)
IEEE_DOUBLE = (18.2 * CM, 8.0 * CM)
IEEE_TALL = (18.2 * CM, 10.2 * CM)
DEFAULT_NEAR_CLIP = 0.3
DEFAULT_FAR_CLIP = 8.3
FAR_CLIP_EPSILON = 0.01
DEFAULT_HORIZONTAL_FOV_DEGREES = 108.0
FLOORPLAN_FOV_RADIUS_M = 2.7
DEPTH_VISIBILITY_TOLERANCE_M = 0.075
SURFACE_TOLERANCE_VOXELS = 1
DEFAULT_DISTANCE_THRESHOLDS_CM = (5.0, 10.0, 15.0)
DEFAULT_HEADLINE_THRESHOLD_CM = 10.0
TABLE_REMOVAL_MARGIN_M = 0.05
TABLE_FIXTURE_EXPAND_VOXELS = 1
ROOM_BOUNDARY_MARGIN_M = 0.16
FLOOR_REMOVAL_HEIGHT_M = 0.03
CEILING_REMOVAL_MARGIN_M = 0.03
ROBERT_MISSING_WEIGHT = 10.0
ROBERT_EXTRA_WEIGHT = 1.0
RIG_4CAM_CLASSIC = "4CamClassic"
RIG_3CAM = "3Cam"
RIG_4CAM_ASYM = "4CamAsym"
RIG_5CAM = "5Cam"

RIG_ORDER = [
    RIG_4CAM_CLASSIC,
    RIG_3CAM,
    RIG_4CAM_ASYM,
    RIG_5CAM,
]

EXPECTED_CAMERA_COUNTS = {
    RIG_4CAM_CLASSIC: 4,
    RIG_3CAM: 3,
    RIG_4CAM_ASYM: 4,
    RIG_5CAM: 5,
}

LAYOUT_ORDER = [
    "C1_standard_cath",
    "C2_radial_echo",
    "N1_mirrored_head",
    "N2_mirrored_support",
]

ROI_BOUNDS = {
    "x": (-4.00, 4.00),
    "y": (0.00, 2.35),
    "z": (-2.00, 2.00),
}

NAVIGATION_SCORE_REDUNDANT_WEIGHT = 0.35
NAVIGATION_SCORE_STRONG_REDUNDANCY_WEIGHT = 0.20
NAVIGATION_SCORE_OBSERVABLE_WEIGHT = 0.10
NAVIGATION_SCORE_NONBLIND_WEIGHT = 0.10
NAVIGATION_SCORE_CERTIFIED_FREE_WEIGHT = 0.10
NAVIGATION_SCORE_REDUNDANT_FREE_WEIGHT = 0.15

SEMANTIC_PRIORITIES = {
    "patient": "P1",
    "doctor_operator": "P1",
    "radiation_shield": "P1",
    "doctor_anesthesia": "P2",
    "anesthesia_machine": "P2",
    "doctor_ultrasound": "P2",
    "ultrasound": "P2",
    "medical_trolley": "P2",
    "uhd_tv": "P3",
    "waste_bin": "P3",
    "ceiling_lamp": "P3",
}

SEMANTIC_ORDER = [
    "patient",
    "doctor_operator",
    "radiation_shield",
    "doctor_anesthesia",
    "anesthesia_machine",
    "doctor_ultrasound",
    "ultrasound",
    "medical_trolley",
    "uhd_tv",
    "waste_bin",
    "ceiling_lamp",
]

SEMANTIC_DISPLAY_NAMES = {
    "patient": "Patient",
    "doctor_operator": "Operator",
    "radiation_shield": "Radiation Shield",
    "doctor_anesthesia": "Anesthesia Doctor",
    "anesthesia_machine": "Anesthesia Machine",
    "doctor_ultrasound": "Ultrasound Doctor",
    "ultrasound": "Ultrasound",
    "medical_trolley": "Medical Trolley",
    "uhd_tv": "Display",
    "waste_bin": "Waste Bin",
    "ceiling_lamp": "Ceiling Lamp",
}

ROBOT_RELEVANT_SEMANTICS = [
    "doctor_operator",
    "doctor_anesthesia",
    "doctor_ultrasound",
    "radiation_shield",
    "anesthesia_machine",
    "ultrasound",
    "medical_trolley",
    "uhd_tv",
    "waste_bin",
]

ROBOT_RELEVANT_GROUPS = {
    "doctor_operator": "Staff",
    "doctor_anesthesia": "Staff",
    "doctor_ultrasound": "Staff",
    "radiation_shield": "Movable Equipment",
    "anesthesia_machine": "Movable Equipment",
    "ultrasound": "Movable Equipment",
    "medical_trolley": "Movable Equipment",
    "uhd_tv": "Movable Equipment",
    "waste_bin": "Low-Priority Obstacle",
}

LAYOUT_DISPLAY_NAMES = {
    "C1_standard_cath": "Cath baseline",
    "C2_radial_echo": "Cath shifted",
    "N1_mirrored_head": "Neuro head-side",
    "N2_mirrored_support": "Neuro shifted",
}

SUMMARY_METRICS = [
    "visible_by_at_least_1_camera",
    "redundant_coverage_fraction",
    "visible_by_at_least_3_cameras",
    "single_view_risk_fraction",
    "blind_surface_fraction",
    "navigation_visibility_score",
    "mean_view_diversity_deg",
]

SAMPLE_RE = re.compile(r"^sample_(\d+)$")
PIXEL_GRID_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


# +++++=====+++++ data containers +++++=====+++++

@dataclass(frozen=True)
class SampleRef:
    path: Path
    sample_name: str
    sample_index: int
    rig_id: str
    layout: str
    relative_path: str


@dataclass
class CameraData:
    index: int
    name: str
    role: str
    mount_wall: str
    depth_path: Path
    depth: np.ndarray
    valid_depth: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_clip: float
    far_clip: float
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    position: np.ndarray


@dataclass(frozen=True)
class GridInfo:
    metadata: dict
    shape: tuple[int, int, int]
    origin: np.ndarray
    voxel_size: float
    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray


# +++++=====+++++ small utilities +++++=====+++++

def parse_sample_indices(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    indices = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.add(int(part))
    return indices


def parse_float_list(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or str(value).strip() == "":
        return list(default)
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        return list(default)
    return sorted(set(values))


def threshold_label_cm(value_cm: float) -> str:
    value_cm = float(value_cm)
    if abs(value_cm - round(value_cm)) < 1e-9:
        return f"{int(round(value_cm))}cm"
    return f"{value_cm:g}cm".replace(".", "p")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sample_index_from_name(name: str) -> int | None:
    match = SAMPLE_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def is_sample_dir(path: Path) -> bool:
    return path.is_dir() and (path / "depth_metadata.json").is_file()


def safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator) / float(denominator)


def format_float(value: float | int | str | None, decimals: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def format_percent(value: float | int | str | None, decimals: int = 1) -> str:
    if value is None or isinstance(value, str):
        return format_float(value, decimals)
    return format_float(float(value) * 100.0, decimals)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = []
    lines.append("|" + "|".join(headers) + "|")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("|" + "|".join(str(item) for item in row) + "|")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def choose_first_sample(samples_by_index: dict[int, dict[str, SampleRef]]) -> SampleRef:
    first_index = sorted(samples_by_index)[0]
    return samples_by_index[first_index][RIG_ORDER[0]]
    rig_map = samples_by_index[first_index]
    for rig in RIG_ORDER:
        if rig in rig_map:
            return rig_map[rig]
    return next(iter(rig_map.values()))


def scope_display_name(metric_scope: str) -> str:
    return "ROI" if metric_scope == "roi" else "Whole-Room"


def scope_surface_key(metric_scope: str, suffix: str) -> str:
    return f"{metric_scope}_surface_{suffix}"


def strict_scope_surface_key(metric_scope: str, suffix: str) -> str:
    return f"strict_{metric_scope}_surface_{suffix}"


def resolve_plot_style_path(value: str | None) -> Path:
    style_path = DEFAULT_PLOT_STYLE if value is None else Path(value)
    if not style_path.is_absolute():
        style_path = Path.cwd() / style_path
    return style_path.resolve()


# +++++=====+++++ dataset discovery copied from azurion_dataset.py +++++=====+++++

def discover_samples(root: Path) -> list[SampleRef]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    if is_sample_dir(root):
        candidate_paths = [root]
        discovery_root = root.parent
    else:
        candidate_paths = []
        discovery_root = root
        for current_root, dirnames, _ in os.walk(root):
            current = Path(current_root)
            for dirname in dirnames:
                candidate = current / dirname
                if dirname.startswith("sample_") and is_sample_dir(candidate):
                    candidate_paths.append(candidate)

    samples: list[SampleRef] = []
    for sample_path in candidate_paths:
        metadata = load_json(sample_path / "depth_metadata.json")
        relative_parts = sample_path.relative_to(discovery_root).parts
        rig_id = str(metadata.get("cameraRigId") or "")
        layout = ""
        if len(relative_parts) >= 3:
            rig_id = rig_id or relative_parts[-3]
            layout = relative_parts[-2]
        elif len(relative_parts) >= 2:
            layout = relative_parts[-2]

        RIG_NAME_MAP = {
            "R0_LegacyCorner4": "4CamClassic",
            "R1_MinimalTriad3": "3Cam",
            "R2_HighWallCross4": "4CamAsym",
            "R3_HybridDense5": "5Cam",
        }
        rig_id = RIG_NAME_MAP.get(rig_id, rig_id)

        sample_index = sample_index_from_name(sample_path.name)
        if sample_index is None or not rig_id:
            continue

        samples.append(
            SampleRef(
                path=sample_path,
                sample_name=sample_path.name,
                sample_index=sample_index,
                rig_id=rig_id,
                layout=layout,
                relative_path=str(sample_path.relative_to(discovery_root)),
            )
        )

    samples.sort(
        key=lambda sample: (
            RIG_ORDER.index(sample.rig_id) if sample.rig_id in RIG_ORDER else 99,
            LAYOUT_ORDER.index(sample.layout) if sample.layout in LAYOUT_ORDER else 99,
            sample.sample_index,
            sample.relative_path,
        )
    )
    return samples


def group_samples_by_index(samples: Iterable[SampleRef]) -> dict[int, dict[str, SampleRef]]:
    grouped: dict[int, dict[str, SampleRef]] = {}
    for sample in samples:
        grouped.setdefault(sample.sample_index, {})[sample.rig_id] = sample
    return dict(sorted(grouped.items()))


# +++++=====+++++ unity voxel loading copied from azurion_dataset.py +++++=====+++++

def load_unity_voxels(sample_path: Path, kind: str) -> tuple[dict, np.ndarray]:
    if kind == "props":
        metadata_name = "voxel_metadata.json"
        raw_name = "voxel_props_occupancy.raw"
    elif kind == "robot":
        metadata_name = "voxel_robot_metadata.json"
        raw_name = "voxel_robot_occupancy.raw"
    elif kind == "scene":
        metadata_name = "voxel_scene_metadata.json"
        raw_name = "voxel_scene_occupancy.raw"
    else:
        raise ValueError(f"unknown voxel kind: {kind}")

    metadata = load_json(sample_path / metadata_name)
    raw_path = sample_path / str(metadata.get("fileName", raw_name))
    size_x = int(metadata["sizeX"])
    size_y = int(metadata["sizeY"])
    size_z = int(metadata["sizeZ"])
    expected = size_x * size_y * size_z
    data = np.fromfile(raw_path, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(f"{raw_path} has {data.size} bytes, expected {expected}")
    return metadata, data.astype(bool)


def reshape_unity_voxel_grid(metadata: dict, data: np.ndarray) -> np.ndarray:
    size_x = int(metadata["sizeX"])
    size_y = int(metadata["sizeY"])
    size_z = int(metadata["sizeZ"])
    flat = np.asarray(data, dtype=bool)
    expected = size_x * size_y * size_z
    if flat.size != expected:
        raise ValueError(f"voxel data has {flat.size} entries, expected {expected}")
    # unity writes x fastest, then z, then y. internal python grids use (x, y, z).
    return flat.reshape((size_x, size_z, size_y), order="F").transpose(0, 2, 1)


def grid_info_from_metadata(metadata: dict) -> GridInfo:
    size_x = int(metadata["sizeX"])
    size_y = int(metadata["sizeY"])
    size_z = int(metadata["sizeZ"])
    voxel_size = float(metadata.get("voxelSize", {}).get("x", 0.05))
    origin = np.array(
        [
            float(metadata.get("origin", {}).get("x", -size_x * voxel_size * 0.5)),
            float(metadata.get("origin", {}).get("y", 0.0)),
            float(metadata.get("origin", {}).get("z", -size_z * voxel_size * 0.5)),
        ],
        dtype=np.float64,
    )
    x_centers = np.arange(size_x, dtype=np.float64) * voxel_size + origin[0] + voxel_size * 0.5
    y_centers = np.arange(size_y, dtype=np.float64) * voxel_size + origin[1] + voxel_size * 0.5
    z_centers = np.arange(size_z, dtype=np.float64) * voxel_size + origin[2] + voxel_size * 0.5
    return GridInfo(
        metadata=metadata,
        shape=(size_x, size_y, size_z),
        origin=origin,
        voxel_size=voxel_size,
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers,
    )


def load_grid_bundle(sample_path: Path) -> tuple[GridInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    props_metadata, props_raw = load_unity_voxels(sample_path, "props")
    robot_metadata, robot_raw = load_unity_voxels(sample_path, "robot")
    scene_metadata, scene_raw = load_unity_voxels(sample_path, "scene")

    grid_info = grid_info_from_metadata(props_metadata)
    props_grid = reshape_unity_voxel_grid(props_metadata, props_raw)
    robot_grid = reshape_unity_voxel_grid(robot_metadata, robot_raw)
    scene_grid = reshape_unity_voxel_grid(scene_metadata, scene_raw)
    if props_grid.shape != robot_grid.shape or props_grid.shape != scene_grid.shape:
        raise ValueError(f"voxel grid shape mismatch under {sample_path}")

    table_fixture_grid = scene_grid & ~props_grid & ~robot_grid
    return grid_info, props_grid, robot_grid, scene_grid, table_fixture_grid


# +++++=====+++++ scene metadata masks copied/adapted from decompositionDraft.py +++++=====+++++

def scene_objects(sample_path: Path) -> list[dict]:
    return load_json(sample_path / "scene_objects.json").get("objects", [])


def object_bounds_record(item: dict) -> dict:
    position = item.get("position", {})
    size = item.get("boundsSize", item.get("scale", {}))
    return {
        "object_id": item.get("objectId", "scene_object"),
        "semantic_id": item.get("semanticId", item.get("objectId", "scene_object")),
        "category": item.get("category", ""),
        "center": np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ],
            dtype=np.float64,
        ),
        "size": np.array(
            [
                float(size.get("x", 0.0)),
                float(size.get("y", 0.0)),
                float(size.get("z", 0.0)),
            ],
            dtype=np.float64,
        ),
        "layout_family": item.get("layoutFamily", ""),
        "clinical_config": item.get("clinicalConfig", ""),
        "semantic_zone": item.get("semanticZone", ""),
    }


def table_bounds(sample_path: Path) -> list[dict]:
    return [object_bounds_record(item) for item in scene_objects(sample_path) if item.get("category") == "table"]


def prop_bounds(sample_path: Path) -> list[dict]:
    records = []
    for item in scene_objects(sample_path):
        if item.get("category") in {"table", "room"}:
            continue
        records.append(object_bounds_record(item))
    return records


def room_dimensions(sample_path: Path, grid_info: GridInfo) -> tuple[float, float, float]:
    for item in scene_objects(sample_path):
        if item.get("objectId") == "room_interior_bounds":
            size = item.get("boundsSize", {})
            return (
                float(size.get("x", grid_info.shape[0] * grid_info.voxel_size)),
                float(size.get("y", grid_info.shape[1] * grid_info.voxel_size)),
                float(size.get("z", grid_info.shape[2] * grid_info.voxel_size)),
            )
    return (
        grid_info.shape[0] * grid_info.voxel_size,
        grid_info.shape[1] * grid_info.voxel_size,
        grid_info.shape[2] * grid_info.voxel_size,
    )


def bounds_mask(grid_info: GridInfo, bounds: list[dict], margin_m: float) -> np.ndarray:
    mask = np.zeros(grid_info.shape, dtype=bool)
    if not bounds:
        return mask

    for item in bounds:
        size = np.asarray(item["size"], dtype=np.float64)
        if np.any(size <= 0.0):
            continue
        center = np.asarray(item["center"], dtype=np.float64)
        half_size = size * 0.5 + margin_m
        x_mask = (grid_info.x_centers >= center[0] - half_size[0]) & (grid_info.x_centers <= center[0] + half_size[0])
        y_mask = (grid_info.y_centers >= center[1] - half_size[1]) & (grid_info.y_centers <= center[1] + half_size[1])
        z_mask = (grid_info.z_centers >= center[2] - half_size[2]) & (grid_info.z_centers <= center[2] + half_size[2])
        mask[np.ix_(x_mask, y_mask, z_mask)] = True

    return mask


def core_roi_bounds_for_sample(sample_path: Path) -> dict[str, tuple[float, float]]:
    return {
        "x": tuple(ROI_BOUNDS["x"]),
        "y": tuple(ROI_BOUNDS["y"]),
        "z": tuple(ROI_BOUNDS["z"]),
    }


def roi_mask(grid_info: GridInfo, roi_bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    x_mask = (grid_info.x_centers >= roi_bounds["x"][0]) & (grid_info.x_centers <= roi_bounds["x"][1])
    y_mask = (grid_info.y_centers >= roi_bounds["y"][0]) & (grid_info.y_centers <= roi_bounds["y"][1])
    z_mask = (grid_info.z_centers >= roi_bounds["z"][0]) & (grid_info.z_centers <= roi_bounds["z"][1])
    mask = np.zeros(grid_info.shape, dtype=bool)
    mask[np.ix_(x_mask, y_mask, z_mask)] = True
    return mask


def room_boundary_mask(grid_info: GridInfo, dimensions: tuple[float, float, float]) -> np.ndarray:
    x_half = dimensions[0] * 0.5
    z_half = dimensions[2] * 0.5
    boundary_x = np.abs(grid_info.x_centers) >= (x_half - ROOM_BOUNDARY_MARGIN_M)
    boundary_z = np.abs(grid_info.z_centers) >= (z_half - ROOM_BOUNDARY_MARGIN_M)
    floor_y = grid_info.y_centers <= FLOOR_REMOVAL_HEIGHT_M
    ceiling_y = grid_info.y_centers >= (dimensions[1] - CEILING_REMOVAL_MARGIN_M)
    mask = np.zeros(grid_info.shape, dtype=bool)
    mask[boundary_x, :, :] = True
    mask[:, floor_y, :] = True
    mask[:, :, boundary_z] = True
    mask[:, ceiling_y, :] = True
    return mask


# +++++=====+++++ boolean morphology +++++=====+++++

def dilate_bool(grid: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return np.array(grid, copy=True)
    padded = np.pad(grid, radius, mode="constant", constant_values=False)
    result = np.zeros_like(grid, dtype=bool)
    sx, sy, sz = grid.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                result |= padded[
                    radius + dx : radius + dx + sx,
                    radius + dy : radius + dy + sy,
                    radius + dz : radius + dz + sz,
                ]
    return result


def surface_band_from_solid(grid: np.ndarray) -> np.ndarray:
    if not np.any(grid):
        return np.zeros_like(grid, dtype=bool)
    padded = np.pad(grid, 1, mode="constant", constant_values=False)
    sx, sy, sz = grid.shape
    interior = np.array(grid, copy=True)
    for dx, dy, dz in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        interior &= padded[1 + dx : 1 + dx + sx, 1 + dy : 1 + dy + sy, 1 + dz : 1 + dz + sz]
    return grid & ~interior


def table_removal_mask(sample_path: Path, grid_info: GridInfo, table_fixture_grid: np.ndarray) -> np.ndarray:
    # robert's comparison rule removes table and robot before evaluating scene content.
    fixture_mask = dilate_bool(table_fixture_grid, TABLE_FIXTURE_EXPAND_VOXELS)
    table_aabb = bounds_mask(grid_info, table_bounds(sample_path), TABLE_REMOVAL_MARGIN_M)
    protected_props = bounds_mask(grid_info, prop_bounds(sample_path), 0.0)
    table_aabb &= ~protected_props
    return fixture_mask | table_aabb


# +++++=====+++++ camera loading and safe projection copied from decompositionDraft.py +++++=====+++++

def valid_depth_mask(depth: np.ndarray, near_clip: float, far_clip: float) -> np.ndarray:
    max_depth = max(float(near_clip), float(far_clip) - FAR_CLIP_EPSILON)
    return np.isfinite(depth) & (depth > float(near_clip)) & (depth < max_depth)


def python_world_to_camera_matrix(camera: dict) -> np.ndarray:
    # this is the safe matrix convention used by draft_add_depth_image_safe and compute_occlusion_grid.
    if "worldToCameraMatrix" in camera:
        matrix = np.array(camera["worldToCameraMatrix"], dtype=np.float64).reshape(4, 4)
    elif "cameraToWorldMatrix" in camera:
        matrix = np.linalg.inv(np.array(camera["cameraToWorldMatrix"], dtype=np.float64).reshape(4, 4))
    else:
        raise KeyError("camera metadata needs worldToCameraMatrix or cameraToWorldMatrix")
    matrix = np.array(matrix, copy=True)
    matrix[1, :] *= -1.0
    matrix[2, :] *= -1.0
    matrix[:, 2] *= -1.0
    return matrix


def pixel_grid(width: int, height: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    key = (width, height, stride)
    cached = PIXEL_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    uu, vv = np.meshgrid(cols, rows)
    PIXEL_GRID_CACHE[key] = (uu, vv)
    return uu, vv


def load_cameras(sample: SampleRef) -> tuple[dict, list[CameraData]]:
    metadata = load_json(sample.path / "depth_metadata.json")
    width = int(metadata["width"])
    height = int(metadata["height"])
    raw_files = metadata.get("fullDepthRawFiles", [])
    cameras: list[CameraData] = []

    for camera in metadata.get("cameras", []):
        index = int(camera["index"])
        depth_file = raw_files[index] if index < len(raw_files) else f"cam{index}_depth.raw"
        depth_path = sample.path / depth_file
        depth = np.fromfile(depth_path, dtype=np.float32).reshape((height, width))
        depth = np.ascontiguousarray(np.flipud(depth))
        near_clip = float(camera.get("nearClip", metadata.get("nearClip", DEFAULT_NEAR_CLIP)))
        far_clip = float(camera.get("farClip", metadata.get("farClip", DEFAULT_FAR_CLIP)))
        valid = valid_depth_mask(depth, near_clip, far_clip)
        world_to_camera = python_world_to_camera_matrix(camera)
        camera_to_world = np.linalg.inv(world_to_camera)
        position = camera.get("position", {})
        cameras.append(
            CameraData(
                index=index,
                name=str(camera.get("name", f"cam{index}")),
                role=str(camera.get("cameraRole", "")),
                mount_wall=str(camera.get("mountWall", "")),
                depth_path=depth_path,
                depth=depth,
                valid_depth=valid,
                width=width,
                height=height,
                fx=float(camera["fx"]),
                fy=float(camera["fy"]),
                cx=float(camera["cx"]),
                cy=float(camera["cy"]),
                near_clip=near_clip,
                far_clip=far_clip,
                world_to_camera=world_to_camera,
                camera_to_world=camera_to_world,
                position=np.array(
                    [
                        float(position.get("x", 0.0)),
                        float(position.get("y", 0.0)),
                        float(position.get("z", 0.0)),
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return metadata, cameras


def reconstruct_depth_surface(
    cameras: list[CameraData],
    grid_info: GridInfo,
    depth_stride: int,
) -> np.ndarray:
    reconstruction = np.zeros(grid_info.shape, dtype=bool)
    shape_array = np.array(grid_info.shape, dtype=np.int32)
    stride = max(1, int(depth_stride))

    for camera in cameras:
        uu, vv = pixel_grid(camera.width, camera.height, stride)
        sampled_depth = camera.depth[::stride, ::stride]
        valid = camera.valid_depth[::stride, ::stride]
        if not np.any(valid):
            continue

        z = sampled_depth[valid].astype(np.float64)
        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        x = (u - camera.cx) * z / camera.fx
        y = (v - camera.cy) * z / camera.fy
        camera_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
        world_points = (camera.camera_to_world @ camera_points).T[:, :3]
        finite = np.all(np.isfinite(world_points), axis=1)
        if not np.any(finite):
            continue

        indices = np.floor((world_points[finite] - grid_info.origin) / grid_info.voxel_size).astype(np.int32)
        inside = np.all((indices >= 0) & (indices < shape_array), axis=1)
        indices = indices[inside]
        if indices.size:
            reconstruction[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return reconstruction


def project_points_to_camera(camera: CameraData, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        empty_bool = np.zeros(0, dtype=bool)
        empty_float = np.zeros(0, dtype=np.float64)
        return empty_bool, empty_float, empty_float

    points_h = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    camera_coords = (camera.world_to_camera @ points_h.T).T
    z_cam = camera_coords[:, 2]
    projectable = (
        np.isfinite(camera_coords[:, 0])
        & np.isfinite(camera_coords[:, 1])
        & np.isfinite(z_cam)
        & (z_cam > camera.near_clip)
    )
    u = np.zeros(points.shape[0], dtype=np.int32)
    v = np.zeros(points.shape[0], dtype=np.int32)
    if np.any(projectable):
        u[projectable] = np.round(camera_coords[projectable, 0] / z_cam[projectable] * camera.fx + camera.cx).astype(np.int32)
        v[projectable] = np.round(camera_coords[projectable, 1] / z_cam[projectable] * camera.fy + camera.cy).astype(np.int32)
    in_frustum = (
        projectable
        & (u >= 0)
        & (u < camera.width)
        & (v >= 0)
        & (v < camera.height)
    )
    sampled_depth = np.zeros(points.shape[0], dtype=np.float64)
    if np.any(in_frustum):
        sampled_depth[in_frustum] = camera.depth[v[in_frustum], u[in_frustum]]
    return in_frustum, z_cam, sampled_depth


def voxel_centers_from_mask(grid_info: GridInfo, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return indices, np.zeros((0, 3), dtype=np.float64)
    centers = np.column_stack(
        [
            grid_info.x_centers[indices[:, 0]],
            grid_info.y_centers[indices[:, 1]],
            grid_info.z_centers[indices[:, 2]],
        ]
    )
    return indices, centers


def visibility_counts_for_mask(
    grid_info: GridInfo,
    target_mask: np.ndarray,
    cameras: list[CameraData],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    indices, centers = voxel_centers_from_mask(grid_info, target_mask)
    visible_counts = np.zeros(centers.shape[0], dtype=np.uint8)
    visible_matrix = np.zeros((centers.shape[0], len(cameras)), dtype=bool)
    if centers.shape[0] == 0:
        return visible_counts, visible_matrix, 0.0, 0

    for camera_index, camera in enumerate(cameras):
        in_frustum, z_cam, sampled_depth = project_points_to_camera(camera, centers)
        valid_surface = in_frustum & valid_depth_mask(sampled_depth, camera.near_clip, camera.far_clip)
        # this follows the existing occlusion/free-space logic: a gt voxel is visible
        # when it is not deeper than the first measured surface by more than the tolerance.
        visible = valid_surface & (z_cam <= sampled_depth + DEPTH_VISIBILITY_TOLERANCE_M)
        visible_matrix[:, camera_index] = visible
        visible_counts += visible.astype(np.uint8)

    diversity_angles = []
    for point_index in np.where(visible_counts >= 2)[0]:
        visible_camera_indices = np.where(visible_matrix[point_index])[0]
        point = centers[point_index]
        for left_pos in range(len(visible_camera_indices)):
            for right_pos in range(left_pos + 1, len(visible_camera_indices)):
                camera_a = cameras[int(visible_camera_indices[left_pos])]
                camera_b = cameras[int(visible_camera_indices[right_pos])]
                ray_a = camera_a.position - point
                ray_b = camera_b.position - point
                norm_a = np.linalg.norm(ray_a)
                norm_b = np.linalg.norm(ray_b)
                if norm_a <= 1e-9 or norm_b <= 1e-9:
                    continue
                cosine = np.clip(np.dot(ray_a, ray_b) / (norm_a * norm_b), -1.0, 1.0)
                diversity_angles.append(math.degrees(math.acos(cosine)))

    mean_angle = float(np.mean(diversity_angles)) if diversity_angles else 0.0
    return visible_counts, visible_matrix, mean_angle, int(indices.shape[0])


# +++++=====+++++ free-space certainty and occlusion-zone artifacts +++++=====+++++

def free_space_view_counts_for_mask(
    grid_info: GridInfo,
    target_mask: np.ndarray,
    cameras: list[CameraData],
) -> tuple[np.ndarray, np.ndarray]:
    indices, centers = voxel_centers_from_mask(grid_info, target_mask)
    free_counts = np.zeros(centers.shape[0], dtype=np.uint8)
    if centers.shape[0] == 0:
        return indices, free_counts

    for camera in cameras:
        in_frustum, z_cam, sampled_depth = project_points_to_camera(camera, centers)
        depth_can_certify = (
            in_frustum
            & np.isfinite(sampled_depth)
            & (sampled_depth > camera.near_clip)
            & (sampled_depth <= camera.far_clip + FAR_CLIP_EPSILON)
            & (z_cam < camera.far_clip - DEPTH_VISIBILITY_TOLERANCE_M)
        )
        # free space is certified only in front of the first measured surface.
        # far/no-hit depth pixels therefore certify free space up to far clip,
        # but they are never interpreted as occupied geometry.
        certified_free = depth_can_certify & (z_cam <= sampled_depth - DEPTH_VISIBILITY_TOLERANCE_M)
        free_counts += certified_free.astype(np.uint8)

    return indices, free_counts


def free_space_masks_from_counts(
    target_mask: np.ndarray,
    indices: np.ndarray,
    free_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    free_view_counts = np.zeros(target_mask.shape, dtype=np.uint8)
    if indices.size:
        free_view_counts[indices[:, 0], indices[:, 1], indices[:, 2]] = free_counts

    unknown_free_mask = target_mask & (free_view_counts == 0)
    single_view_free_mask = target_mask & (free_view_counts == 1)
    redundant_free_mask = target_mask & (free_view_counts >= 2)
    return free_view_counts, unknown_free_mask, single_view_free_mask, redundant_free_mask


def write_free_space_artifact(
    occlusion_root: Path,
    sample: SampleRef,
    metric_scope: str,
    grid_info: GridInfo,
    roi_bounds: dict[str, tuple[float, float]],
    camera_count: int,
    free_target_mask: np.ndarray,
    free_view_counts: np.ndarray,
    unknown_free_mask: np.ndarray,
    single_view_free_mask: np.ndarray,
    redundant_free_mask: np.ndarray,
    free_metrics: dict,
) -> None:
    artifact_dir = occlusion_root / sample.rig_id / sample.layout
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / f"{sample.sample_name}.npz"
    metadata_path = artifact_dir / f"{sample.sample_name}.json"

    np.savez_compressed(
        npz_path,
        free_target_mask=free_target_mask,
        free_view_counts=free_view_counts,
        unknown_free_mask=unknown_free_mask,
        single_view_free_mask=single_view_free_mask,
        redundant_free_mask=redundant_free_mask,
    )
    metadata = {
        "rig_id": sample.rig_id,
        "layout": sample.layout,
        "sample_name": sample.sample_name,
        "sample_index": sample.sample_index,
        "metric_scope": metric_scope,
        "roi_bounds": {axis: [float(bounds[0]), float(bounds[1])] for axis, bounds in roi_bounds.items()},
        "voxel_size_m": float(grid_info.voxel_size),
        "grid_shape": [int(value) for value in grid_info.shape],
        "num_depth_cameras": int(camera_count),
        "depth_visibility_tolerance_m": float(DEPTH_VISIBILITY_TOLERANCE_M),
        "free_volume_voxels_total": int(free_metrics["free_volume_voxels_total"]),
        "certified_free_volume_fraction": float(free_metrics["certified_free_volume_fraction"]),
        "single_view_certified_free_volume_fraction": float(free_metrics["single_view_certified_free_volume_fraction"]),
        "redundantly_certified_free_volume_fraction": float(free_metrics["redundantly_certified_free_volume_fraction"]),
        "unknown_free_volume_fraction": float(free_metrics["unknown_free_volume_fraction"]),
        "semantics_note": "unknown_free_mask means not certified free by any camera; it is not occupied obstacle geometry.",
    }
    write_json(metadata_path, metadata)


def compute_SUMMARY_free_space_certainty(
    sample: SampleRef,
    grid_info: GridInfo,
    props_grid: np.ndarray,
    robot_grid: np.ndarray,
    table_mask: np.ndarray,
    boundary_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    cameras: list[CameraData],
    metric_scope: str,
    roi_bounds: dict[str, tuple[float, float]],
    occlusion_root: Path | None,
) -> dict:
    # the free-space target excludes scene obstacles, table fixtures, room shell,
    # and the robot's own current body occupancy.
    environment_occupied = props_grid | table_mask | robot_grid | boundary_mask
    free_target_mask = evaluation_mask & ~environment_occupied
    indices, free_counts = free_space_view_counts_for_mask(grid_info, free_target_mask, cameras)
    free_view_counts, unknown_free_mask, single_view_free_mask, redundant_free_mask = free_space_masks_from_counts(
        free_target_mask,
        indices,
        free_counts,
    )

    free_total = int(np.count_nonzero(free_target_mask))
    single_fraction = safe_divide(np.count_nonzero(single_view_free_mask), free_total, empty_value=0.0)
    redundant_fraction = safe_divide(np.count_nonzero(redundant_free_mask), free_total, empty_value=0.0)
    certified_fraction = single_fraction + redundant_fraction
    unknown_fraction = safe_divide(np.count_nonzero(unknown_free_mask), free_total, empty_value=0.0)
    metrics = {
        "free_volume_voxels_total": free_total,
        "certified_free_volume_fraction": certified_fraction,
        "single_view_certified_free_volume_fraction": single_fraction,
        "redundantly_certified_free_volume_fraction": redundant_fraction,
        "unknown_free_volume_fraction": unknown_fraction,
        f"{metric_scope}_certified_free_volume_fraction": certified_fraction,
        f"{metric_scope}_single_view_certified_free_volume_fraction": single_fraction,
        f"{metric_scope}_redundantly_certified_free_volume_fraction": redundant_fraction,
        f"{metric_scope}_unknown_free_volume_fraction": unknown_fraction,
    }

    if occlusion_root is not None:
        write_free_space_artifact(
            occlusion_root=occlusion_root,
            sample=sample,
            metric_scope=metric_scope,
            grid_info=grid_info,
            roi_bounds=roi_bounds,
            camera_count=len(cameras),
            free_target_mask=free_target_mask,
            free_view_counts=free_view_counts,
            unknown_free_mask=unknown_free_mask,
            single_view_free_mask=single_view_free_mask,
            redundant_free_mask=redundant_free_mask,
            free_metrics=metrics,
        )

    return metrics


# +++++=====+++++ distance-tolerant surface metrics +++++=====+++++

def mask_from_visibility_counts(target_mask: np.ndarray, visible_counts: np.ndarray, minimum_views: int) -> np.ndarray:
    indices = np.argwhere(target_mask)
    if indices.shape[0] != visible_counts.shape[0]:
        raise ValueError("visibility count vector does not match target mask")
    mask = np.zeros_like(target_mask, dtype=bool)
    selected = indices[visible_counts >= minimum_views]
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return mask


def maximum_grid_distance_cm(grid_info: GridInfo) -> float:
    extents = np.array(grid_info.shape, dtype=np.float64) * float(grid_info.voxel_size)
    return float(np.linalg.norm(extents) * 100.0)


def nearest_distances_cm(
    grid_info: GridInfo,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    _source_indices, source_centers = voxel_centers_from_mask(grid_info, source_mask)
    if source_centers.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    _target_indices, target_centers = voxel_centers_from_mask(grid_info, target_mask)
    if target_centers.shape[0] == 0:
        return np.full(source_centers.shape[0], maximum_grid_distance_cm(grid_info), dtype=np.float64)

    tree = cKDTree(target_centers)
    distances_m, _nearest_indices = tree.query(source_centers, k=1, workers=-1)
    return np.asarray(distances_m, dtype=np.float64) * 100.0


def fbeta_from_precision_recall(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    return safe_divide((1.0 + beta_squared) * precision * recall, beta_squared * precision + recall, empty_value=0.0)


def compute_distance_surface_metrics(
    grid_info: GridInfo,
    prediction: np.ndarray,
    gt_surface: np.ndarray,
    evaluation_mask: np.ndarray,
    visible_gt_mask: np.ndarray,
    thresholds_cm: list[float],
) -> dict:
    # recall uses only visible gt surface, so hidden backsides are not counted as reconstruction failures.
    prediction_eval = prediction & evaluation_mask
    gt_surface_eval = gt_surface & evaluation_mask
    visible_gt_eval = visible_gt_mask & evaluation_mask

    gt_to_reconstruction = nearest_distances_cm(grid_info, visible_gt_eval, prediction_eval)
    reconstruction_to_gt = nearest_distances_cm(grid_info, prediction_eval, gt_surface_eval)

    metrics = {
        "visible_gt_surface_voxels": int(np.count_nonzero(visible_gt_eval)),
        "distance_gt_surface_voxels": int(np.count_nonzero(gt_surface_eval)),
        "distance_reconstruction_voxels": int(np.count_nonzero(prediction_eval)),
        "gt_to_reconstruction_median_cm": float(np.median(gt_to_reconstruction)) if gt_to_reconstruction.size else 0.0,
        "gt_to_reconstruction_p95_cm": float(np.percentile(gt_to_reconstruction, 95)) if gt_to_reconstruction.size else 0.0,
        "reconstruction_to_gt_median_cm": float(np.median(reconstruction_to_gt)) if reconstruction_to_gt.size else 0.0,
        "reconstruction_to_gt_p95_cm": float(np.percentile(reconstruction_to_gt, 95)) if reconstruction_to_gt.size else 0.0,
    }

    for threshold_cm in thresholds_cm:
        label = threshold_label_cm(threshold_cm)
        recall = safe_divide(
            np.count_nonzero(gt_to_reconstruction <= threshold_cm),
            gt_to_reconstruction.size,
            empty_value=1.0,
        )
        precision = safe_divide(
            np.count_nonzero(reconstruction_to_gt <= threshold_cm),
            reconstruction_to_gt.size,
            empty_value=1.0,
        )
        metrics[f"visible_gt_recall_at_{label}"] = recall
        metrics[f"surface_precision_at_{label}"] = precision
        metrics[f"surface_f1_at_{label}"] = fbeta_from_precision_recall(precision, recall, beta=1.0)
        metrics[f"surface_f2_at_{label}"] = fbeta_from_precision_recall(precision, recall, beta=2.0)

    return metrics


# +++++=====+++++ robert-inspired evaluators copied/adapted from Evaluation.py +++++=====+++++

class VoxelIoUEvaluator:
    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        center_coords: tuple[int, int, int] | None = None,
        sigma_fraction: float = 0.3,
        mask: np.ndarray | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.mask = mask if mask is not None else np.ones(grid_shape, dtype=bool)
        if center_coords is None:
            center_coords = (grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2)

        x, y, z = np.indices(grid_shape)
        sigma_x = max(grid_shape[0] * sigma_fraction, 1e-9)
        sigma_y = max(grid_shape[1] * sigma_fraction, 1e-9)
        sigma_z = max(grid_shape[2] * sigma_fraction, 1e-9)
        dist_squared = (
            ((x - center_coords[0]) / sigma_x) ** 2
            + ((y - center_coords[1]) / sigma_y) ** 2
            + ((z - center_coords[2]) / sigma_z) ** 2
        )
        self.weights = np.exp(-dist_squared / 2.0) * self.mask

    def calculate_iou(self, pred_grid: np.ndarray, gt_grid: np.ndarray, apply_weights: bool = True) -> float:
        pred_grid = pred_grid.astype(bool)
        gt_grid = gt_grid.astype(bool)
        intersection = pred_grid & gt_grid
        union = pred_grid | gt_grid
        if apply_weights:
            weighted_intersection = np.sum(intersection * self.weights)
            weighted_union = np.sum(union * self.weights)
            if weighted_union == 0:
                return 1.0
            return float(weighted_intersection / weighted_union)

        masked_intersection = intersection & self.mask
        masked_union = union & self.mask
        if masked_union.sum() == 0:
            return 1.0
        return float(masked_intersection.sum() / masked_union.sum())


def roi_center_index(grid_info: GridInfo, roi_bounds: dict[str, tuple[float, float]]) -> tuple[int, int, int]:
    center_world = np.array(
        [
            0.5 * (roi_bounds["x"][0] + roi_bounds["x"][1]),
            0.5 * (roi_bounds["y"][0] + roi_bounds["y"][1]),
            0.5 * (roi_bounds["z"][0] + roi_bounds["z"][1]),
        ],
        dtype=np.float64,
    )
    index = np.floor((center_world - grid_info.origin) / grid_info.voxel_size).astype(int)
    index = np.clip(index, 0, np.array(grid_info.shape) - 1)
    return int(index[0]), int(index[1]), int(index[2])


def precision_recall_fbeta(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    tolerance_voxels: int,
    beta: float = 2.0,
) -> dict:
    prediction_masked = prediction & mask
    target_masked = target & mask
    prediction_dilated = dilate_bool(prediction_masked, tolerance_voxels)
    target_dilated = dilate_bool(target_masked, tolerance_voxels)

    true_positive_for_recall = np.count_nonzero(target_masked & prediction_dilated)
    true_positive_for_precision = np.count_nonzero(prediction_masked & target_dilated)
    target_count = int(np.count_nonzero(target_masked))
    prediction_count = int(np.count_nonzero(prediction_masked))
    recall = safe_divide(true_positive_for_recall, target_count, empty_value=1.0)
    precision = safe_divide(true_positive_for_precision, prediction_count, empty_value=1.0)
    beta_squared = beta * beta
    fbeta = safe_divide((1.0 + beta_squared) * precision * recall, beta_squared * precision + recall, empty_value=0.0)
    unmatched_prediction = max(prediction_count - true_positive_for_precision, 0)
    overfill_ratio = safe_divide(unmatched_prediction, target_count, empty_value=0.0)
    return {
        "recall": recall,
        "precision": precision,
        "fbeta": fbeta,
        "false_empty_rate": 1.0 - recall,
        "overfill_ratio": overfill_ratio,
        "target_count": target_count,
        "prediction_count": prediction_count,
    }


def robert_weighted_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    evaluator: VoxelIoUEvaluator,
    tolerance_voxels: int,
) -> dict:
    prediction_masked = prediction & mask
    target_masked = target & mask
    prediction_dilated = dilate_bool(prediction_masked, tolerance_voxels)
    target_dilated = dilate_bool(target_masked, tolerance_voxels)
    missing = target_masked & ~prediction_dilated
    extra = prediction_masked & ~target_dilated
    weighted_target = float(np.sum(target_masked * evaluator.weights))
    weighted_missing = float(np.sum(missing * evaluator.weights))
    weighted_extra = float(np.sum(extra * evaluator.weights))
    loss = safe_divide(
        ROBERT_MISSING_WEIGHT * weighted_missing + ROBERT_EXTRA_WEIGHT * weighted_extra,
        weighted_target + 1e-7,
        empty_value=0.0,
    )
    coverage = 1.0 - safe_divide(weighted_missing, weighted_target, empty_value=0.0)
    overfill = safe_divide(weighted_extra, weighted_target, empty_value=0.0)
    return {
        "robert_loss": loss,
        "robert_coverage": coverage,
        "robert_overfill": overfill,
        "weighted_missing": weighted_missing,
        "weighted_extra": weighted_extra,
    }


def compute_roi_metrics(
    prediction: np.ndarray,
    gt_surface: np.ndarray,
    gt_full: np.ndarray,
    roi_mask_array: np.ndarray,
    evaluator: VoxelIoUEvaluator,
) -> dict:
    surface_scores = precision_recall_fbeta(
        prediction,
        gt_surface,
        roi_mask_array,
        tolerance_voxels=SURFACE_TOLERANCE_VOXELS,
        beta=2.0,
    )
    loss_scores = robert_weighted_loss(
        prediction,
        gt_surface,
        roi_mask_array,
        evaluator,
        tolerance_voxels=SURFACE_TOLERANCE_VOXELS,
    )
    return {
        "strict_roi_surface_recall": surface_scores["recall"],
        "strict_roi_surface_precision": surface_scores["precision"],
        "strict_roi_surface_f2": surface_scores["fbeta"],
        "strict_roi_false_empty_rate": surface_scores["false_empty_rate"],
        "strict_roi_overfill_ratio": surface_scores["overfill_ratio"],
        "strict_roi_surface_iou_weighted": evaluator.calculate_iou(prediction, gt_surface, apply_weights=True),
        "strict_roi_surface_iou_unweighted": evaluator.calculate_iou(prediction, gt_surface, apply_weights=False),
        "strict_roi_filled_iou_weighted": evaluator.calculate_iou(prediction, gt_full, apply_weights=True),
        "strict_roi_filled_iou_unweighted": evaluator.calculate_iou(prediction, gt_full, apply_weights=False),
        "strict_roi_robert_loss": loss_scores["robert_loss"],
        "strict_roi_robert_coverage": loss_scores["robert_coverage"],
        "strict_roi_robert_overfill": loss_scores["robert_overfill"],
        "strict_roi_gt_surface_voxels": surface_scores["target_count"],
        "strict_roi_reconstruction_voxels": surface_scores["prediction_count"],
    }


def compute_whole_room_metrics(
    prediction: np.ndarray,
    gt_surface: np.ndarray,
    whole_room_mask: np.ndarray,
) -> dict:
    surface_scores = precision_recall_fbeta(
        prediction,
        gt_surface,
        whole_room_mask,
        tolerance_voxels=SURFACE_TOLERANCE_VOXELS,
        beta=2.0,
    )
    return {
        "strict_whole_surface_recall": surface_scores["recall"],
        "strict_whole_surface_precision": surface_scores["precision"],
        "strict_whole_surface_f2": surface_scores["fbeta"],
        "strict_whole_false_empty_rate": surface_scores["false_empty_rate"],
        "strict_whole_overfill_ratio": surface_scores["overfill_ratio"],
        "strict_whole_gt_surface_voxels": surface_scores["target_count"],
        "strict_whole_reconstruction_voxels": surface_scores["prediction_count"],
    }


def navigation_visibility_score(
    observable_surface: float,
    redundant_surface: float,
    strong_redundancy: float,
    blind_surface: float,
    certified_free_volume: float,
    redundantly_certified_free_volume: float,
) -> float:
    # the score is stored as percentage points so report tables can read directly.
    return 100.0 * (
        NAVIGATION_SCORE_REDUNDANT_WEIGHT * redundant_surface
        + NAVIGATION_SCORE_STRONG_REDUNDANCY_WEIGHT * strong_redundancy
        + NAVIGATION_SCORE_OBSERVABLE_WEIGHT * observable_surface
        + NAVIGATION_SCORE_NONBLIND_WEIGHT * (1.0 - blind_surface)
        + NAVIGATION_SCORE_CERTIFIED_FREE_WEIGHT * certified_free_volume
        + NAVIGATION_SCORE_REDUNDANT_FREE_WEIGHT * redundantly_certified_free_volume
    )


# +++++=====+++++ per-sample metrics +++++=====+++++

def camera_diagnostic_rows(sample: SampleRef, metadata: dict, cameras: list[CameraData]) -> list[dict]:
    rows = []
    for camera in cameras:
        total_pixels = int(camera.depth.size)
        valid_pixels = int(np.count_nonzero(camera.valid_depth))
        raw_mb = camera.depth_path.stat().st_size / 1_000_000.0
        rows.append(
            {
                "sample_index": sample.sample_index,
                "layout": sample.layout,
                "rig_id": sample.rig_id,
                "camera_index": camera.index,
                "camera_name": camera.name,
                "camera_role": camera.role,
                "mount_wall": camera.mount_wall,
                "valid_depth_fraction": safe_divide(valid_pixels, total_pixels),
                "far_no_hit_fraction": 1.0 - safe_divide(valid_pixels, total_pixels),
                "raw_mb": raw_mb,
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
            }
        )
    return rows


def evaluate_one_sample(
    sample: SampleRef,
    depth_stride: int,
    metric_scope: str,
    distance_thresholds_cm: list[float],
    headline_threshold_cm: float,
    occlusion_root: Path | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    metadata, cameras = load_cameras(sample)
    grid_info, props_grid, robot_grid, _scene_grid, table_fixture_grid = load_grid_bundle(sample.path)
    dimensions = room_dimensions(sample.path, grid_info)
    boundary_mask = room_boundary_mask(grid_info, dimensions)
    table_mask = table_removal_mask(sample.path, grid_info, table_fixture_grid)

    gt_full = props_grid & ~boundary_mask
    gt_surface = surface_band_from_solid(gt_full)
    prediction = reconstruct_depth_surface(cameras, grid_info, depth_stride=depth_stride)
    prediction &= ~robot_grid
    prediction &= ~table_mask
    prediction &= ~boundary_mask

    roi_bounds = core_roi_bounds_for_sample(sample.path)
    if metric_scope == "roi":
        evaluation_mask = roi_mask(grid_info, roi_bounds)
        evaluator = VoxelIoUEvaluator(
            grid_shape=grid_info.shape,
            center_coords=roi_center_index(grid_info, roi_bounds),
            sigma_fraction=0.3,
            mask=evaluation_mask,
        )
        metrics = compute_roi_metrics(prediction, gt_surface, gt_full, evaluation_mask, evaluator)
    else:
        evaluation_mask = np.ones(grid_info.shape, dtype=bool)
        metrics = compute_whole_room_metrics(prediction, gt_surface, evaluation_mask)

    visible_counts, _visible_matrix, mean_angle, visible_target_count = visibility_counts_for_mask(
        grid_info,
        gt_surface & evaluation_mask,
        cameras,
    )
    visible_by_at_least_1_camera = safe_divide(
        np.count_nonzero(visible_counts >= 1),
        visible_target_count,
        empty_value=1.0,
    )
    redundant_coverage_fraction = safe_divide(
        np.count_nonzero(visible_counts >= 2),
        visible_target_count,
        empty_value=1.0,
    )
    visible_by_at_least_3_cameras = safe_divide(
        np.count_nonzero(visible_counts >= 3),
        visible_target_count,
        empty_value=1.0,
    )
    single_view_risk_fraction = safe_divide(
        np.count_nonzero(visible_counts == 1),
        visible_target_count,
        empty_value=0.0,
    )
    blind_surface_fraction = safe_divide(
        np.count_nonzero(visible_counts == 0),
        visible_target_count,
        empty_value=0.0,
    )
    visible_gt_mask = mask_from_visibility_counts(gt_surface & evaluation_mask, visible_counts, minimum_views=1)
    distance_metrics = compute_distance_surface_metrics(
        grid_info=grid_info,
        prediction=prediction,
        gt_surface=gt_surface,
        evaluation_mask=evaluation_mask,
        visible_gt_mask=visible_gt_mask,
        thresholds_cm=distance_thresholds_cm,
    )
    metrics.update(distance_metrics)
    free_space_metrics = compute_SUMMARY_free_space_certainty(
        sample=sample,
        grid_info=grid_info,
        props_grid=props_grid,
        robot_grid=robot_grid,
        table_mask=table_mask,
        boundary_mask=boundary_mask,
        evaluation_mask=evaluation_mask,
        cameras=cameras,
        metric_scope=metric_scope,
        roi_bounds=roi_bounds,
        occlusion_root=occlusion_root,
    )
    metrics.update(free_space_metrics)
    headline_label = threshold_label_cm(headline_threshold_cm)

    metrics.update(
        {
            f"{metric_scope}_visible_1plus": visible_by_at_least_1_camera,
            f"{metric_scope}_visible_2plus": redundant_coverage_fraction,
            f"{metric_scope}_visible_3plus": visible_by_at_least_3_cameras,
            "visible_by_at_least_1_camera": visible_by_at_least_1_camera,
            "visible_by_at_least_2_cameras": redundant_coverage_fraction,
            "visible_by_at_least_3_cameras": visible_by_at_least_3_cameras,
            "redundant_coverage_fraction": redundant_coverage_fraction,
            "single_view_risk_fraction": single_view_risk_fraction,
            "blind_surface_fraction": blind_surface_fraction,
            "observable_surface_fraction": visible_by_at_least_1_camera,
            "redundant_surface_fraction": redundant_coverage_fraction,
            "strong_redundancy_fraction": visible_by_at_least_3_cameras,
            "single_view_surface_fraction": single_view_risk_fraction,
            "strict_missed_surface_fraction": 1.0 - metrics[strict_scope_surface_key(metric_scope, "recall")],
            "strict_extra_surface_fraction": 1.0 - metrics[strict_scope_surface_key(metric_scope, "precision")],
            "navigation_visibility_score": navigation_visibility_score(
                visible_by_at_least_1_camera,
                redundant_coverage_fraction,
                visible_by_at_least_3_cameras,
                blind_surface_fraction,
                free_space_metrics["certified_free_volume_fraction"],
                free_space_metrics["redundantly_certified_free_volume_fraction"],
            ),
            "headline_threshold_cm": float(headline_threshold_cm),
            "mean_view_diversity_deg": mean_angle,
            "metric_scope": metric_scope,
            "num_depth_cameras": int(metadata.get("numDepthCameras", len(cameras))),
            "raw_depth_mb_per_sample": sum(camera.depth_path.stat().st_size for camera in cameras) / 1_000_000.0,
            "reconstruction_voxels_total": int(np.count_nonzero(prediction)),
            "gt_props_voxels_total": int(np.count_nonzero(gt_full)),
            "gt_surface_voxels_total": int(np.count_nonzero(gt_surface)),
        }
    )

    if metric_scope == "roi":
        metrics.update(
            {
                "roi_x_min": roi_bounds["x"][0],
                "roi_x_max": roi_bounds["x"][1],
                "roi_y_min": roi_bounds["y"][0],
                "roi_y_max": roi_bounds["y"][1],
                "roi_z_min": roi_bounds["z"][0],
                "roi_z_max": roi_bounds["z"][1],
            }
        )

    metrics_row = {
        "sample_index": sample.sample_index,
        "sample_name": sample.sample_name,
        "layout": sample.layout,
        "rig_id": sample.rig_id,
        **metrics,
    }

    semantic_rows = semantic_metrics_for_sample(sample, grid_info, prediction, gt_surface, cameras)
    camera_rows = camera_diagnostic_rows(sample, metadata, cameras)
    return metrics_row, semantic_rows, camera_rows


def semantic_metrics_for_sample(
    sample: SampleRef,
    grid_info: GridInfo,
    prediction: np.ndarray,
    gt_surface: np.ndarray,
    cameras: list[CameraData],
) -> list[dict]:
    rows = []
    for item in prop_bounds(sample.path):
        semantic_id = str(item["semantic_id"])
        if semantic_id not in SEMANTIC_PRIORITIES:
            continue
        object_mask = bounds_mask(grid_info, [item], margin_m=0.05)
        target = gt_surface & object_mask
        pred_region = prediction & object_mask
        scores = precision_recall_fbeta(
            pred_region,
            target,
            object_mask,
            tolerance_voxels=SURFACE_TOLERANCE_VOXELS,
            beta=2.0,
        )
        visible_counts, _visible_matrix, mean_angle, target_count = visibility_counts_for_mask(
            grid_info,
            target,
            cameras,
        )
        visible_1plus = safe_divide(np.count_nonzero(visible_counts >= 1), target_count, empty_value=1.0)
        visible_2plus = safe_divide(np.count_nonzero(visible_counts >= 2), target_count, empty_value=1.0)
        visible_3plus = safe_divide(np.count_nonzero(visible_counts >= 3), target_count, empty_value=1.0)
        presence_success = scores["recall"] >= 0.05 if target_count > 0 else bool(np.any(pred_region))
        rows.append(
            {
                "sample_index": sample.sample_index,
                "layout": sample.layout,
                "rig_id": sample.rig_id,
                "semantic_id": semantic_id,
                "priority": SEMANTIC_PRIORITIES[semantic_id],
                "object_id": item["object_id"],
                "semantic_zone": item["semantic_zone"],
                "object_region_recall": scores["recall"],
                "object_region_precision": scores["precision"],
                "object_region_f2": scores["fbeta"],
                "object_presence_success": int(presence_success),
                "object_target_surface_voxels": scores["target_count"],
                "object_prediction_voxels": scores["prediction_count"],
                "object_visible_surface_recall": visible_1plus,
                "object_visible_1plus": visible_1plus,
                "object_visible_2plus": visible_2plus,
                "object_visible_3plus": visible_3plus,
                "object_mean_view_diversity_deg": mean_angle,
            }
        )
    return rows


# +++++=====+++++ integrity audit +++++=====+++++

def build_SUMMARY_integrity_audit(samples: list[SampleRef], samples_by_index: dict[int, dict[str, SampleRef]]) -> tuple[dict, str]:
    counts_by_rig_layout: dict[tuple[str, str], int] = {}
    camera_count_mismatches = []
    missing_pairs = []
    hash_mismatch_files = []
    expected_files = [
        "scene_objects.json",
        "robot_state.json",
        "voxel_props_occupancy.raw",
        "voxel_robot_occupancy.raw",
    ]

    for sample in samples:
        counts_by_rig_layout[(sample.rig_id, sample.layout)] = counts_by_rig_layout.get((sample.rig_id, sample.layout), 0) + 1
        metadata = load_json(sample.path / "depth_metadata.json")
        expected_camera_count = EXPECTED_CAMERA_COUNTS.get(sample.rig_id)
        actual_camera_count = int(metadata.get("numDepthCameras", -1))
        if expected_camera_count is not None and actual_camera_count != expected_camera_count:
            camera_count_mismatches.append(
                {
                    "sample_index": sample.sample_index,
                    "rig_id": sample.rig_id,
                    "expected": expected_camera_count,
                    "actual": actual_camera_count,
                }
            )

    for sample_index, rig_map in samples_by_index.items():
        missing = [rig for rig in RIG_ORDER if rig not in rig_map]
        if missing:
            missing_pairs.append({"sample_index": sample_index, "missing_rigs": ",".join(missing)})
        for filename in expected_files:
            hashes = {}
            for rig in RIG_ORDER:
                sample = rig_map.get(rig)
                if sample is None:
                    continue
                file_path = sample.path / filename
                hashes[rig] = file_sha256(file_path) if file_path.exists() else "missing"
            if hashes and len(set(hashes.values())) > 1:
                hash_mismatch_files.append(
                    {
                        "sample_index": sample_index,
                        "filename": filename,
                        "hashes": hashes,
                    }
                )

    passed = (
        len(samples) == 160
        and len(samples_by_index) == 40
        and not missing_pairs
        and not camera_count_mismatches
        and not hash_mismatch_files
    )
    audit = {
        "passed": passed,
        "sample_folder_count": len(samples),
        "matched_scene_index_count": len(samples_by_index),
        "counts_by_rig_layout": {f"{rig}/{layout}": count for (rig, layout), count in sorted(counts_by_rig_layout.items())},
        "camera_count_mismatches": camera_count_mismatches,
        "missing_pairs": missing_pairs,
        "hash_mismatch_files": hash_mismatch_files,
    }

    rows = []
    for rig in RIG_ORDER:
        for layout in LAYOUT_ORDER:
            rows.append([rig, layout, counts_by_rig_layout.get((rig, layout), 0)])

    content = "# Integrity audit\n\n"
    content += f"- passed: `{passed}`\n"
    content += f"- sample folders: `{len(samples)}`\n"
    content += f"- matched scene indices: `{len(samples_by_index)}`\n"
    content += f"- camera count mismatches: `{len(camera_count_mismatches)}`\n"
    content += f"- missing rig/sample pairs: `{len(missing_pairs)}`\n"
    content += f"- cross-rig scene hash mismatches: `{len(hash_mismatch_files)}`\n\n"
    content += "## Counts by rig and layout\n\n"
    content += markdown_table(["rig", "layout", "samples"], rows)
    if camera_count_mismatches:
        content += "\n## Camera count mismatches\n\n"
        content += markdown_table(
            ["sample_index", "rig_id", "expected", "actual"],
            [[row["sample_index"], row["rig_id"], row["expected"], row["actual"]] for row in camera_count_mismatches],
        )
    if missing_pairs:
        content += "\n## Missing paired samples\n\n"
        content += markdown_table(
            ["sample_index", "missing_rigs"],
            [[row["sample_index"], row["missing_rigs"]] for row in missing_pairs],
        )
    if hash_mismatch_files:
        content += "\n## Hash mismatches\n\n"
        content += markdown_table(
            ["sample_index", "filename"],
            [[row["sample_index"], row["filename"]] for row in hash_mismatch_files],
        )
    return audit, content


# +++++=====+++++ summaries and paired statistics +++++=====+++++

def group_rows(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_values(values: list[float]) -> dict:
    array = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
    }


def stakeholder_interpretation(rig: str) -> str:
    if rig == RIG_5CAM:
        return "Upper-bound robustness; strongest redundancy at highest data burden."
    if rig == RIG_4CAM_ASYM:
        return "Practical recommendation; large robustness gain over 3Cam for one added camera."
    if rig == RIG_3CAM:
        return "Cost-minimal option; lower data burden but high single-view fragility."
    return "Baseline reference; useful context for the original four-camera geometry."


def build_SUMMARY_camera_rig_visibility(metrics_rows: list[dict], metric_scope: str) -> tuple[list[dict], str]:
    csv_rows = []
    markdown_rows = []
    scope_label = scope_display_name(metric_scope)
    for rig in RIG_ORDER:
        rig_rows = [row for row in metrics_rows if row["rig_id"] == rig]
        if not rig_rows:
            continue

        def median_metric(metric_name: str) -> float:
            return summarize_values([row[metric_name] for row in rig_rows])["median"]

        cameras = int(round(median_metric("num_depth_cameras")))
        observable = median_metric("observable_surface_fraction")
        redundant = median_metric("redundant_surface_fraction")
        strong = median_metric("strong_redundancy_fraction")
        single_view = median_metric("single_view_surface_fraction")
        blind = median_metric("blind_surface_fraction")
        raw_mb = median_metric("raw_depth_mb_per_sample")
        score = median_metric("navigation_visibility_score")
        interpretation = stakeholder_interpretation(rig)

        csv_row = {
            "Rig": rig,
            "Cameras": cameras,
            "Raw Depth MB/Sample": raw_mb,
            "Observable Surface (%)": observable * 100.0,
            "Redundant Surface (%)": redundant * 100.0,
            "Strong Redundancy (%)": strong * 100.0,
            "Single-View Surface (%)": single_view * 100.0,
            "Blind Surface (%)": blind * 100.0,
            "Navigation Visibility Score (%)": score,
            "Stakeholder Interpretation": interpretation,
        }
        csv_rows.append(csv_row)
        markdown_rows.append(
            [
                rig,
                cameras,
                format_float(csv_row["Raw Depth MB/Sample"], 1),
                format_float(csv_row["Observable Surface (%)"], 1),
                format_float(csv_row["Redundant Surface (%)"], 1),
                format_float(csv_row["Strong Redundancy (%)"], 1),
                format_float(csv_row["Single-View Surface (%)"], 1),
                format_float(csv_row["Blind Surface (%)"], 1),
                format_float(csv_row["Navigation Visibility Score (%)"], 1),
                interpretation,
            ]
        )

    content = "# Camera Rig Visibility Summary\n\n"
    content += f"Values are medians across matched scenes. These {scope_label}-based headline metrics evaluate camera-rig support for autonomous robot navigation through surface observability, redundancy, and free-space certainty, not reconstruction overlap.\n\n"
    content += markdown_table(
        [
            "Rig",
            "Cameras",
            "Raw Depth MB/Sample",
            "Observable Surface (%)",
            "Redundant Surface (%)",
            "Strong Redundancy (%)",
            "Single-View Surface (%)",
            "Blind Surface (%)",
            "Navigation Visibility Score (%)",
            "Stakeholder Interpretation",
        ],
        markdown_rows,
    )
    return csv_rows, content


def build_SUMMARY_diagnostic_reconstruction(
    metrics_rows: list[dict],
    metric_scope: str,
    headline_threshold_cm: float,
) -> tuple[list[dict], str]:
    label = threshold_label_cm(headline_threshold_cm)
    f1_key = f"surface_f1_at_{label}"
    recall_key = f"visible_gt_recall_at_{label}"
    precision_key = f"surface_precision_at_{label}"
    strict_recall_key = strict_scope_surface_key(metric_scope, "recall")
    strict_precision_key = strict_scope_surface_key(metric_scope, "precision")
    strict_f2_key = strict_scope_surface_key(metric_scope, "f2")
    csv_rows = []
    markdown_rows = []
    for rig in RIG_ORDER:
        rig_rows = [row for row in metrics_rows if row["rig_id"] == rig]
        if not rig_rows:
            continue
        row = {
            "Rig": rig,
            f"F-Score@{label} (%)": summarize_values([item[f1_key] for item in rig_rows])["median"] * 100.0,
            f"Visible-GT Recall@{label} (%)": summarize_values([item[recall_key] for item in rig_rows])["median"] * 100.0,
            f"Surface Precision@{label} (%)": summarize_values([item[precision_key] for item in rig_rows])["median"] * 100.0,
            "Median GT Error (cm)": summarize_values([item["gt_to_reconstruction_median_cm"] for item in rig_rows])["median"],
            "P95 GT Error (cm)": summarize_values([item["gt_to_reconstruction_p95_cm"] for item in rig_rows])["median"],
            "Strict Voxel Recall (%)": summarize_values([item[strict_recall_key] for item in rig_rows])["median"] * 100.0,
            "Strict Voxel Precision (%)": summarize_values([item[strict_precision_key] for item in rig_rows])["median"] * 100.0,
            "Strict Voxel F2 (%)": summarize_values([item[strict_f2_key] for item in rig_rows])["median"] * 100.0,
        }
        csv_rows.append(row)
        markdown_rows.append(
            [
                rig,
                format_float(row[f"F-Score@{label} (%)"], 1),
                format_float(row[f"Visible-GT Recall@{label} (%)"], 1),
                format_float(row[f"Surface Precision@{label} (%)"], 1),
                format_float(row["Median GT Error (cm)"], 1),
                format_float(row["P95 GT Error (cm)"], 1),
                format_float(row["Strict Voxel Recall (%)"], 1),
                format_float(row["Strict Voxel Precision (%)"], 1),
                format_float(row["Strict Voxel F2 (%)"], 1),
            ]
        )

    content = "# Diagnostic Reconstruction Metrics\n\n"
    content += "These metrics are retained for alignment and traceability. They are not headline performance metrics because depth-derived first-hit surfaces and Unity occupancy-derived surface bands are different geometric representations.\n\n"
    content += markdown_table(
        [
            "Rig",
            f"F-Score@{label} (%)",
            f"Visible-GT Recall@{label} (%)",
            f"Surface Precision@{label} (%)",
            "Median GT Error (cm)",
            "P95 GT Error (cm)",
            "Strict Voxel Recall (%)",
            "Strict Voxel Precision (%)",
            "Strict Voxel F2 (%)",
        ],
        markdown_rows,
    )
    return csv_rows, content


def build_layout_summary(metrics_rows: list[dict], metric_scope: str) -> tuple[list[dict], str]:
    csv_rows = []
    markdown_rows = []
    grouped = group_rows(metrics_rows, ["layout", "rig_id"])
    scope_label = scope_display_name(metric_scope)
    for layout in LAYOUT_ORDER:
        for rig in RIG_ORDER:
            group = grouped.get((layout, rig), [])
            if not group:
                continue
            row = {
                "Layout": LAYOUT_DISPLAY_NAMES.get(layout, layout),
                "Layout Id": layout,
                "Rig": rig,
                "Samples": len(group),
                "Navigation Visibility Score (%)": summarize_values([item["navigation_visibility_score"] for item in group])["median"],
                "Observable Surface (%)": summarize_values([item["observable_surface_fraction"] for item in group])["median"] * 100.0,
                "Redundant Surface (%)": summarize_values([item["redundant_surface_fraction"] for item in group])["median"] * 100.0,
                "Strong Redundancy (%)": summarize_values([item["strong_redundancy_fraction"] for item in group])["median"] * 100.0,
                "Blind Surface (%)": summarize_values([item["blind_surface_fraction"] for item in group])["median"] * 100.0,
            }
            csv_rows.append(row)
            markdown_rows.append(
                [
                    row["Layout"],
                    rig,
                    row["Samples"],
                    format_float(row["Navigation Visibility Score (%)"], 1),
                    format_float(row["Observable Surface (%)"], 1),
                    format_float(row["Redundant Surface (%)"], 1),
                    format_float(row["Strong Redundancy (%)"], 1),
                    format_float(row["Blind Surface (%)"], 1),
                ]
            )
    content = "# Layout Robustness Summary\n\n"
    content += f"Layout-specific medians use the active metric scope: `{metric_scope}`. The score is based on observability and redundancy, not reconstruction overlap.\n\n"
    content += markdown_table(
        ["Layout", "Rig", "Samples", f"{scope_label} Navigation Visibility Score (%)", "Observable Surface (%)", "Redundant Surface (%)", "Strong Redundancy (%)", "Blind Surface (%)"],
        markdown_rows,
    )
    return csv_rows, content


def build_SUMMARY_robot_relevant_object_visibility(semantic_rows: list[dict]) -> tuple[list[dict], str]:
    csv_rows = []
    markdown_rows = []
    grouped = group_rows(semantic_rows, ["semantic_id", "rig_id"])
    for semantic_id in ROBOT_RELEVANT_SEMANTICS:
        for rig in RIG_ORDER:
            group = grouped.get((semantic_id, rig), [])
            if not group:
                continue
            visible_1 = summarize_values([row["object_visible_surface_recall"] for row in group])
            visible_2 = summarize_values([row["object_visible_2plus"] for row in group])
            visible_3 = summarize_values([row["object_visible_3plus"] for row in group])
            row = {
                "Object": SEMANTIC_DISPLAY_NAMES.get(semantic_id, semantic_id),
                "Semantic Id": semantic_id,
                "Object Group": ROBOT_RELEVANT_GROUPS.get(semantic_id, "Robot-Relevant Object"),
                "Rig": rig,
                "Samples": visible_1["n"],
                "Visible Surface (%)": visible_1["median"] * 100.0,
                "Redundant Object Surface (%)": visible_2["median"] * 100.0,
                "Strongly Redundant Object Surface (%)": visible_3["median"] * 100.0,
            }
            csv_rows.append(row)
            markdown_rows.append(
                [
                    row["Object"],
                    row["Object Group"],
                    rig,
                    row["Samples"],
                    format_float(row["Visible Surface (%)"], 1),
                    format_float(row["Redundant Object Surface (%)"], 1),
                    format_float(row["Strongly Redundant Object Surface (%)"], 1),
                ]
            )
    content = "# Robot-Relevant Object Visibility\n\n"
    content += "These values use object AABB regions and aggregate prop voxels. Patient and ceiling lamp are intentionally excluded from this stakeholder-facing table because autonomous robot movement depends on room obstacles, staff, and equipment rather than patient-surface reconstruction.\n\n"
    content += markdown_table(
        ["Object", "Object Group", "Rig", "Samples", "Visible Surface (%)", "Redundant Object Surface (%)", "Strongly Redundant Object Surface (%)"],
        markdown_rows,
    )
    return csv_rows, content


def build_SUMMARY_camera_rig_tradeoff(metrics_rows: list[dict]) -> tuple[list[dict], str]:
    def median_for(rig: str, key: str) -> float:
        rows = [row for row in metrics_rows if row["rig_id"] == rig]
        return summarize_values([row[key] for row in rows])["median"]

    comparisons = [
        ("Redundant Surface Gain", "redundant_surface_fraction", 100.0, "percentage points", "higher"),
        ("Strong Redundancy Gain", "strong_redundancy_fraction", 100.0, "percentage points", "higher"),
        ("Single-View Surface Reduction", "single_view_surface_fraction", -100.0, "percentage points", "lower"),
        ("Blind Surface Reduction", "blind_surface_fraction", -100.0, "percentage points", "lower"),
        ("Not-Proven-Free Volume Reduction", "unknown_free_volume_fraction", -100.0, "percentage points", "lower"),
        ("Added Raw Depth Data", "raw_depth_mb_per_sample", 1.0, "MB/sample", "lower cost"),
        ("Added Cameras", "num_depth_cameras", 1.0, "camera", "lower cost"),
    ]
    csv_rows = []
    markdown_rows = []
    for label, metric, scale, unit, direction in comparisons:
        three_value = median_for(RIG_3CAM, metric)
        four_value = median_for(RIG_4CAM_ASYM, metric)
        if metric in {"single_view_surface_fraction", "blind_surface_fraction", "unknown_free_volume_fraction"}:
            change = (three_value - four_value) * abs(scale)
        else:
            change = (four_value - three_value) * scale
        row = {
            "Comparison": "3Cam To 4CamAsym",
            "Metric": label,
            "3Cam Value": three_value * (100.0 if metric.endswith("_fraction") else 1.0),
            "4CamAsym Value": four_value * (100.0 if metric.endswith("_fraction") else 1.0),
            "Change": change,
            "Unit": unit,
            "Preferred Direction": direction,
        }
        csv_rows.append(row)
        markdown_rows.append(
            [
                label,
                format_float(row["3Cam Value"], 1),
                format_float(row["4CamAsym Value"], 1),
                format_float(change, 1),
                unit,
            ]
        )

    content = "# 3Cam To 4CamAsym Trade-Off\n\n"
    content += "This is the main head-to-head stakeholder comparison: one extra camera and additional depth data are exchanged for more redundant observability and less single-view/blind surface.\n\n"
    content += markdown_table(["Metric", "3Cam", "4CamAsym", "Change", "Unit"], markdown_rows)
    return csv_rows, content


def build_free_space_certainty_summary(metrics_rows: list[dict], metric_scope: str) -> tuple[list[dict], str]:
    csv_rows = []
    markdown_rows = []
    scope_label = scope_display_name(metric_scope)
    for rig in RIG_ORDER:
        rig_rows = [row for row in metrics_rows if row["rig_id"] == rig]
        if not rig_rows:
            continue
        row = {
            "Rig": rig,
            "Samples": len(rig_rows),
            "Free Volume Voxels (Median)": summarize_values([item["free_volume_voxels_total"] for item in rig_rows])["median"],
            "Certified Free Volume (%)": summarize_values([item["certified_free_volume_fraction"] for item in rig_rows])["median"] * 100.0,
            "Single-View Certified Free Volume (%)": summarize_values([item["single_view_certified_free_volume_fraction"] for item in rig_rows])["median"] * 100.0,
            "Redundantly Certified Free Volume (%)": summarize_values([item["redundantly_certified_free_volume_fraction"] for item in rig_rows])["median"] * 100.0,
            "Unknown Free Volume (%)": summarize_values([item["unknown_free_volume_fraction"] for item in rig_rows])["median"] * 100.0,
        }
        csv_rows.append(row)
        markdown_rows.append(
            [
                rig,
                row["Samples"],
                format_float(row["Free Volume Voxels (Median)"], 0),
                format_float(row["Certified Free Volume (%)"], 1),
                format_float(row["Single-View Certified Free Volume (%)"], 1),
                format_float(row["Redundantly Certified Free Volume (%)"], 1),
                format_float(row["Unknown Free Volume (%)"], 1),
            ]
        )

    content = "# Free-Space Certainty Metrics\n\n"
    content += f"Values are medians across matched scenes using the active metric scope: `{metric_scope}`. These metrics describe whether GT-empty {scope_label} voxels are certified free by the depth cameras. Unknown free volume means not certified free; it is not treated as occupied geometry.\n\n"
    content += markdown_table(
        [
            "Rig",
            "Samples",
            "Free Volume Voxels (Median)",
            "Certified Free Volume (%)",
            "Single-View Certified Free Volume (%)",
            "Redundantly Certified Free Volume (%)",
            "Unknown Free Volume (%)",
        ],
        markdown_rows,
    )
    return csv_rows, content


# +++++=====+++++ figures +++++=====+++++

def configure_matplotlib(plot_style: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_style.is_file():
        plt.style.use(str(plot_style))
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.6,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "grid.alpha": 0.35,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.facecolor": "white",
        }
    )
    return plt


def rig_color(rig: str) -> str:
    return {
        RIG_4CAM_CLASSIC: "#000000",
        RIG_3CAM: "#0072B2",
        RIG_4CAM_ASYM: "#B00020",
        RIG_5CAM: "#CC79A7",
    }.get(rig, "#E69F00")


def save_figure(fig, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf", facecolor="white", edgecolor="none")
    fig.savefig(figures_dir / f"{name}.png", dpi=600, facecolor="white", edgecolor="none")


def remove_stale_numbered_figures(figures_dir: Path) -> None:
    for path in figures_dir.glob("fig_*.pdf"):
        path.unlink()
    for path in figures_dir.glob("fig_*.png"):
        path.unlink()
    for path in figures_dir.glob("*.pdf"):
        path.unlink()
    for path in figures_dir.glob("*.png"):
        path.unlink()


def build_figures(
    results_dir: Path,
    samples_by_index: dict[int, dict[str, SampleRef]],
    metrics_rows: list[dict],
    semantic_rows: list[dict],
    depth_stride: int,
    plot_style: Path,
    metric_scope: str,
    headline_threshold_cm: float,
) -> None:
    plt = configure_matplotlib(plot_style)
    figures_dir = results_dir / "figures"
    remove_stale_numbered_figures(figures_dir)

    plot_pipeline_and_metric_story(plt, figures_dir)
    plot_rig_floorplan(plt, figures_dir, samples_by_index)
    plot_visibility_risk_horizontal(plt, figures_dir, metrics_rows, metric_scope)
    plot_free_space_certainty_horizontal(plt, figures_dir, metrics_rows, metric_scope)
    plot_SUMMARY_camera_rig_tradeoff(plt, figures_dir, metrics_rows)
    plot_navigation_score_by_rig(plt, figures_dir, metrics_rows)
    plot_redundant_and_blind_surface_by_rig(plt, figures_dir, metrics_rows, metric_scope)
    plot_SUMMARY_layout_robustness_horizontal_heatmap(plt, figures_dir, metrics_rows, metric_scope)
    plot_SUMMARY_robot_relevant_object_visibility(plt, figures_dir, semantic_rows)
    plt.close("all")


def rig_median(metrics_rows: list[dict], rig: str, metric_name: str) -> float:
    rows = [row for row in metrics_rows if row["rig_id"] == rig]
    return summarize_values([row[metric_name] for row in rows])["median"] if rows else np.nan


def plot_pipeline_and_metric_story(plt, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.axis("off")

    stages = [
        ("Unity Paired\nSimulation", "40 matched room scenes\n4 rig exports per scene\nsame robot, props, layout"),
        ("Python Visibility\nEvaluation", "project ROI surface\ncount visibility risk\nkeep reconstruction diagnostic"),
        ("MuJoCo\nHandoff", "robot state and scene geometry\navailable for later\nplanning benchmarks"),
    ]
    x_positions = [0.06, 0.39, 0.72]
    box_width = 0.23
    box_height = 0.46
    for index, ((heading, body), x_position) in enumerate(zip(stages, x_positions)):
        rect = plt.Rectangle(
            (x_position, 0.38),
            box_width,
            box_height,
            facecolor="#F7F7F7",
            edgecolor=rig_color(RIG_4CAM_ASYM) if index == 1 else "#000000",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x_position + box_width * 0.5, 0.75, heading, ha="center", va="center", fontsize=7.7, fontweight="bold", linespacing=1.15)
        ax.text(x_position + box_width * 0.5, 0.55, body, ha="center", va="center", fontsize=6.3, linespacing=1.28)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.035, 0.61),
                xytext=(x_position + box_width + 0.035, 0.61),
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#000000"},
            )
    ax.text(
        0.50,
        0.18,
        "Camera-rig metrics evaluate camera layouts by robust observability: blind surface is unavailable, single-view surface is fragile, and redundant surface supports autonomous navigation.",
        ha="center",
        va="center",
        fontsize=7.6,
        wrap=True,
    )
    save_figure(fig, figures_dir, "pipeline_and_metric_story")


def plot_visibility_risk_horizontal(plt, figures_dir: Path, metrics_rows: list[dict], metric_scope: str) -> None:
    blind_values = []
    single_values = []
    redundant_values = []
    for rig in RIG_ORDER:
        blind_values.append(rig_median(metrics_rows, rig, "blind_surface_fraction") * 100.0)
        single_values.append(rig_median(metrics_rows, rig, "single_view_surface_fraction") * 100.0)
        redundant_values.append(rig_median(metrics_rows, rig, "redundant_surface_fraction") * 100.0)

    y = np.arange(len(RIG_ORDER))
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.grid(True, axis="x", alpha=0.35)
    ax.barh(y, blind_values, color="#B00020", edgecolor="#000000", linewidth=0.3, label="Blind Surface")
    ax.barh(y, single_values, left=blind_values, color="#E69F00", edgecolor="#000000", linewidth=0.3, label="Single-View Surface")
    left = np.asarray(blind_values) + np.asarray(single_values)
    ax.barh(y, redundant_values, left=left, color="#0072B2", edgecolor="#000000", linewidth=0.3, label="Redundant Surface")
    ax.set_yticks(y)
    ax.set_yticklabels(RIG_ORDER)
    ax.set_xlabel(f"{scope_display_name(metric_scope)} Surface Partition (%)")
    ax.set_xlim(0.0, 100.0)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    for y_index, value in enumerate(redundant_values):
        ax.text(left[y_index] + value * 0.5, y_index, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="#FFFFFF")
    save_figure(fig, figures_dir, "visibility_risk_horizontal")


def plot_free_space_certainty_horizontal(plt, figures_dir: Path, metrics_rows: list[dict], metric_scope: str) -> None:
    unknown_values = []
    single_values = []
    redundant_values = []
    for rig in RIG_ORDER:
        unknown_values.append(rig_median(metrics_rows, rig, "unknown_free_volume_fraction") * 100.0)
        single_values.append(rig_median(metrics_rows, rig, "single_view_certified_free_volume_fraction") * 100.0)
        redundant_values.append(rig_median(metrics_rows, rig, "redundantly_certified_free_volume_fraction") * 100.0)

    y = np.arange(len(RIG_ORDER))
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.grid(True, axis="x", alpha=0.35)
    ax.barh(y, unknown_values, color="#B00020", edgecolor="#000000", linewidth=0.3, label="Unknown Free Volume")
    ax.barh(y, single_values, left=unknown_values, color="#E69F00", edgecolor="#000000", linewidth=0.3, label="Single-View Certified Free Volume")
    left = np.asarray(unknown_values) + np.asarray(single_values)
    ax.barh(y, redundant_values, left=left, color="#0072B2", edgecolor="#000000", linewidth=0.3, label="Redundantly Certified Free Volume")
    ax.set_yticks(y)
    ax.set_yticklabels(RIG_ORDER)
    ax.set_xlabel(f"{scope_display_name(metric_scope)} Free-Space Certainty Partition (%)")
    ax.set_xlim(0.0, 100.0)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3)
    for y_index, value in enumerate(redundant_values):
        if value >= 8.0:
            ax.text(left[y_index] + value * 0.5, y_index, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="#FFFFFF")
    save_figure(fig, figures_dir, "free_space_certainty_horizontal")


def plot_SUMMARY_camera_rig_tradeoff(plt, figures_dir: Path, metrics_rows: list[dict]) -> None:
    benefit_labels = [
        "Redundant Surface",
        "Strong Redundancy",
        "Single-View Reduction",
        "Blind Surface Reduction",
        "Not-Proven-Free Volume Reduction",
    ]
    benefit_values = [
        (rig_median(metrics_rows, RIG_4CAM_ASYM, "redundant_surface_fraction") - rig_median(metrics_rows, RIG_3CAM, "redundant_surface_fraction")) * 100.0,
        (rig_median(metrics_rows, RIG_4CAM_ASYM, "strong_redundancy_fraction") - rig_median(metrics_rows, RIG_3CAM, "strong_redundancy_fraction")) * 100.0,
        (rig_median(metrics_rows, RIG_3CAM, "single_view_surface_fraction") - rig_median(metrics_rows, RIG_4CAM_ASYM, "single_view_surface_fraction")) * 100.0,
        (rig_median(metrics_rows, RIG_3CAM, "blind_surface_fraction") - rig_median(metrics_rows, RIG_4CAM_ASYM, "blind_surface_fraction")) * 100.0,
        (rig_median(metrics_rows, RIG_3CAM, "unknown_free_volume_fraction") - rig_median(metrics_rows, RIG_4CAM_ASYM, "unknown_free_volume_fraction")) * 100.0,
    ]

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    y_benefit = np.arange(len(benefit_labels))
    ax.barh(y_benefit, benefit_values, color=rig_color(RIG_4CAM_ASYM), edgecolor="#000000", linewidth=0.4)
    ax.set_yticks(y_benefit)
    ax.set_yticklabels(benefit_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Benefit From 3Cam To 4CamAsym (Percentage Points)")
    ax.set_xlim(0.0, max(benefit_values) * 1.18)
    ax.grid(True, axis="x", alpha=0.35)
    for y_index, value in enumerate(benefit_values):
        ax.text(value + 0.5, y_index, f"+{value:.1f}", va="center", ha="left", fontsize=7)
    save_figure(fig, figures_dir, "SUMMARY_camera_rig_tradeoff")


def plot_navigation_score_by_rig(plt, figures_dir: Path, metrics_rows: list[dict]) -> None:
    values = [rig_median(metrics_rows, rig, "navigation_visibility_score") for rig in RIG_ORDER]
    labels = ["Baseline", "Cost-Minimal", "Practical Recommendation", "Upper Bound"]
    y = np.arange(len(RIG_ORDER))
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.grid(True, axis="x", alpha=0.35)
    ax.barh(y, values, color=[rig_color(rig) for rig in RIG_ORDER], edgecolor="#000000", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{rig}\n{label}" for rig, label in zip(RIG_ORDER, labels)])
    ax.invert_yaxis()
    ax.set_xlabel("Navigation Visibility Score (%)")
    ax.set_xlim(0.0, max(values) * 1.18)
    for y_index, value in enumerate(values):
        ax.text(value + 0.6, y_index, f"{value:.1f}", va="center", ha="left", fontsize=7)
    save_figure(fig, figures_dir, "navigation_score_by_rig")


def plot_redundant_and_blind_surface_by_rig(plt, figures_dir: Path, metrics_rows: list[dict], metric_scope: str) -> None:
    y = np.arange(len(RIG_ORDER))
    redundant = [rig_median(metrics_rows, rig, "redundant_surface_fraction") * 100.0 for rig in RIG_ORDER]
    blind = [rig_median(metrics_rows, rig, "blind_surface_fraction") * 100.0 for rig in RIG_ORDER]

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.barh(y - 0.15, redundant, height=0.28, color="#0072B2", edgecolor="#000000", linewidth=0.3, label="Redundant Surface")
    ax.barh(y + 0.15, blind, height=0.28, color="#B00020", edgecolor="#000000", linewidth=0.3, label="Blind Surface")
    ax.set_yticks(y)
    ax.set_yticklabels(RIG_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel(f"{scope_display_name(metric_scope)} Surface (%)")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_xlim(0.0, max(max(redundant), max(blind)) * 1.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    for y_index, (redundant_value, blind_value) in enumerate(zip(redundant, blind)):
        ax.text(redundant_value + 1.0, y_index - 0.15, f"{redundant_value:.0f}%", va="center", ha="left", fontsize=6)
        ax.text(blind_value + 1.0, y_index + 0.15, f"{blind_value:.0f}%", va="center", ha="left", fontsize=6)
    save_figure(fig, figures_dir, "redundant_and_blind_surface_by_rig")


def plot_SUMMARY_layout_robustness_horizontal_heatmap(plt, figures_dir: Path, metrics_rows: list[dict], metric_scope: str) -> None:
    matrix = np.full((len(LAYOUT_ORDER), len(RIG_ORDER)), np.nan, dtype=np.float64)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, rig in enumerate(RIG_ORDER):
            rows = [row for row in metrics_rows if row["layout"] == layout and row["rig_id"] == rig]
            if rows:
                matrix[i, j] = np.median([float(row["navigation_visibility_score"]) for row in rows])

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    im = ax.imshow(matrix, cmap="RdBu", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(RIG_ORDER)))
    ax.set_xticklabels(RIG_ORDER)
    ax.tick_params(axis="x", labelsize=6.0, rotation=25)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    ax.set_yticks(np.arange(len(LAYOUT_ORDER)))
    ax.set_yticklabels([LAYOUT_DISPLAY_NAMES[item] for item in LAYOUT_ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6, color="#000000")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"{scope_display_name(metric_scope)} Navigation Visibility Score (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save_figure(fig, figures_dir, "SUMMARY_layout_robustness_horizontal_heatmap")


def plot_SUMMARY_robot_relevant_object_visibility(plt, figures_dir: Path, semantic_rows: list[dict]) -> None:
    matrix = np.full((len(ROBOT_RELEVANT_SEMANTICS), len(RIG_ORDER)), np.nan, dtype=np.float64)
    for i, semantic_id in enumerate(ROBOT_RELEVANT_SEMANTICS):
        for j, rig in enumerate(RIG_ORDER):
            values = [float(row["object_visible_surface_recall"]) for row in semantic_rows if row["semantic_id"] == semantic_id and row["rig_id"] == rig]
            if values:
                matrix[i, j] = float(np.median(values)) * 100.0
    fig, ax = plt.subplots(figsize=IEEE_TALL)
    im = ax.imshow(matrix, vmin=0.0, vmax=100.0, cmap="RdBu", aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(RIG_ORDER)))
    ax.set_xticklabels(RIG_ORDER)
    ax.tick_params(axis="x", labelsize=6.0, rotation=25)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    ax.set_yticks(np.arange(len(ROBOT_RELEVANT_SEMANTICS)))
    ax.set_yticklabels([SEMANTIC_DISPLAY_NAMES[item] for item in ROBOT_RELEVANT_SEMANTICS])
    ax.tick_params(axis="y", labelsize=6.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=5.8, color="#000000")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Visible Robot-Relevant Object Surface (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save_figure(fig, figures_dir, "SUMMARY_robot_relevant_object_visibility")


def plot_rig_floorplan(plt, figures_dir: Path, samples_by_index: dict[int, dict[str, SampleRef]]) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Wedge

    first_sample = choose_first_sample(samples_by_index)
    scene = scene_objects(first_sample.path)
    room_item = next(item for item in scene if item.get("objectId") == "room_interior_bounds")
    room_size = room_item["boundsSize"]
    table_items = [object_bounds_record(item) for item in scene if item.get("category") == "table"]

    fig, ax = plt.subplots(figsize=(18.2 * CM, 10.5 * CM))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")

    x_half = float(room_size["x"]) * 0.5
    z_half = float(room_size["z"]) * 0.5
    ax.plot([-x_half, x_half, x_half, -x_half, -x_half], [-z_half, -z_half, z_half, z_half, -z_half], color="#000000", linewidth=0.8, zorder=6)
    for table in table_items:
        center = table["center"]
        size = table["size"]
        ax.add_patch(
            plt.Rectangle(
                (center[0] - size[0] * 0.5, center[2] - size[2] * 0.5),
                size[0],
                size[2],
                edgecolor="#666666",
                facecolor="#CCCCCC",
                alpha=0.25,
                linewidth=0.5,
                zorder=3,
            )
        )
    roi_x = ROI_BOUNDS["x"]
    roi_z = ROI_BOUNDS["z"]
    ax.add_patch(
        plt.Rectangle(
            (roi_x[0], roi_z[0]),
            roi_x[1] - roi_x[0],
            roi_z[1] - roi_z[0],
            edgecolor="#B00020",
            facecolor="none",
            linewidth=1.0,
            linestyle="--",
            zorder=7,
        )
    )

    for rig in RIG_ORDER:
        if rig not in samples_by_index[first_sample.sample_index]:
            continue
        sample = samples_by_index[first_sample.sample_index][rig]
        metadata = load_json(sample.path / "depth_metadata.json")
        for camera in metadata.get("cameras", []):
            position = camera.get("position", {})
            target = camera.get("lookAtTarget", {})
            color = rig_color(rig)
            camera_x = float(position["x"])
            camera_z = float(position["z"])
            target_x = float(target.get("x", 0.0))
            target_z = float(target.get("z", 0.0))
            look_angle = math.degrees(math.atan2(target_z - camera_z, target_x - camera_x))
            horizontal_fov = float(camera.get("horizontalFovDegrees", metadata.get("horizontalFovDegrees", DEFAULT_HORIZONTAL_FOV_DEGREES)))
            ax.add_patch(
                Wedge(
                    (camera_x, camera_z),
                    FLOORPLAN_FOV_RADIUS_M,
                    look_angle - horizontal_fov * 0.5,
                    look_angle + horizontal_fov * 0.5,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.055,
                    linewidth=0.45,
                    zorder=1,
                )
            )
            ax.scatter(camera_x, camera_z, color=color, edgecolor="#000000", linewidth=0.35, s=28, marker="o", zorder=5)
            ax.plot(
                [camera_x, target_x],
                [camera_z, target_z],
                color=color,
                linewidth=0.55,
                alpha=0.72,
                zorder=4,
            )

    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(-3.1, 3.1)
    legend_handles = [
        Line2D([0], [0], color="#000000", linewidth=0.8, label="Room"),
        Patch(facecolor="none", edgecolor="#B00020", linestyle="--", linewidth=1.0, label="ROI"),
    ]
    for rig in RIG_ORDER:
        if rig not in samples_by_index[first_sample.sample_index]:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=rig_color(rig),
                marker="o",
                markersize=4.2,
                linewidth=0.8,
                label=rig,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.095),
        ncol=6,
        fontsize=6.2,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.32,
        columnspacing=0.68,
        borderaxespad=0.0,
    )
    save_figure(fig, figures_dir, "camera_rig_floorplan")


# +++++=====+++++ decisions document +++++=====+++++

def build_SUMMARY_metric_decisions(
    audit: dict,
    args: argparse.Namespace,
    metrics_rows: list[dict],
    semantic_rows: list[dict],
) -> str:
    content = "# Metric Decisions\n\n"
    content += "This file documents the stakeholder-grade metric choices implemented by `source_camera_rig_metrics.py`.\n\n"
    content += "## Scope\n\n"
    content += "- Camera-rig metrics write to `camera_rig_metrics/` inside each run directory.\n"
    content += "- The headline story is camera-rig support for autonomous robot navigation through observability and redundancy, not reconstruction overlap.\n"
    content += "- MuJoCo remains a pipeline handoff and demonstration target until a controlled planning benchmark is defined.\n\n"
    content += "## Dataset Design\n\n"
    content += "- The paired camera-rig design is valid because the audit found "
    content += f"`{audit['matched_scene_index_count']}` matched scene indices and `{audit['sample_folder_count']}` sample folders.\n"
    content += "- Scene, robot, props, and robot voxel files are required to hash-match across rigs for each scene index.\n"
    content += "- The paired design isolates the effect of camera placement because every rig sees the same accepted scene.\n\n"
    content += "## Headline Metrics\n\n"
    content += f"- Active metric scope for this run: `{args.metric_scope}`.\n"
    content += f"- ROI bounds are `X [{ROI_BOUNDS['x'][0]:.1f}, {ROI_BOUNDS['x'][1]:.1f}]`, `Y [{ROI_BOUNDS['y'][0]:.1f}, {ROI_BOUNDS['y'][1]:.2f}]`, and `Z [{ROI_BOUNDS['z'][0]:.1f}, {ROI_BOUNDS['z'][1]:.1f}]` meters.\n"
    content += "- Observable surface is the fraction of ROI GT surface seen by at least one camera.\n"
    content += "- Redundant surface is the fraction seen by at least two cameras.\n"
    content += "- Strong redundancy is the fraction seen by at least three cameras.\n"
    content += "- Single-view surface is visible but fragile because it depends on one camera.\n"
    content += "- Blind surface is not observable by the rig.\n"
    content += f"- Navigation Visibility Score is `100 * ({NAVIGATION_SCORE_REDUNDANT_WEIGHT:.2f} * redundant surface + {NAVIGATION_SCORE_STRONG_REDUNDANCY_WEIGHT:.2f} * strong redundancy + {NAVIGATION_SCORE_OBSERVABLE_WEIGHT:.2f} * observable surface + {NAVIGATION_SCORE_NONBLIND_WEIGHT:.2f} * non-blind surface + {NAVIGATION_SCORE_CERTIFIED_FREE_WEIGHT:.2f} * certified free volume + {NAVIGATION_SCORE_REDUNDANT_FREE_WEIGHT:.2f} * redundantly certified free volume)`.\n"
    content += "- These weights reflect a clinical/corporate safety preference: robust multi-view obstacle evidence first, certified motion-space evidence second, and simple one-camera coverage last.\n\n"
    content += "## Semantic Metrics\n\n"
    content += "- Stakeholder-facing semantic outputs exclude patient and ceiling lamp.\n"
    content += "- Patient visibility is available in the full semantic diagnostic CSV, but it is not a headline metric because the autonomous robot should move around the room rather than over the patient.\n"
    content += "- Object-region visibility uses exported AABB regions and aggregate prop voxels; it is not exact mesh segmentation.\n\n"
    content += "## Free-Space Certainty And Occlusion Artifacts\n\n"
    content += "- Camera-rig metrics recompute free-space certainty from the current depth captures on every non-audit run.\n"
    content += "- Stored artifacts live under `camera_rig_metrics/occlusion_zones/<Rig>/<Layout>/sample_XXXX.npz` with companion JSON metadata.\n"
    content += "- Certified free volume means GT-empty voxels that are in front of the measured depth surface for at least one camera.\n"
    content += "- Unknown free volume means GT-empty voxels that no camera could certify as free; this is conservative unknown space, not occupied obstacle geometry.\n"
    content += "- Far/no-hit depth pixels can certify free space up to far clip, but they are never treated as occupied surfaces.\n\n"
    content += "## Reconstruction Diagnostics\n\n"
    content += "- Distance-tolerant and exact voxel-overlap reconstruction metrics are written to `SUMMARY_diagnostic_reconstruction.*` only.\n"
    content += "- They are representation-alignment diagnostics because the reconstruction is a depth-derived first-hit surface while Unity GT is an occupancy-derived surface band.\n\n"
    content += "## Figure style\n\n"
    content += "- Figures are written as both `.pdf` and `.png` under `camera_rig_metrics/figures`.\n"
    content += f"- Plot style file: `{args.plot_style}`.\n"
    content += "- Figures use IEEE single-column or double-column canvases, no plot titles, grid where useful, and horizontal bar-style layouts where possible.\n"
    content += "- PNG figures are exported at 600 dpi, while PDF figures keep embedded TrueType-compatible fonts for report/presentation use.\n\n"
    content += "## Runtime settings\n\n"
    content += f"- capture root: `{args.capture_root}`\n"
    content += f"- results dir: `{args.results_dir}`\n"
    content += f"- depth stride: `{args.depth_stride}`\n"
    content += f"- metric scope: `{args.metric_scope}`\n"
    content += f"- metrics rows: `{len(metrics_rows)}`\n"
    content += f"- semantic rows: `{len(semantic_rows)}`\n"
    return content


# +++++=====+++++ main orchestration +++++=====+++++

def run(args: argparse.Namespace) -> None:
    capture_root = Path(args.capture_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    plot_style = resolve_plot_style_path(args.plot_style)
    args.plot_style = str(plot_style)
    distance_thresholds_cm = parse_float_list(args.distance_thresholds_cm, DEFAULT_DISTANCE_THRESHOLDS_CM)
    if float(args.headline_threshold_cm) not in distance_thresholds_cm:
        distance_thresholds_cm.append(float(args.headline_threshold_cm))
        distance_thresholds_cm = sorted(set(distance_thresholds_cm))
    args.distance_thresholds_cm = distance_thresholds_cm
    args.headline_threshold_cm = float(args.headline_threshold_cm)
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    requested_indices = parse_sample_indices(args.sample_indices)
    samples = discover_samples(capture_root)
    if requested_indices is not None:
        samples = [sample for sample in samples if sample.sample_index in requested_indices]
    samples_by_index = group_samples_by_index(samples)

    audit, audit_markdown = build_SUMMARY_integrity_audit(samples, samples_by_index)
    write_json(results_dir / "SUMMARY_integrity_audit.json", audit)
    write_markdown(results_dir / "SUMMARY_integrity_audit.md", audit_markdown)

    if args.audit_only:
        decisions = build_SUMMARY_metric_decisions(audit, args, [], [])
        write_markdown(results_dir / "SUMMARY_metric_decisions.md", decisions)
        print(f"audit written to {results_dir / 'SUMMARY_integrity_audit.md'}")
        return

    metrics_rows: list[dict] = []
    semantic_rows: list[dict] = []
    camera_rows: list[dict] = []
    occlusion_root = results_dir / "occlusion_zones"
    if occlusion_root.exists():
        shutil.rmtree(occlusion_root)
    occlusion_root.mkdir(parents=True, exist_ok=True)

    total_samples = len(samples)
    for sample_number, sample in enumerate(samples, start=1):
        print(f"[{sample_number:03d}/{total_samples:03d}] evaluating {sample.relative_path}")
        metrics_row, sample_semantic_rows, sample_camera_rows = evaluate_one_sample(
            sample,
            args.depth_stride,
            args.metric_scope,
            distance_thresholds_cm,
            args.headline_threshold_cm,
            occlusion_root,
        )
        metrics_rows.append(metrics_row)
        semantic_rows.extend(sample_semantic_rows)
        camera_rows.extend(sample_camera_rows)

    validate_metric_invariants(metrics_rows, args.metric_scope, distance_thresholds_cm)
    write_csv(results_dir / "BULK_camera_rig_per_sample.csv", metrics_rows)
    write_csv(results_dir / "BULK_semantic_object_metrics.csv", semantic_rows)
    write_csv(results_dir / "BULK_camera_diagnostics.csv", camera_rows)

    headline_rows, headline_md = build_SUMMARY_camera_rig_visibility(metrics_rows, args.metric_scope)
    tradeoff_rows, tradeoff_md = build_SUMMARY_camera_rig_tradeoff(metrics_rows)
    layout_rows, layout_md = build_layout_summary(metrics_rows, args.metric_scope)
    robot_object_rows, robot_object_md = build_SUMMARY_robot_relevant_object_visibility(semantic_rows)
    diagnostic_rows, diagnostic_md = build_SUMMARY_diagnostic_reconstruction(metrics_rows, args.metric_scope, args.headline_threshold_cm)
    free_space_rows, free_space_md = build_free_space_certainty_summary(metrics_rows, args.metric_scope)

    write_csv(results_dir / "SUMMARY_camera_rig_visibility.csv", headline_rows)
    write_csv(results_dir / "SUMMARY_camera_rig_tradeoff.csv", tradeoff_rows)
    write_csv(results_dir / "SUMMARY_layout_robustness.csv", layout_rows)
    write_csv(results_dir / "SUMMARY_robot_relevant_object_visibility.csv", robot_object_rows)
    write_csv(results_dir / "SUMMARY_diagnostic_reconstruction.csv", diagnostic_rows)
    write_csv(results_dir / "SUMMARY_free_space_certainty.csv", free_space_rows)
    write_markdown(results_dir / "SUMMARY_camera_rig_visibility.md", headline_md)
    write_markdown(results_dir / "SUMMARY_camera_rig_tradeoff.md", tradeoff_md)
    for stale_paired_path in [results_dir / "paired_delta_summary.csv", results_dir / "paired_delta_summary.md"]:
        if stale_paired_path.exists():
            stale_paired_path.unlink()
    write_markdown(results_dir / "SUMMARY_layout_robustness.md", layout_md)
    write_markdown(results_dir / "SUMMARY_robot_relevant_object_visibility.md", robot_object_md)
    write_markdown(results_dir / "SUMMARY_diagnostic_reconstruction.md", diagnostic_md)
    write_markdown(results_dir / "SUMMARY_free_space_certainty.md", free_space_md)

    decisions = build_SUMMARY_metric_decisions(audit, args, metrics_rows, semantic_rows)
    write_markdown(results_dir / "SUMMARY_metric_decisions.md", decisions)

    if not args.skip_figures:
        build_figures(
            results_dir,
            samples_by_index,
            metrics_rows,
            semantic_rows,
            args.depth_stride,
            plot_style,
            args.metric_scope,
            args.headline_threshold_cm,
        )

    print(f"results written to {results_dir}")


def validate_metric_invariants(metrics_rows: list[dict], metric_scope: str, distance_thresholds_cm: list[float]) -> None:
    identity = np.zeros((4, 4, 4), dtype=bool)
    identity[1:3, 1:3, 1:3] = True
    evaluator = VoxelIoUEvaluator(identity.shape)
    if abs(evaluator.calculate_iou(identity, identity) - 1.0) > 1e-12:
        raise AssertionError("robert-style iou identity check failed")

    for row in metrics_rows:
        visible_1 = float(row["visible_by_at_least_1_camera"])
        visible_2 = float(row["redundant_coverage_fraction"])
        visible_3 = float(row["visible_by_at_least_3_cameras"])
        if not (visible_3 <= visible_2 <= visible_1 + 1e-12):
            raise AssertionError(
                f"visibility invariant failed for sample={row['sample_index']} rig={row['rig_id']}: {visible_1}, {visible_2}, {visible_3}"
            )
        missed = float(row["strict_missed_surface_fraction"])
        recall = float(row[strict_scope_surface_key(metric_scope, "recall")])
        extra = float(row["strict_extra_surface_fraction"])
        precision = float(row[strict_scope_surface_key(metric_scope, "precision")])
        single_view = float(row["single_view_risk_fraction"])
        blind = float(row["blind_surface_fraction"])
        observable = float(row["observable_surface_fraction"])
        redundant = float(row["redundant_surface_fraction"])
        strong = float(row["strong_redundancy_fraction"])
        score = float(row["navigation_visibility_score"])
        free_total = int(row["free_volume_voxels_total"])
        certified_free = float(row["certified_free_volume_fraction"])
        single_free = float(row["single_view_certified_free_volume_fraction"])
        redundant_free = float(row["redundantly_certified_free_volume_fraction"])
        unknown_free = float(row["unknown_free_volume_fraction"])
        if abs(missed - (1.0 - recall)) > 1e-12:
            raise AssertionError(f"missed-surface invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if abs(extra - (1.0 - precision)) > 1e-12:
            raise AssertionError(f"extra-surface invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if single_view + visible_2 > visible_1 + 1e-12:
            raise AssertionError(f"single-view/redundant invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if abs(blind - (1.0 - visible_1)) > 1e-12:
            raise AssertionError(f"blind-surface invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if not (strong <= redundant <= observable + 1e-12):
            raise AssertionError(f"navigation visibility invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if abs((blind + single_view + redundant) - 1.0) > 1e-10:
            raise AssertionError(f"surface partition invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        if not (0.0 <= score <= 100.0):
            raise AssertionError(f"navigation visibility score out of range for sample={row['sample_index']} rig={row['rig_id']}: {score}")
        if free_total > 0:
            if abs((unknown_free + single_free + redundant_free) - 1.0) > 1e-10:
                raise AssertionError(f"free-space partition invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
            if abs(certified_free - (single_free + redundant_free)) > 1e-10:
                raise AssertionError(f"certified-free invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
            if abs(unknown_free - (1.0 - certified_free)) > 1e-10:
                raise AssertionError(f"unknown-free invariant failed for sample={row['sample_index']} rig={row['rig_id']}")
        sorted_thresholds = sorted(distance_thresholds_cm)
        for left, right in zip(sorted_thresholds, sorted_thresholds[1:]):
            left_label = threshold_label_cm(left)
            right_label = threshold_label_cm(right)
            if float(row[f"visible_gt_recall_at_{right_label}"]) + 1e-12 < float(row[f"visible_gt_recall_at_{left_label}"]):
                raise AssertionError(f"distance recall monotonicity failed for sample={row['sample_index']} rig={row['rig_id']}")
            if float(row[f"surface_precision_at_{right_label}"]) + 1e-12 < float(row[f"surface_precision_at_{left_label}"]):
                raise AssertionError(f"distance precision monotonicity failed for sample={row['sample_index']} rig={row['rig_id']}")
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise AssertionError(f"non-finite value for {key} in sample={row['sample_index']} rig={row['rig_id']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="camera-rig visibility metrics generator")
    parser.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT), help="path to DepthCaptures or DepthCaptures_demo")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="directory where tables and figures are written")
    parser.add_argument("--audit-only", action="store_true", help="only validate paired dataset integrity")
    parser.add_argument("--sample-indices", default=None, help="comma-separated scene indices to process, for example 0,1,2")
    parser.add_argument("--skip-figures", action="store_true", help="write tables only")
    parser.add_argument("--depth-stride", type=int, default=1, help="depth pixel stride for reconstruction; keep 1 for final results")
    parser.add_argument(
        "--metric-scope",
        choices=["roi", "whole"],
        default="roi",
        help="metric scope for headline tables and figures; default is roi",
    )
    parser.add_argument(
        "--distance-thresholds-cm",
        default=",".join(f"{value:g}" for value in DEFAULT_DISTANCE_THRESHOLDS_CM),
        help="comma-separated distance thresholds in centimeters for tolerant surface scoring",
    )
    parser.add_argument(
        "--headline-threshold-cm",
        type=float,
        default=DEFAULT_HEADLINE_THRESHOLD_CM,
        help="distance threshold in centimeters used only by diagnostic reconstruction metrics",
    )
    parser.add_argument(
        "--plot-style",
        default=None,
        help="matplotlib style file for report figures; default is results/figures/styles/ieee_matlab.mplstyle",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
