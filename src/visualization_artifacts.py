"""Save and load Open3D decomposition viewer artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


_HIDDEN_OCCLUSION_COLORS = np.asarray(
    [
        [0.48, 0.27, 0.11],
        [0.15, 0.25, 0.45],
    ],
    dtype=np.float64,
)


def _sample_view_dir(run_dir: Path, sample_rel_path: str) -> Path:
    return run_dir / "visualizations" / "decomposition" / sample_rel_path


def decomposition_view_path(run_dir: Path, sample_rel_path: str, sample_name: str) -> Path:
    return _sample_view_dir(run_dir, sample_rel_path) / f"{sample_name}_decomposition_view.npz"


def save_decomposition_view(path: Path, geometries: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] = []
    arrays: dict[str, np.ndarray] = {}

    for index, geometry in enumerate(geometries):
        prefix = f"g{index}"
        if isinstance(geometry, o3d.geometry.VoxelGrid):
            voxels = geometry.get_voxels()
            indices = np.asarray([voxel.grid_index for voxel in voxels], dtype=np.int32)
            colors = np.asarray([voxel.color for voxel in voxels], dtype=np.float64)
            arrays[f"{prefix}_indices"] = indices.reshape((-1, 3))
            arrays[f"{prefix}_colors"] = colors.reshape((-1, 3))
            arrays[f"{prefix}_origin"] = np.asarray(geometry.origin, dtype=np.float64)
            arrays[f"{prefix}_voxel_size"] = np.asarray([float(geometry.voxel_size)], dtype=np.float64)
            metadata.append({"kind": "VoxelGrid", "prefix": prefix})
        elif isinstance(geometry, o3d.geometry.LineSet):
            arrays[f"{prefix}_points"] = np.asarray(geometry.points, dtype=np.float64)
            arrays[f"{prefix}_lines"] = np.asarray(geometry.lines, dtype=np.int32)
            arrays[f"{prefix}_colors"] = np.asarray(geometry.colors, dtype=np.float64)
            metadata.append({"kind": "LineSet", "prefix": prefix})
        elif isinstance(geometry, o3d.geometry.OrientedBoundingBox):
            arrays[f"{prefix}_center"] = np.asarray(geometry.center, dtype=np.float64)
            arrays[f"{prefix}_rotation"] = np.asarray(geometry.R, dtype=np.float64)
            arrays[f"{prefix}_extent"] = np.asarray(geometry.extent, dtype=np.float64)
            arrays[f"{prefix}_color"] = np.asarray(geometry.color, dtype=np.float64)
            metadata.append({"kind": "OrientedBoundingBox", "prefix": prefix})

    arrays["metadata"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    return path


def load_decomposition_view(path: Path) -> list[object]:
    geometries: list[object] = []
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata"]))
        for item in metadata:
            prefix = item["prefix"]
            kind = item["kind"]
            if kind == "VoxelGrid":
                grid = o3d.geometry.VoxelGrid()
                grid.origin = archive[f"{prefix}_origin"].astype(np.float64)
                grid.voxel_size = float(archive[f"{prefix}_voxel_size"][0])
                indices = archive[f"{prefix}_indices"].astype(np.int32)
                colors = archive[f"{prefix}_colors"].astype(np.float64)
                for grid_index, color in zip(indices, colors):
                    grid.add_voxel(o3d.geometry.Voxel(grid_index, color))
                geometries.append(grid)
            elif kind == "LineSet":
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(archive[f"{prefix}_points"].astype(np.float64))
                line_set.lines = o3d.utility.Vector2iVector(archive[f"{prefix}_lines"].astype(np.int32))
                colors = archive[f"{prefix}_colors"].astype(np.float64)
                if colors.size:
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(line_set)
            elif kind == "OrientedBoundingBox":
                box = o3d.geometry.OrientedBoundingBox(
                    archive[f"{prefix}_center"].astype(np.float64),
                    archive[f"{prefix}_rotation"].astype(np.float64),
                    archive[f"{prefix}_extent"].astype(np.float64),
                )
                box.color = archive[f"{prefix}_color"].astype(np.float64)
                geometries.append(box)
    return geometries


def _hidden_color_mask(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64).reshape((-1, 3))
    matches = np.isclose(colors[:, None, :], _HIDDEN_OCCLUSION_COLORS[None, :, :], atol=1e-6)
    return np.any(np.all(matches, axis=2), axis=1)


def _filter_voxel_grid(grid: o3d.geometry.VoxelGrid) -> o3d.geometry.VoxelGrid | None:
    voxels = grid.get_voxels()
    if not voxels:
        return grid

    colors = np.asarray([voxel.color for voxel in voxels], dtype=np.float64)
    keep = ~_hidden_color_mask(colors)
    if np.all(keep):
        return grid
    if not np.any(keep):
        return None

    filtered = o3d.geometry.VoxelGrid()
    filtered.origin = np.asarray(grid.origin, dtype=np.float64)
    filtered.voxel_size = float(grid.voxel_size)
    for voxel, keep_voxel in zip(voxels, keep):
        if keep_voxel:
            filtered.add_voxel(o3d.geometry.Voxel(voxel.grid_index, voxel.color))
    return filtered


def _filter_line_set(line_set: o3d.geometry.LineSet) -> o3d.geometry.LineSet | None:
    colors = np.asarray(line_set.colors, dtype=np.float64)
    lines = np.asarray(line_set.lines, dtype=np.int32)
    if colors.size == 0 or len(colors) != len(lines):
        return line_set

    keep = ~_hidden_color_mask(colors)
    if np.all(keep):
        return line_set
    if not np.any(keep):
        return None

    filtered = o3d.geometry.LineSet()
    filtered.points = line_set.points
    filtered.lines = o3d.utility.Vector2iVector(lines[keep])
    filtered.colors = o3d.utility.Vector3dVector(colors[keep])
    return filtered


def hide_occlusion_geometries(geometries: list[object]) -> list[object]:
    filtered: list[object] = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.VoxelGrid):
            visible_geometry = _filter_voxel_grid(geometry)
        elif isinstance(geometry, o3d.geometry.LineSet):
            visible_geometry = _filter_line_set(geometry)
        elif isinstance(geometry, o3d.geometry.OrientedBoundingBox) and _hidden_color_mask(np.asarray([geometry.color])):
            visible_geometry = None
        else:
            visible_geometry = geometry
        if visible_geometry is not None:
            filtered.append(visible_geometry)
    return filtered
