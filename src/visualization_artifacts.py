"""Save and load Open3D decomposition viewer artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


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
