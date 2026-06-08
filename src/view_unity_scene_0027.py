"""Open the Unity full-scene occupancy grid for demo sample 0027."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from . import azurion_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "DepthCaptures_demo" / "4CamAsym" / "N2_mirrored_support" / "sample_0027"

PROPS_COLOR = np.array([0.72, 0.74, 0.70], dtype=np.float64)
ROBOT_COLOR = np.array([0.90, 0.22, 0.08], dtype=np.float64)
TABLE_COLOR = np.array([0.34, 0.34, 0.34], dtype=np.float64)
TABLE_WIREFRAME_COLOR = np.array([0.02, 0.02, 0.02], dtype=np.float64)
MIRROR_DISPLAY_Z = True


def _load_grid(sample_path: Path, kind: str) -> tuple[dict, np.ndarray]:
    metadata, raw = azurion_dataset.load_unity_voxels(str(sample_path), kind)
    return metadata, azurion_dataset.reshape_unity_voxel_grid(metadata, raw)


def _unity_grid_origin(metadata: dict) -> np.ndarray:
    origin = metadata.get("origin", {})
    return np.array(
        [
            float(origin.get("x", 0.0)),
            float(origin.get("y", 0.0)),
            float(origin.get("z", 0.0)),
        ],
        dtype=np.float64,
    )


def _voxel_grid_from_mask(mask: np.ndarray, metadata: dict, color: np.ndarray) -> o3d.geometry.VoxelGrid:
    voxel_size = metadata.get("voxelSize", {})
    grid = o3d.geometry.VoxelGrid()
    grid.origin = _unity_grid_origin(metadata)
    grid.voxel_size = float(voxel_size.get("x", 0.05))
    display_mask = mask[:, :, ::-1] if MIRROR_DISPLAY_Z else mask
    for index in np.argwhere(display_mask):
        grid.add_voxel(o3d.geometry.Voxel(index.astype(np.int32), color))
    return grid


def _table_bounds_mask(sample_path: Path, metadata: dict, shape: tuple[int, int, int]) -> np.ndarray:
    origin = _unity_grid_origin(metadata)
    voxel_size = metadata.get("voxelSize", {})
    size = float(voxel_size.get("x", 0.05))
    x_centers = np.arange(shape[0], dtype=np.float64) * size + origin[0] + size * 0.5
    y_centers = np.arange(shape[1], dtype=np.float64) * size + origin[1] + size * 0.5
    z_centers = np.arange(shape[2], dtype=np.float64) * size + origin[2] + size * 0.5

    scene = azurion_dataset.load_scene_objects(str(sample_path))
    mask = np.zeros(shape, dtype=bool)
    for item in scene.get("objects", []):
        if item.get("category") != "table":
            continue
        position = item.get("position", {})
        bounds_size = item.get("boundsSize", {})
        center = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        extent = np.array(
            [
                float(bounds_size.get("x", 0.0)),
                float(bounds_size.get("y", 0.0)),
                float(bounds_size.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        if np.any(extent <= 0.0):
            continue
        half_extent = extent * 0.5
        x_mask = (x_centers >= center[0] - half_extent[0]) & (x_centers <= center[0] + half_extent[0])
        y_mask = (y_centers >= center[1] - half_extent[1]) & (y_centers <= center[1] + half_extent[1])
        z_mask = (z_centers >= center[2] - half_extent[2]) & (z_centers <= center[2] + half_extent[2])
        mask[np.ix_(x_mask, y_mask, z_mask)] = True
    return mask


def _table_wireframes(sample_path: Path) -> list[o3d.geometry.LineSet]:
    scene = azurion_dataset.load_scene_objects(str(sample_path))
    wireframes: list[o3d.geometry.LineSet] = []
    for item in scene.get("objects", []):
        if item.get("category") != "table":
            continue
        position = item.get("position", {})
        size = item.get("boundsSize", {})
        center = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                -float(position.get("z", 0.0)) if MIRROR_DISPLAY_Z else float(position.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        extent = np.array(
            [
                float(size.get("x", 0.0)),
                float(size.get("y", 0.0)),
                float(size.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        if np.any(extent <= 0.0):
            continue
        box = o3d.geometry.AxisAlignedBoundingBox(center - extent * 0.5, center + extent * 0.5)
        wireframe = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        wireframe.paint_uniform_color(TABLE_WIREFRAME_COLOR)
        wireframes.append(wireframe)
    return wireframes


def load_unity_scene_geometries(sample_path: Path = SAMPLE_PATH) -> tuple[list[object], dict[str, int]]:
    scene_metadata, scene_grid = _load_grid(sample_path, "scene")
    _props_metadata, props_grid = _load_grid(sample_path, "props")
    _robot_metadata, robot_grid = _load_grid(sample_path, "robot")

    if scene_grid.shape != props_grid.shape or scene_grid.shape != robot_grid.shape:
        raise ValueError(f"Unity voxel grid shape mismatch under {sample_path}")

    props_mask = scene_grid & props_grid
    robot_mask = scene_grid & robot_grid
    unique_table_mask = scene_grid & ~props_mask & ~robot_mask
    combined = props_mask | robot_mask | unique_table_mask
    if not np.array_equal(combined, scene_grid):
        raise ValueError("raw occupancy layers do not reconstruct the full scene grid")

    expected_occupied = int(scene_metadata.get("occupiedVoxels", -1))
    scene_occupied = int(np.count_nonzero(scene_grid))
    if expected_occupied >= 0 and scene_occupied != expected_occupied:
        raise ValueError(f"scene occupancy has {scene_occupied} voxels, metadata says {expected_occupied}")

    table_bounds_mask = _table_bounds_mask(sample_path, scene_metadata, scene_grid.shape)
    geometries = [
        _voxel_grid_from_mask(table_bounds_mask, scene_metadata, TABLE_COLOR),
        _voxel_grid_from_mask(props_mask, scene_metadata, PROPS_COLOR),
        _voxel_grid_from_mask(robot_mask, scene_metadata, ROBOT_COLOR),
    ]
    geometries.extend(_table_wireframes(sample_path))
    counts = {
        "scene": scene_occupied,
        "props": int(np.count_nonzero(props_mask)),
        "robot": int(np.count_nonzero(robot_mask)),
        "unique_table_and_fixtures": int(np.count_nonzero(unique_table_mask)),
        "table_bounds_guide": int(np.count_nonzero(table_bounds_mask)),
        "table_wireframes": len(geometries) - 3,
    }
    return geometries, counts


def main() -> int:
    geometries, counts = load_unity_scene_geometries()
    print(f"Opening Unity full-scene occupancy: {SAMPLE_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "Voxels: "
        f"scene={counts['scene']}, props={counts['props']}, "
        f"robot={counts['robot']}, unique table/fixtures={counts['unique_table_and_fixtures']}, "
        f"table guide={counts['table_bounds_guide']}"
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Unity full scene occupancy - sample 0027", width=1400, height=900)
    for geometry in geometries:
        vis.add_geometry(geometry, reset_bounding_box=True)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    print("Open3D Unity occupancy viewer running. Close the window to exit.")
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
