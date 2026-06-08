"""Open3D point-cloud display helpers.

This module only styles point clouds for human inspection. It must not change
pipeline computations or saved artifacts.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def height_colored_copy(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Return a copy with the same points and a Y-height color gradient."""
    styled = o3d.geometry.PointCloud(point_cloud)
    points = np.asarray(styled.points)
    if len(points) == 0:
        return styled

    y = points[:, 1]
    denom = max(float(y.max() - y.min()), 1e-9)
    t = (y - y.min()) / denom
    colors = np.column_stack(
        (
            0.15 + 0.65 * t,
            0.35 + 0.35 * (1.0 - t),
            0.75 - 0.45 * t,
        )
    )
    styled.colors = o3d.utility.Vector3dVector(colors)
    return styled


def draw_height_colored(point_cloud: o3d.geometry.PointCloud, window_name: str) -> None:
    """Open a readable white-background point-cloud viewer."""
    styled = height_colored_copy(point_cloud)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=1400, height=900)
    vis.add_geometry(styled, reset_bounding_box=True)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()
