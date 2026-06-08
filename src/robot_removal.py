"""Robot-removal wrappers around the pinned robert2/Ruben-derived source."""

from __future__ import annotations

from . import source_robert2_decomposition as source


RobotModel = source.RobotModel


def compute_robot_voxel_mask(robot_model, resolution, room_dimensions, extra_margin_m=0.01):
    return source.draft_compute_robot_voxel_mask(
        robot_model,
        resolution,
        room_dimensions,
        extra_margin_m=extra_margin_m,
    )

