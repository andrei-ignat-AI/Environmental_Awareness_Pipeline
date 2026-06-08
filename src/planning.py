"""Headless Ruben-style planning smoke wrapper."""

from __future__ import annotations

import random
from pathlib import Path

import mujoco
import numpy as np

from . import source_ruben_planning as planner
from . import mujoco_export
from .config import PipelineConfig
from .types import PlanningRunResult, SampleRunResult


def apply_config(config: PipelineConfig) -> None:
    cfg = config.planning
    planner.GOAL_POI = np.asarray(cfg.goal_poi, dtype=float)
    planner.MAX_ITERATIONS = int(cfg.max_iterations)
    planner.STEP_SIZE = float(cfg.step_size)
    planner.GOAL_SAMPLE_RATE = float(cfg.goal_sample_rate)
    planner.GOAL_TOLERANCE = float(cfg.goal_tolerance)
    planner.REWIRE_RADIUS = float(cfg.rewire_radius)
    planner.COLLISION_CHECK_STEP = float(cfg.collision_check_step)
    planner.IK_RANDOM_RESTARTS = int(cfg.ik_random_restarts)
    planner.MAX_GOAL_CANDIDATES = int(cfg.max_goal_candidates)


def _path_to_lists(path: list[np.ndarray] | None) -> list[list[float]]:
    if path is None:
        return []
    return [np.asarray(point, dtype=float).tolist() for point in path]


def _poi_path(model, data, joint_ids, path: list[np.ndarray]) -> list[list[float]]:
    points: list[list[float]] = []
    for q in path:
        planner.set_q(model, data, joint_ids, q)
        points.append(planner.poi_position(model, data).tolist())
    return points


def run_headless(sample_result: SampleRunResult, config: PipelineConfig) -> PlanningRunResult:
    apply_config(config)
    goal_poi = tuple(float(value) for value in config.planning.goal_poi)
    if sample_result.status != "success" or sample_result.xml_path is None:
        return PlanningRunResult(
            status="skipped",
            sample=sample_result.sample,
            goal_poi=goal_poi,
            message="sample did not export successfully",
        )
    if not config.mujoco_export.include_robot:
        return PlanningRunResult(
            status="skipped",
            sample=sample_result.sample,
            goal_poi=goal_poi,
            message=(
                "planning requires a robot-inclusive MuJoCo export; "
                "include_robot=False is the Robert-facing metrics default"
            ),
        )

    try:
        random.seed(7)
        np.random.seed(7)
        model = mujoco.MjModel.from_xml_path(str(Path(sample_result.xml_path)))
        data = mujoco.MjData(model)
        mujoco_export.source.apply_robot_pose_from_json(model, data, str(sample_result.sample.path))

        joint_ids, _ = planner.get_joint_info(model)
        start_q = planner.get_q(model, data, joint_ids)
        joint_ids, limits = planner.get_joint_info(model, start_q)
        goal = np.asarray(config.planning.goal_poi, dtype=float)
        goal_candidates, best_goal = planner.solve_goal_configurations(
            model,
            data,
            joint_ids,
            limits,
            start_q,
            goal,
        )
        best_error = float(best_goal[0]) if best_goal is not None else None
        free_goal_candidates = [
            (error, q) for error, q, collision_free in goal_candidates if collision_free
        ]
        if not free_goal_candidates:
            return PlanningRunResult(
                status="failed",
                sample=sample_result.sample,
                goal_poi=goal_poi,
                message=f"no collision-free IK candidate; best_error={best_error}",
            )

        path = None
        chosen_error = None
        for goal_error, candidate_q in free_goal_candidates:
            chosen_error = float(goal_error)
            path = planner.direct_path_if_clear(model, data, joint_ids, start_q, candidate_q)
            if path is None:
                try:
                    path = planner.informed_rrt_star(model, data, joint_ids, limits, start_q, candidate_q)
                except RuntimeError:
                    path = None
            if path is not None:
                break

        if path is None:
            return PlanningRunResult(
                status="failed",
                sample=sample_result.sample,
                goal_poi=goal_poi,
                message=f"no path found across {len(free_goal_candidates)} collision-free IK candidates",
            )

        dense_path = planner.densify_path(path)
        return PlanningRunResult(
            status="success",
            sample=sample_result.sample,
            goal_poi=goal_poi,
            path=_path_to_lists(dense_path),
            poi_path=_poi_path(model, data, joint_ids, dense_path),
            message=f"path points={len(dense_path)}; chosen_ik_error={chosen_error}",
        )
    except Exception as exc:
        return PlanningRunResult(
            status="error",
            sample=sample_result.sample,
            goal_poi=goal_poi,
            error=repr(exc),
        )
