"""
Basic informed RRT* planner for the C-arm POI.

Edit GOAL_POI below to choose the desired world coordinate for the POI
(the center/origin of the robot_carc body). The script loads one exported
MuJoCo sample, plans a collision-free joint path, then animates it.
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize

from .load_scene_with_robot import (
    DATASET_FOLDER,
    DEFAULT_SAMPLE_INDEX,
    apply_robot_pose,
    expected_xml_path,
    load_robot_pose,
    require_mjpython_for_viewer,
    xml_needs_regeneration,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Desired world position [x, y, z] for the POI / center of the C arc.
GOAL_POI = np.array([3.0, 2.0, 1.0], dtype=float)

ROBOT_BODY_NAME = "robot_carc"
JOINT_NAMES = ("Long", "Z1Rot", "Z2Rot", "Prop", "CArc")

MAX_ITERATIONS = 1500
STEP_SIZE = 0.20 #size between two nodes in the RRT* tree
GOAL_SAMPLE_RATE = 0.12 # how often we sample in the direction of the POI
GOAL_TOLERANCE = 0.08
REWIRE_RADIUS = 0.55 #connects two nodes if they are within this distance
COLLISION_CHECK_STEP = 0.08 #checks if collosion happens between two nodes
IK_RANDOM_RESTARTS = 80 # we have to convert the goal POI to a joint configuration. try this amount of configurations
MAX_GOAL_CANDIDATES = 12 #select this many of the IK solutions
SPACE_KEY = 32

# Planner limits. These override the XML ranges for the joints where the
# intended mechanical movement is wider than the current exported XML.
PLANNER_LIMIT_OVERRIDES = {
    "Long": ("relative", 5.0),     # +/- 5 m from the captured start pose
    "Z1Rot": ("absolute", math.pi),
    "Z2Rot": ("absolute", math.pi),
    "Prop": ("absolute", math.pi),
    "CArc": None,                 # keep XML range
}


@dataclass
class Node:
    q: np.ndarray
    parent: int
    cost: float


def q_distance(a, b):
    return float(np.linalg.norm(a - b))


def get_joint_info(model, start_q=None):
    joint_ids = []
    limits = []
    for index, name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in model: {name}")
        joint_ids.append(joint_id)
        joint_range = model.jnt_range[joint_id].copy()
        override = PLANNER_LIMIT_OVERRIDES.get(name)
        if override is not None:
            mode, value = override
            if mode == "relative":
                if start_q is not None:
                    joint_range = np.array([start_q[index] - value, start_q[index] + value])
            elif mode == "absolute":
                joint_range = np.array([-value, value])
        limits.append(joint_range)
    return joint_ids, np.asarray(limits, dtype=float)


def get_q(model, data, joint_ids):
    q = np.zeros(len(joint_ids), dtype=float)
    for i, joint_id in enumerate(joint_ids):
        q[i] = data.qpos[model.jnt_qposadr[joint_id]]
    return q


def set_q(model, data, joint_ids, q):
    for joint_id, value in zip(joint_ids, q):
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)


def poi_position(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f"Body not found in model: {ROBOT_BODY_NAME}")
    return data.xpos[body_id].copy()


def is_collision_free(model, data, joint_ids, q):
    set_q(model, data, joint_ids, q)
    return data.ncon == 0


def edge_is_collision_free(model, data, joint_ids, q_from, q_to):
    distance = q_distance(q_from, q_to)
    steps = max(2, int(math.ceil(distance / COLLISION_CHECK_STEP)))
    for alpha in np.linspace(0.0, 1.0, steps):
        q = (1.0 - alpha) * q_from + alpha * q_to
        if not is_collision_free(model, data, joint_ids, q):
            return False
    return True


def steer(q_from, q_to, step_size):
    delta = q_to - q_from
    distance = np.linalg.norm(delta)
    if distance <= step_size:
        return q_to.copy()
    return q_from + delta / distance * step_size


def solve_goal_configurations(model, data, joint_ids, limits, start_q, goal_poi):
    """Find multiple joint-space goals whose C-arc center is near goal_poi."""

    def objective(q):
        set_q(model, data, joint_ids, q)
        position_error = np.linalg.norm(poi_position(model, data) - goal_poi)
        regularization = 0.02 * np.linalg.norm(q - start_q)
        return position_error + regularization

    best = None
    starts = [start_q]
    for _ in range(IK_RANDOM_RESTARTS):
        starts.append(np.array([random.uniform(lo, hi) for lo, hi in limits]))

    candidates = []
    bounds = [(float(lo), float(hi)) for lo, hi in limits]
    for initial_q in starts:
        result = minimize(
            objective,
            initial_q,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-5, "disp": False},
        )
        q = np.clip(result.x, limits[:, 0], limits[:, 1])
        set_q(model, data, joint_ids, q)
        error = np.linalg.norm(poi_position(model, data) - goal_poi)
        if best is None or error < best[0]:
            best = (error, q.copy())
        if all(q_distance(q, existing_q) > 0.12 for _, existing_q, _ in candidates):
            candidates.append((error, q.copy(), data.ncon == 0))

    candidates.sort(key=lambda candidate: (not candidate[2], candidate[0]))
    return candidates[:MAX_GOAL_CANDIDATES], best


def rotation_to_world_frame(start, goal):
    direction = goal - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return np.eye(len(start))

    e1 = direction / length
    basis = np.eye(len(start))
    basis[:, 0] = e1
    q, _ = np.linalg.qr(basis)
    if np.dot(q[:, 0], e1) < 0.0:
        q[:, 0] *= -1.0
    return q


def sample_informed(start_q, goal_q, best_cost, limits):
    direct_cost = q_distance(start_q, goal_q)
    if best_cost is None or best_cost == float("inf"):
        return np.array([random.uniform(lo, hi) for lo, hi in limits])

    center = 0.5 * (start_q + goal_q)
    rotation = rotation_to_world_frame(start_q, goal_q)
    radii = np.full(len(start_q), math.sqrt(max(best_cost**2 - direct_cost**2, 0.0)) / 2.0)
    radii[0] = best_cost / 2.0

    for _ in range(100):
        direction = np.random.normal(size=len(start_q))
        direction /= np.linalg.norm(direction)
        radius = random.random() ** (1.0 / len(start_q))
        q = center + rotation @ (direction * radius * radii)
        if np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]):
            return q

    return np.array([random.uniform(lo, hi) for lo, hi in limits])


def nearest_node(nodes, q):
    return min(range(len(nodes)), key=lambda i: q_distance(nodes[i].q, q))


def near_nodes(nodes, q, radius):
    return [i for i, node in enumerate(nodes) if q_distance(node.q, q) <= radius]


def is_ancestor(nodes, possible_ancestor, node_index):
    index = node_index
    while index != -1:
        if index == possible_ancestor:
            return True
        index = nodes[index].parent
    return False


def extract_path(nodes, goal_index):
    path = []
    index = goal_index
    while index != -1:
        path.append(nodes[index].q.copy())
        index = nodes[index].parent
    path.reverse()
    return path


def informed_rrt_star(model, data, joint_ids, limits, start_q, goal_q):
    if not is_collision_free(model, data, joint_ids, start_q):
        raise RuntimeError("Start configuration is in collision.")
    if not is_collision_free(model, data, joint_ids, goal_q):
        raise RuntimeError("Goal configuration is in collision.")

    nodes = [Node(start_q.copy(), parent=-1, cost=0.0)]
    best_goal_index = None
    best_cost = float("inf")

    for iteration in range(MAX_ITERATIONS):
        if random.random() < GOAL_SAMPLE_RATE:
            q_sample = goal_q
        else:
            q_sample = sample_informed(start_q, goal_q, best_cost, limits)

        nearest_index = nearest_node(nodes, q_sample)
        q_new = steer(nodes[nearest_index].q, q_sample, STEP_SIZE)
        q_new = np.clip(q_new, limits[:, 0], limits[:, 1])

        if not edge_is_collision_free(model, data, joint_ids, nodes[nearest_index].q, q_new):
            continue

        parent_index = nearest_index
        parent_cost = nodes[nearest_index].cost + q_distance(nodes[nearest_index].q, q_new)

        for near_index in near_nodes(nodes, q_new, REWIRE_RADIUS):
            candidate_cost = nodes[near_index].cost + q_distance(nodes[near_index].q, q_new)
            if candidate_cost < parent_cost and edge_is_collision_free(
                model, data, joint_ids, nodes[near_index].q, q_new
            ):
                parent_index = near_index
                parent_cost = candidate_cost

        nodes.append(Node(q_new, parent=parent_index, cost=parent_cost))
        new_index = len(nodes) - 1

        for near_index in near_nodes(nodes[:-1], q_new, REWIRE_RADIUS):
            if is_ancestor(nodes, near_index, new_index):
                continue
            new_cost = nodes[new_index].cost + q_distance(nodes[new_index].q, nodes[near_index].q)
            if new_cost < nodes[near_index].cost and edge_is_collision_free(
                model, data, joint_ids, q_new, nodes[near_index].q
            ):
                nodes[near_index].parent = new_index
                nodes[near_index].cost = new_cost

        if q_distance(q_new, goal_q) <= GOAL_TOLERANCE:
            total_cost = nodes[new_index].cost + q_distance(q_new, goal_q)
            if total_cost < best_cost and edge_is_collision_free(model, data, joint_ids, q_new, goal_q):
                nodes.append(Node(goal_q.copy(), parent=new_index, cost=total_cost))
                best_goal_index = len(nodes) - 1
                best_cost = total_cost
                print(f"found path at iteration {iteration}, cost={best_cost:.3f}")

    if best_goal_index is None:
        return None
    return extract_path(nodes, best_goal_index)


def densify_path(path, max_step=0.04):
    dense = []
    for q0, q1 in zip(path[:-1], path[1:]):
        steps = max(2, int(math.ceil(q_distance(q0, q1) / max_step)))
        for alpha in np.linspace(0.0, 1.0, steps, endpoint=False):
            dense.append((1.0 - alpha) * q0 + alpha * q1)
    dense.append(path[-1])
    return dense


def add_goal_marker(viewer, goal_poi):
    viewer.user_scn.ngeom = 0
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.06, 0.06, 0.06], dtype=float),
        goal_poi,
        np.eye(3).reshape(-1),
        np.array([1.0, 0.0, 0.0, 1.0], dtype=float),
    )
    viewer.user_scn.ngeom += 1


def show_static_view(model, data, goal_poi, message):
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        print(message)
        while viewer.is_running():
            add_goal_marker(viewer, goal_poi)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.03)


def animate_path(model, data, joint_ids, path, goal_poi):
    state = {"paused": False}

    def key_callback(keycode):
        if keycode == SPACE_KEY:
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Resumed")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        print("Animating path. Space: pause/resume  |  Close the viewer to exit.")
        while viewer.is_running():
            for q in path:
                if not viewer.is_running():
                    break
                while viewer.is_running() and state["paused"]:
                    add_goal_marker(viewer, goal_poi)
                    viewer.sync()
                    time.sleep(0.03)
                set_q(model, data, joint_ids, q)
                add_goal_marker(viewer, goal_poi)
                viewer.sync()
                time.sleep(0.03)


def direct_path_if_clear(model, data, joint_ids, start_q, goal_q):
    if not edge_is_collision_free(model, data, joint_ids, start_q, goal_q):
        return None
    return densify_path([start_q, goal_q])


def ensure_xml(sample_index):
    xml_path = expected_xml_path(DATASET_FOLDER, sample_index)
    if not xml_needs_regeneration(xml_path):
        return xml_path

    from . import decompositionDraft as decomposition
    from . import mujocoExportDraft as mje

    mje.DATASET_FOLDER = DATASET_FOLDER
    dataset = decomposition.Dataset(DATASET_FOLDER)
    _, xml_path, _, _, _ = mje.load_sample_for_export(dataset, sample_index)
    return xml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--goal", type=float, nargs=3, default=GOAL_POI.tolist())
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    if not args.no_viewer:
        require_mjpython_for_viewer("informed_rrt_star_demo.py")

    random.seed(7)
    np.random.seed(7)

    xml_path = ensure_xml(args.sample)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    joint_positions = load_robot_pose(DATASET_FOLDER, args.sample)
    apply_robot_pose(model, data, joint_positions)

    joint_ids, _ = get_joint_info(model)
    start_q = get_q(model, data, joint_ids)
    joint_ids, limits = get_joint_info(model, start_q)
    goal_poi = np.array(args.goal, dtype=float)

    print(f"start POI: {poi_position(model, data)}")
    print(f"goal  POI: {goal_poi}")

    goal_candidates, best_goal = solve_goal_configurations(
        model, data, joint_ids, limits, start_q, goal_poi
    )
    best_goal_error, best_goal_q = best_goal
    print(f"best IK POI error: {best_goal_error:.4f} m")

    free_goal_candidates = [
        (error, q) for error, q, collision_free in goal_candidates if collision_free
    ]
    print(
        f"IK candidates: {len(goal_candidates)} total, "
        f"{len(free_goal_candidates)} collision-free"
    )

    path = None
    goal_q = None
    for candidate_index, (goal_error, candidate_q) in enumerate(free_goal_candidates, start=1):
        print(f"trying goal candidate {candidate_index}: IK error={goal_error:.4f} m")
        print("goal q:", dict(zip(JOINT_NAMES, candidate_q.round(4))))
        path = direct_path_if_clear(model, data, joint_ids, start_q, candidate_q)
        if path is not None:
            print("direct collision-free path found")
            goal_q = candidate_q
            break
        try:
            path = informed_rrt_star(model, data, joint_ids, limits, start_q, candidate_q)
        except RuntimeError as exc:
            print(f"candidate rejected: {exc}")
            continue
        if path is not None:
            goal_q = candidate_q
            break

    if not free_goal_candidates:
        print("No collision-free IK goal candidate found.")
        if not args.no_viewer:
            set_q(model, data, joint_ids, start_q)
            show_static_view(
                model,
                data,
                goal_poi,
                "No path available. Showing original robot position and red GOAL_POI marker. Close the viewer to exit.",
            )
        return

    if path is None:
        print("No path found to any collision-free IK candidate.")
        if not args.no_viewer:
            set_q(model, data, joint_ids, start_q)
            show_static_view(
                model,
                data,
                goal_poi,
                "No path available. Showing original robot position and red GOAL_POI marker. Close the viewer to exit.",
            )
        return

    path = densify_path(path)
    print(f"path points: {len(path)}")

    if args.no_viewer:
        return

    animate_path(model, data, joint_ids, path, goal_poi)


if __name__ == "__main__":
    main()
