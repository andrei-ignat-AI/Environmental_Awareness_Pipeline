"""
load_scene_with_robot.py
------------------------
Loads the combined MuJoCo scene (depth-derived objects + C-arm robot) and sets
the robot's initial joint positions from the captured robot pose.

Controls
--------
Right arrow  : next sample
Left arrow   : previous sample
"""

import json
import os
import platform
import shutil
import sys
import time

import mujoco
import mujoco.viewer

from . import azurion_dataset


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DATASET_FOLDER = os.environ.get(
    "AZURION_CAPTURE_FOLDER",
    azurion_dataset.DEFAULT_CAPTURE_ROOT,
)

DEFAULT_SAMPLE_INDEX = 11
MUJOCO_TIMESTEP = 0.01

# Map legacy robot_pose.json joint names to MuJoCo XML joint names.
JOINT_NAME_MAP = {
    "Carriage": "Long",
    "HorBeam": "Z1Rot",
    "VerBeam": "Z2Rot",
    "Sleeve": "Prop",
    "CArc": "CArc",
}

ROBOT_JOINT_NAMES = {"Long", "Z1Rot", "Z2Rot", "Prop", "CArc"}

RIGHT_ARROW_KEY = 262
LEFT_ARROW_KEY = 263


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_robot_pose(dataset_folder, sample_index):
    """Return {mujoco_joint_name: position_value} for the given sample."""
    sample = get_sample(dataset_folder, sample_index)
    sample_path = sample.path
    state_path = os.path.join(sample_path, "robot_state.json")
    pose_path = os.path.join(sample_path, "robot_pose.json")

    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            pose_data = json.load(f)
        return {
            joint["jointName"]: float(joint["jointPosition"])
            for joint in pose_data.get("joints", [])
            if joint.get("jointName") in ROBOT_JOINT_NAMES
        }

    with open(pose_path, encoding="utf-8") as f:
        pose_data = json.load(f)

    joint_positions = {}
    for joint in pose_data["joints"]:
        mj_name = JOINT_NAME_MAP.get(joint["name"])
        if mj_name is not None:
            joint_positions[mj_name] = float(joint["jointPosition"])
    return joint_positions


def count_samples(dataset_folder):
    """Count how many sample_XXXX folders exist in dataset_folder."""
    return len(get_samples(dataset_folder))


def get_samples(dataset_folder):
    return azurion_dataset.discover_samples(dataset_folder)


def get_sample(dataset_folder, sample_index):
    samples = get_samples(dataset_folder)
    if not samples:
        raise RuntimeError(f"No sample folders found in: {dataset_folder}")
    return samples[sample_index % len(samples)]


def expected_xml_path(dataset_folder, sample_index, export_folder=None):
    if export_folder is None:
        export_folder = os.environ.get("MUJOCO_EXPORT_FOLDER", "mujoco_exports")
    sample = get_sample(dataset_folder, sample_index)
    return os.path.join(export_folder, sample.relative_path, f"{sample.sample_name}_mujoco.xml")


def apply_robot_pose(model, data, joint_positions):
    """Write joint positions into data.qpos and call mj_forward."""
    for joint_name, position in joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            print(f"  warning: joint '{joint_name}' not found in model - skipped")
            continue
        data.qpos[model.jnt_qposadr[joint_id]] = position
    mujoco.mj_forward(model, data)


def require_mjpython_for_viewer(script_name):
    if platform.system() != "Darwin":
        return
    if os.environ.get("MJPYTHON_BIN") or os.path.basename(sys.executable) == "mjpython":
        return
    mjpython = shutil.which("mjpython") or "/opt/homebrew/bin/mjpython"
    raise RuntimeError(
        "MuJoCo viewer scripts must use mjpython on macOS.\n"
        f"Run: AZURION_CAPTURE_FOLDER='{DATASET_FOLDER}' {mjpython} {script_name}"
    )


def xml_needs_regeneration(xml_path):
    """Detect missing/stale XML exports."""
    if not os.path.isfile(xml_path):
        return True

    with open(xml_path, encoding="utf-8") as f:
        xml_content = f.read()

    return (
        "robot_carc" not in xml_content
        or 'pos="1.2 0 0"' not in xml_content
        or 'axis="0 1 0"' not in xml_content
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    require_mjpython_for_viewer("load_scene_with_robot.py")

    from . import mujocoExportDraft as mje
    from . import decompositionDraft as decomposition

    num_samples = count_samples(DATASET_FOLDER)
    if num_samples == 0:
        raise RuntimeError(f"No sample folders found in: {DATASET_FOLDER}")
    print(f"Dataset: {DATASET_FOLDER}  ({num_samples} samples)")

    state = {
        "sample_index": int(os.environ.get("SAMPLE_INDEX", DEFAULT_SAMPLE_INDEX)),
        "requested_delta": None,
        "viewer": None,
    }

    def key_callback(keycode):
        if keycode == RIGHT_ARROW_KEY:
            state["requested_delta"] = 1
            if state["viewer"] is not None:
                state["viewer"].close()
        elif keycode == LEFT_ARROW_KEY:
            state["requested_delta"] = -1
            if state["viewer"] is not None:
                state["viewer"].close()

    dataset = None
    keep_running = True

    while keep_running:
        idx = state["sample_index"]
        sample = get_sample(DATASET_FOLDER, idx)
        xml_path = expected_xml_path(DATASET_FOLDER, idx)

        if xml_needs_regeneration(xml_path):
            print(f"{sample.relative_path}: generating XML with robot (this may take a moment)...")
            mje.DATASET_FOLDER = DATASET_FOLDER
            if dataset is None:
                dataset = decomposition.Dataset(DATASET_FOLDER)
            _, xml_path, _, _, _ = mje.load_sample_for_export(dataset, idx)

        print(f"\n--- {sample.relative_path} ---")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print(f"Loaded: {model.ngeom} geoms, {model.njnt} joints")

        joint_positions = load_robot_pose(DATASET_FOLDER, idx)
        for joint_name, value in joint_positions.items():
            print(f"  {joint_name} = {value:+.4f}")
        apply_robot_pose(model, data, joint_positions)

        state["requested_delta"] = None
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            state["viewer"] = viewer
            print("Right arrow: next sample  |  Left arrow: previous sample  |  Close to exit")

            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(MUJOCO_TIMESTEP)

        state["viewer"] = None

        if state["requested_delta"] is None:
            keep_running = False
        else:
            state["sample_index"] = (
                state["sample_index"] + state["requested_delta"]
            ) % num_samples


if __name__ == "__main__":
    main()
