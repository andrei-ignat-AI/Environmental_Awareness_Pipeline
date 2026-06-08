import json
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from . import decompositionDraft as decomposition
import open3d as o3d


# ---------------------------------draft export settings---------------------------------
DATASET_FOLDER = os.environ.get(
    "AZURION_CAPTURE_FOLDER",
    decomposition.DepthImageFolder,
)
EXPORT_FOLDER = os.environ.get("MUJOCO_EXPORT_FOLDER", "mujoco_exports")
DEFAULT_SAMPLE_INDEX = 0
RIGHT_ARROW_KEY = 262
LEFT_ARROW_KEY = 263
SAVE_KEY = 83
# keep this false for mjpython on macos. like open3d, matplotlib's macos gui
# backend cannot create windows from the script thread while mujoco owns main.
SHOW_SIDE_RGB_IMAGE = False
# keep this false for mjpython on macos. mjpython gives the main thread to mujoco,
# and open3d crashes if it tries to create its own window from the script thread.
SHOW_OPEN3D_DECOMPOSITION = False
SHOW_MUJOCO_VIEWER = True
MUJOCO_TIMESTEP = 0.01
EXPORT_SINGLE_BOX_PER_OBJECT = False
REMOVE_TABLE_IN_MUJOCO_EXPORT = True #os.environ.get("MUJOCO_REMOVE_TABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
INCLUDE_TABLE_IN_MUJOCO_EXPORT = not REMOVE_TABLE_IN_MUJOCO_EXPORT
MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT = False
EXPORT_OCCLUSION_ZONE_BOXES = True
EXPORT_BLIND_SPOTS = True
EXPORT_GEOM_MODE = "convex_hull"  # "convex_hull" is tighter; "box" is the old fallback.
CONVEX_HULL_MARGIN_M = 0.10
# Robert's MuJoCo voxel diagnostic shift is intentional; keep it unless explicitly retuning that path.
MANUAL_DIAGNOSTIC_SHIFT = np.array([-0.00, -0.00, -0.00], dtype=np.float64)#np.array([-0.05, -0.05, -0.05], dtype=np.float64)
INCLUDE_ROBOT = False
ROBOT_HULLS_FOLDER = "Robotarm/hulls"
CONVEX_HULL_MARGIN_M = 0.10
VALIDATE_AXIS_CORRESPONDENCE = True
AXIS_CHECK_TOLERANCE_M = 0.02
EXPORT_ALL=True
OPEN3D_TO_MUJOCO_AXIS_SWAP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

_ROBOT_BODY_HULLS = [
    ("robot_carriage", ["Carriage00"]),
    ("robot_hor_beam", ["HorBeam000", "HorBeam001"]),
    ("robot_ver_beam", ["VerBeam000", "VerBeam001"]),
    ("robot_sleeve", ["Sleeve000", "Sleeve001", "Sleeve002"]),
    ("robot_carc", ["CArc000", "CArc001", "CArc002", "CArc003", "CArc004", "CArc005", "CArc006"]),
]

_ROBOT_JOINT_NAME_MAP = {
    "Carriage": "Long",
    "HorBeam": "Z1Rot",
    "VerBeam": "Z2Rot",
    "Sleeve": "Prop",
    "CArc": "CArc",
}

_ROBOT_ORIGIN_OFFSET = 2.59999 * 0.63
_ROBOT_BODY_ZERO_POS = {
    "robot_carriage": np.array([_ROBOT_ORIGIN_OFFSET, 0.0, 2.6], dtype=np.float64),
    "robot_hor_beam": np.array([_ROBOT_ORIGIN_OFFSET, 0.0, 2.6], dtype=np.float64),
    "robot_ver_beam": np.array([_ROBOT_ORIGIN_OFFSET + 1.2, 0.0, 2.6], dtype=np.float64),
    "robot_sleeve": np.array([_ROBOT_ORIGIN_OFFSET + 0.4, 0.0, 1.1], dtype=np.float64),
    "robot_carc": np.array([_ROBOT_ORIGIN_OFFSET + 1.8, 0.0, 1.1], dtype=np.float64),
}
_robot_hull_cache = None


def mj_format(values):
    # mujoco xml wants plain space-separated numbers.
    return " ".join(f"{float(value):.6g}" for value in values)


def open3d_vector_to_mujoco(vector):
    # open3d uses y-up here; mujoco is z-up. the -z keeps the frame right-handed.
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([x, -z, y], dtype=np.float64)


def mujoco_to_open3d_vector(vector):
    # This is the inverse of open3d_vector_to_mujoco ([x, -z, y]).
    # To map MJ world coordinates back to the Unity/Open3D frame:
    # MJ(x) -> O3D(x)
    # MJ(y) -> O3D(-z)
    # MJ(z) -> O3D(y)
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([x, z, -y], dtype=np.float64)


def open3d_rotation_to_mujoco(rotation):
    # same right-handed axis swap as positions: [x, y, z] -> [x, -z, y].
    return OPEN3D_TO_MUJOCO_AXIS_SWAP @ np.asarray(rotation, dtype=np.float64) @ OPEN3D_TO_MUJOCO_AXIS_SWAP.T


def rotation_matrix_to_mujoco_quat(rotation):
    # scipy gives xyzw, mujoco xml expects wxyz.
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def color_to_rgba(color):
    rgb = np.asarray(color, dtype=np.float64)
    return np.array([rgb[0], rgb[1], rgb[2], 0.55], dtype=np.float64)


def component_to_voxel_corner_points(component, room_dimensions, margin_m):
    # use voxel corners instead of centers so the hull encloses the actual 5 cm grid cells.
    centers = decomposition.draft_component_to_points(
        component,
        decomposition.default_voxel_size,
        room_dimensions,
    )
    half_width = decomposition.default_voxel_size * 0.5 + float(margin_m)
    corner_offsets = np.asarray([
        [sx, sy, sz]
        for sx in (-half_width, half_width)
        for sy in (-half_width, half_width)
        for sz in (-half_width, half_width)
    ], dtype=np.float64)
    return (centers[:, None, :] + corner_offsets[None, :, :] + MANUAL_DIAGNOSTIC_SHIFT).reshape((-1, 3))


def build_convex_hull_mesh(points_open3d):
    points_mj = np.asarray([open3d_vector_to_mujoco(point) for point in points_open3d], dtype=np.float64)
    points_mj = np.unique(np.round(points_mj, decimals=6), axis=0)
    if len(points_mj) < 4:
        raise ValueError("convex hull needs at least four unique points")

    try:
        hull = ConvexHull(points_mj)
    except QhullError:
        # QJ slightly perturbs nearly flat/degenerate voxel sets so qhull can still form a valid hull.
        hull = ConvexHull(points_mj, qhull_options="QJ")

    used_indices = np.asarray(hull.vertices, dtype=np.int32)
    index_map = {int(old_index): new_index for new_index, old_index in enumerate(used_indices)}
    faces = []
    for simplex in np.asarray(hull.simplices, dtype=np.int32):
        if all(int(index) in index_map for index in simplex):
            faces.append([index_map[int(index)] for index in simplex])
    if not faces:
        raise ValueError("convex hull produced no triangle faces")

    return points_mj[used_indices], np.asarray(faces, dtype=np.int32)


def robot_rail_pos_from_sample(sample_path):
    state_path = os.path.join(sample_path, "robot_state.json")
    pose_path = os.path.join(sample_path, "robot_pose.json")
    carriage_position = None
    long_position = 0.0

    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            pose_data = json.load(handle)
        for joint in pose_data.get("joints", []):
            if joint.get("jointName") == "Long":
                carriage_position = joint.get("worldPosition")
                long_position = float(joint.get("jointPosition", 0.0))
                break
    elif os.path.isfile(pose_path):
        with open(pose_path, encoding="utf-8") as handle:
            pose_data = json.load(handle)
        for joint in pose_data.get("joints", []):
            if joint.get("name") == "Carriage":
                carriage_position = joint.get("worldPosition")
                long_position = float(joint.get("jointPosition", 0.0))
                break

    if carriage_position is None:
        return _ROBOT_BODY_ZERO_POS["robot_carriage"].copy()

    carriage_mj = np.array([
        float(carriage_position["x"]),
        -float(carriage_position["z"]),
        float(carriage_position["y"]),
    ], dtype=np.float64)
    return carriage_mj - np.array([-1.0, 0.0, 0.0], dtype=np.float64) * long_position


def build_robot_hull_records(hulls_folder):
    global _robot_hull_cache
    if _robot_hull_cache is not None:
        return _robot_hull_cache

    records = {}
    for body_name, stems in _ROBOT_BODY_HULLS:
        body_world_pos = _ROBOT_BODY_ZERO_POS[body_name]
        for stem in stems:
            mesh = o3d.io.read_triangle_mesh(os.path.join(hulls_folder, f"{stem}.stl"))
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            # Robot hull STLs are in the imported Unity frame. This is Ruben's
            # validated conversion into MuJoCo body-local coordinates.
            vertices_world = np.column_stack([
                _ROBOT_ORIGIN_OFFSET - vertices[:, 1],
                vertices[:, 0],
                vertices[:, 2],
            ])
            records[stem] = {
                "stem": stem,
                "body": body_name,
                "mesh_name": f"robot_{stem}_mesh",
                "geom_name": f"robot_{stem}",
                "vertices_mujoco": (vertices_world - body_world_pos).tolist(),
                "faces": np.asarray(mesh.triangles, dtype=np.int32).tolist(),
            }

    _robot_hull_cache = records
    return records


def add_robot_body_tree(worldbody, hull_records, rail_pos):
    body_to_stems = dict(_ROBOT_BODY_HULLS)

    def add_geoms(body_elem, body_name):
        for stem in body_to_stems.get(body_name, []):
            record = hull_records[stem]
            ET.SubElement(body_elem, "geom", {
                "name": record["geom_name"],
                "type": "mesh",
                "mesh": record["mesh_name"],
                "contype": "2",
                "conaffinity": "1",
                "rgba": "0.7 0.7 0.8 0.8",
            })

    rail = ET.SubElement(worldbody, "body", {"name": "robot_rail", "pos": mj_format(rail_pos)})
    carriage = ET.SubElement(rail, "body", {"name": "robot_carriage"})
    ET.SubElement(carriage, "joint", {"name": "Long", "type": "slide", "axis": "-1 0 0", "range": "-4 4"})

    hor_beam = ET.SubElement(carriage, "body", {"name": "robot_hor_beam"})
    ET.SubElement(hor_beam, "joint", {"name": "Z1Rot", "type": "hinge", "axis": "0 0 1", "range": "-3.142 3.142"})

    ver_beam = ET.SubElement(hor_beam, "body", {"name": "robot_ver_beam", "pos": "1.2 0 0"})
    ET.SubElement(ver_beam, "joint", {"name": "Z2Rot", "type": "hinge", "axis": "0 0 1", "range": "-3.142 3.142"})

    sleeve = ET.SubElement(ver_beam, "body", {"name": "robot_sleeve", "pos": "-0.8 0 -1.5"})
    ET.SubElement(sleeve, "joint", {"name": "Prop", "type": "hinge", "axis": "-1 0 0", "range": "-3.142 3.142"})

    carc = ET.SubElement(sleeve, "body", {"name": "robot_carc", "pos": "1.4 0 0"})
    ET.SubElement(carc, "joint", {"name": "CArc", "type": "hinge", "axis": "0 1 0", "range": "-3.142 3.142"})

    add_geoms(carc, "robot_carc")
    add_geoms(sleeve, "robot_sleeve")
    add_geoms(ver_beam, "robot_ver_beam")
    add_geoms(hor_beam, "robot_hor_beam")
    add_geoms(carriage, "robot_carriage")


def apply_robot_pose_from_json(model, data, sample_path):
    if not INCLUDE_ROBOT:
        return

    state_path = os.path.join(sample_path, "robot_state.json")
    pose_path = os.path.join(sample_path, "robot_pose.json")
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            pose_data = json.load(handle)
        for joint in pose_data.get("joints", []):
            joint_name = joint.get("jointName")
            if joint_name not in {"Long", "Z1Rot", "Z2Rot", "Prop", "CArc"}:
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                data.qpos[model.jnt_qposadr[joint_id]] = float(joint["jointPosition"])
        mujoco.mj_forward(model, data)
        return

    if not os.path.isfile(pose_path):
        print("robot pose not found, using zero robot pose:", pose_path)
        return
    with open(pose_path, encoding="utf-8") as handle:
        pose_data = json.load(handle)
    for joint in pose_data.get("joints", []):
        joint_name = _ROBOT_JOINT_NAME_MAP.get(joint.get("name"))
        if joint_name is None:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = float(joint["jointPosition"])
    mujoco.mj_forward(model, data)


def box_to_export_record(box, record):
    center_mj = open3d_vector_to_mujoco(box.center)
    rotation_mj = open3d_rotation_to_mujoco(box.R)
    quat_mj = rotation_matrix_to_mujoco_quat(rotation_mj)
    half_size_mj = np.asarray(box.extent, dtype=np.float64) * 0.5

    kind = record.get("kind", "object")
    name = f"{kind}_{record['object_index']:02d}_box_{record['box_index']:02d}"
    return {
        "name": name,
        "kind": kind,
        "object_index": int(record["object_index"]),
        "box_index": int(record["box_index"]),
        "voxel_count": int(record["voxel_count"]),
        "occlusion_voxel_count": int(record.get("occlusion_voxel_count", 0)),
        "center_open3d": np.asarray(box.center, dtype=np.float64).tolist(),
        "extent_open3d": np.asarray(box.extent, dtype=np.float64).tolist(),
        "rotation_open3d": np.asarray(box.R, dtype=np.float64).tolist(),
        "pos_mujoco": center_mj.tolist(),
        "quat_mujoco": quat_mj.tolist(),
        "half_size_mujoco": half_size_mj.tolist(),
        "rgba": color_to_rgba(box.color).tolist(),
    }


def component_to_mesh_export_record(component, record, room_dimensions):
    kind = record.get("kind", "entity")
    object_index = int(record["object_index"])
    chunk_index = int(record.get("chunk_index", 0))
    name = f"{kind}_{object_index:02d}_hull_{chunk_index:02d}"
    points_open3d = component_to_voxel_corner_points(
        component,
        room_dimensions,
        CONVEX_HULL_MARGIN_M,
    )
    vertices, faces = build_convex_hull_mesh(points_open3d)
    
    if kind == "occlusion":
        color = decomposition.OCCLUSION_COLOR
    elif kind == "blind_spot":
        color = decomposition.BLIND_SPOT_COLOR
    else:
        color = decomposition.draft_component_color(object_index)

    return {
        "name": name,
        "mesh_name": f"{name}_mesh",
        "geom_type": "mesh",
        "kind": kind,
        "object_index": object_index,
        "chunk_index": chunk_index,
        "box_index": int(record.get("box_index", chunk_index)),
        "voxel_count": int(record["voxel_count"]),
        "occlusion_voxel_count": int(record.get("occlusion_voxel_count", 0)),
        "convex_hull_margin_m": float(CONVEX_HULL_MARGIN_M),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_mujoco": vertices.tolist(),
        "faces": faces.tolist(),
        "rgba": color_to_rgba(color).tolist(),
    }


def build_mujoco_xml(export_records, sample_name, robot_hull_records=None, robot_rail_pos=None):
    root = ET.Element("mujoco", {"model": f"{sample_name}_decomposition"})

    ET.SubElement(root, "compiler", {
        "angle": "radian",
    })
    ET.SubElement(root, "option", {
        "timestep": str(MUJOCO_TIMESTEP),
        "gravity": "0 0 0",
    })

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "1 1 1",
        "rgb2": "0.72 0.82 0.95",
        "width": "512",
        "height": "512",
    })
    for record in export_records:
        if record.get("geom_type") != "mesh":
            continue
        ET.SubElement(asset, "mesh", {
            "name": record["mesh_name"],
            "vertex": mj_format(np.asarray(record["vertices_mujoco"]).ravel()),
            "face": " ".join(str(int(value)) for value in np.asarray(record["faces"]).ravel()),
        })

    if robot_hull_records:
        for record in robot_hull_records.values():
            ET.SubElement(asset, "mesh", {
                "name": record["mesh_name"],
                "vertex": mj_format(np.asarray(record["vertices_mujoco"]).ravel()),
                "face": " ".join(str(int(value)) for value in np.asarray(record["faces"]).ravel()),
            })

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {
        "name": "main_light",
        "pos": "0 -4 6",
        "dir": "0 1 -1",
        "diffuse": "0.8 0.8 0.8",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "overview",
        "pos": "0 -8 5",
        "xyaxes": "1 0 0 0 0.707 0.707",
    })

    for record in export_records:
        if record.get("geom_type") == "mesh":
            ET.SubElement(worldbody, "geom", {
                "name": record["name"],
                "type": "mesh",
                "mesh": record["mesh_name"],
                "rgba": mj_format(record["rgba"]),
                "contype": "1",
                "conaffinity": "1",
            })
        else:
            ET.SubElement(worldbody, "geom", {
                "name": record["name"],
                "type": "box",
                "pos": mj_format(record["pos_mujoco"]),
                "quat": mj_format(record["quat_mujoco"]),
                "size": mj_format(record["half_size_mujoco"]),
                "rgba": mj_format(record["rgba"]),
                "contype": "1",
                "conaffinity": "1",
            })

    if robot_hull_records:
        add_robot_body_tree(worldbody, robot_hull_records, robot_rail_pos)

    rough_xml = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough_xml).toprettyxml(indent="  ")


def save_export_files(result):
    sample_path = result["sample_path"]
    sample_name = os.path.basename(sample_path)
    try:
        sample_rel_path = os.path.relpath(os.path.abspath(sample_path), os.path.abspath(DATASET_FOLDER))
    except ValueError:
        sample_rel_path = sample_name
    if sample_rel_path.startswith(".."):
        sample_rel_path = sample_name
    export_dir = os.path.join(EXPORT_FOLDER, sample_rel_path)
    os.makedirs(export_dir, exist_ok=True)

    if EXPORT_GEOM_MODE == "convex_hull":
        export_records = [
            component_to_mesh_export_record(component, record, result["room_dimensions"])
            for component, record in zip(result.get("hull_chunks", []), result.get("hull_records", []))
        ]
        if EXPORT_BLIND_SPOTS:
            export_records.extend(
                component_to_mesh_export_record(component, record, result["room_dimensions"])
                for component, record in zip(result.get("blind_spot_hull_chunks", []), result.get("blind_spot_hull_records", []))
            )
        if EXPORT_OCCLUSION_ZONE_BOXES and not MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT:
            export_records.extend(
                component_to_mesh_export_record(component, record, result["room_dimensions"])
                for component, record in zip(result.get("occlusion_hull_chunks", []), result.get("occlusion_hull_records", []))
            )
    elif EXPORT_GEOM_MODE == "box":
        export_records = [
            box_to_export_record(box, record)
            for box, record in zip(result["boxes"], result["box_records"])
        ]
    else:
        raise ValueError("EXPORT_GEOM_MODE must be 'convex_hull' or 'box'")

    if EXPORT_OCCLUSION_ZONE_BOXES and EXPORT_GEOM_MODE == "box":
        export_records.extend(
            box_to_export_record(box, record)
            for box, record in zip(result["occlusion_boxes"], result["occlusion_box_records"])
        )
        if EXPORT_BLIND_SPOTS:
            export_records.extend(
                box_to_export_record(box, record)
                for box, record in zip(result.get("blind_spot_boxes", []), result.get("blind_spot_box_records", []))
            )

    robot_hull_records = build_robot_hull_records(ROBOT_HULLS_FOLDER) if INCLUDE_ROBOT else None
    robot_rail_pos = robot_rail_pos_from_sample(sample_path) if INCLUDE_ROBOT else None
    xml_text = build_mujoco_xml(export_records, sample_name, robot_hull_records, robot_rail_pos)
    xml_path = os.path.join(export_dir, f"{sample_name}_mujoco.xml")
    json_path = os.path.join(export_dir, f"{sample_name}_boxes.json")

    with open(xml_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_text)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump({
            "sample_name": sample_name,
            "sample_rel_path": sample_rel_path,
            "sample_path": sample_path,
            "geom_export_mode": EXPORT_GEOM_MODE,
            "geom_count": len(export_records),
            "box_count": len(export_records),
            "convex_hull_margin_m": CONVEX_HULL_MARGIN_M if EXPORT_GEOM_MODE == "convex_hull" else None,
            "remove_table": REMOVE_TABLE_IN_MUJOCO_EXPORT,
            "include_table": INCLUDE_TABLE_IN_MUJOCO_EXPORT,
            "include_robot": INCLUDE_ROBOT,
            "merge_occlusions_with_objects": MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT,
            "export_occlusion_zone_boxes": EXPORT_OCCLUSION_ZONE_BOXES,
            "export_blind_spots": EXPORT_BLIND_SPOTS,
            "geoms": export_records,
            "boxes": export_records,
        }, json_file, indent=2)

    print("mujoco xml saved:", xml_path)
    print("debug geom json saved:", json_path)
    print("mujoco geom export mode:", EXPORT_GEOM_MODE)
    return xml_path, json_path, export_records


def compiled_mesh_global_vertices(model, data, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError("geom is not backed by a mesh asset")

    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    local_vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    return local_vertices @ rotation.T + position


def validate_axis_correspondence(model, data, export_records):
    # this checks the exact handoff assumption:
    # python/open3d is y-up, mujoco is z-up, so we export [x, y, z] as [x, -z, y].
    if not VALIDATE_AXIS_CORRESPONDENCE:
        return

    axis_determinant = float(np.linalg.det(OPEN3D_TO_MUJOCO_AXIS_SWAP))
    if axis_determinant <= 0.0:
        raise RuntimeError(
            "axis correspondence check failed: python-to-mujoco transform mirrors the scene "
            f"(determinant {axis_determinant:.1f})"
        )

    mujoco.mj_forward(model, data)
    max_error = 0.0
    checked = 0

    print("validating python -> mujoco axis correspondence...")
    print("axis mapping: python/open3d (x, y/up, z) -> mujoco (x, -z, y/up)")
    print(f"axis transform determinant: {axis_determinant:.1f}")

    for record in export_records:
        if record.get("geom_type") != "mesh":
            continue

        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            record["name"],
        )
        if geom_id < 0:
            raise RuntimeError(f"axis check failed: geom missing in mujoco model: {record['name']}")

        expected_vertices = np.asarray(record["vertices_mujoco"], dtype=np.float64)
        actual_vertices = compiled_mesh_global_vertices(model, data, geom_id)

        expected_min = expected_vertices.min(axis=0)
        expected_max = expected_vertices.max(axis=0)
        actual_min = actual_vertices.min(axis=0)
        actual_max = actual_vertices.max(axis=0)
        bounds_error = float(np.max(np.abs(
            np.concatenate([actual_min - expected_min, actual_max - expected_max])
        )))
        max_error = max(max_error, bounds_error)
        checked += 1

    if checked == 0:
        print("axis correspondence check skipped: no mesh geoms exported")
        return

    print(f"axis correspondence checked geoms: {checked}")
    print(f"axis correspondence max bounds error: {max_error:.6f} m")
    if max_error > AXIS_CHECK_TOLERANCE_M:
        raise RuntimeError(
            "axis correspondence check failed: compiled mujoco mesh bounds do not match "
            f"the exported python bounds within {AXIS_CHECK_TOLERANCE_M:.3f} m"
        )
    print("axis correspondence check passed")


def configure_decomposition_for_export():
    decomposition.SPLIT_OBJECTS_INTO_SMALLER_BOXES = not EXPORT_SINGLE_BOX_PER_OBJECT
    decomposition.REMOVE_TABLE_IN_DECOMPOSITION = REMOVE_TABLE_IN_MUJOCO_EXPORT
    decomposition.INCLUDE_TABLE_IN_DECOMPOSITION = INCLUDE_TABLE_IN_MUJOCO_EXPORT
    decomposition.MERGE_OCCLUSIONS_WITH_OBJECTS = MERGE_OCCLUSIONS_WITH_OBJECTS_FOR_EXPORT
    decomposition.BUILD_OCCLUSION_ZONE_BOXES = EXPORT_OCCLUSION_ZONE_BOXES
    decomposition.SHOW_BLIND_SPOTS = EXPORT_BLIND_SPOTS
    decomposition.CONVEX_HULL_MARGIN_M = CONVEX_HULL_MARGIN_M


def load_sample_for_export(dataset, sample_index):
    configure_decomposition_for_export()
    result = decomposition.draft_process_sample(dataset, sample_index)
    xml_path, json_path, export_records = save_export_files(result)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    validate_axis_correspondence(model, data, export_records)
    apply_robot_pose_from_json(model, data, result["sample_path"])
    return result, xml_path, json_path, model, data


def update_side_rgb_window(sample_path, sample_index, state):
    # this is a reference image only; the export itself stays depth-derived.
    image_path = os.path.join(sample_path, decomposition.SIDE_RGB_FILE_NAME)
    print("side rgb reference:", image_path)

    if not SHOW_SIDE_RGB_IMAGE:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/azurion_matplotlib")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if not os.path.exists(image_path):
        print("side rgb image missing:", image_path)
        return

    image = mpimg.imread(image_path)
    title = f"side rgb reference - sample_{sample_index:04d}"

    if state.get("figure") is None:
        plt.ion()
        figure, axis = plt.subplots(num="side rgb reference")
        artist = axis.imshow(image)
        axis.axis("off")
        axis.set_title(title)
        state["figure"] = figure
        state["axis"] = axis
        state["artist"] = artist
        plt.show(block=False)
    else:
        state["artist"].set_data(image)
        state["axis"].set_title(title)
        state["figure"].canvas.draw_idle()

    plt.pause(0.001)


def update_open3d_window(vis, state, geometries):
    if not SHOW_OPEN3D_DECOMPOSITION:
        return

    for geometry in state.get("open3d_geometries", []):
        vis.remove_geometry(geometry, reset_bounding_box=False)

    state["open3d_geometries"] = geometries
    for geometry in geometries:
        vis.add_geometry(geometry, reset_bounding_box=True)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.line_width = 2.0
    vis.poll_events()
    vis.update_renderer()


def run_mujoco_viewer_loop(dataset):
    if SHOW_OPEN3D_DECOMPOSITION:
        print("warning: Open3D viewer is enabled, but it can crash under mjpython on macos.")

    state = {
        "sample_index": DEFAULT_SAMPLE_INDEX,
        "side_rgb": {},
        "open3d_geometries": [],
        "requested_delta": None,
        "viewer": None,
    }

    open3d_vis = None
    if SHOW_OPEN3D_DECOMPOSITION:
        import open3d as o3d

        open3d_vis = o3d.visualization.Visualizer()
        open3d_vis.create_window(window_name="decomposition view")

    def request_sample_delta(delta):
        state["requested_delta"] = delta
        if state.get("viewer") is not None:
            state["viewer"].close()

    def key_callback(keycode):
        # glfw right/left arrows are 262/263, matching the decomposition draft.
        if keycode == RIGHT_ARROW_KEY:
            request_sample_delta(1)
        elif keycode == LEFT_ARROW_KEY:
            request_sample_delta(-1)
        elif keycode == SAVE_KEY:
            sample_name = os.path.basename(dataset.List_of_Samples[state["sample_index"]])
            export_voxels_from_mujoco(model, data, sample_name)

    keep_running = True
    while keep_running:
        result, _, _, model, data = load_sample_for_export(dataset, state["sample_index"])
        update_side_rgb_window(result["sample_path"], state["sample_index"], state["side_rgb"])
        if open3d_vis is not None:
            update_open3d_window(open3d_vis, state, result["geometries"])

        state["requested_delta"] = None
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            state["viewer"] = viewer
            print("right arrow: next sample")
            print("left arrow: previous sample")

            while viewer.is_running():
                if open3d_vis is not None:
                    open3d_vis.poll_events()
                    open3d_vis.update_renderer()

                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(MUJOCO_TIMESTEP)

        state["viewer"] = None
        if state["requested_delta"] is None:
            keep_running = False
        else:
            state["sample_index"] = (
                state["sample_index"] + state["requested_delta"]
            ) % len(dataset.List_of_Samples)
            print("loading sample index:", state["sample_index"])

    if open3d_vis is not None:
        open3d_vis.destroy_window()


def run_export_only(dataset):
    result, xml_path, json_path, model, data = load_sample_for_export(dataset, DEFAULT_SAMPLE_INDEX)
    print("mujoco model loaded successfully:", model.ngeom, "geoms")
    print("export-only mode wrote:", xml_path)
    print("export-only mode wrote:", json_path)


def run_export_all(dataset):
    print(f"Starting bulk export for {len(dataset.samples)} samples...")
    for i, sample in enumerate(dataset.samples):
        print(f"\n--- [{i+1}/{len(dataset.samples)}] Exporting {sample.relative_path} ---")
        try:
            result, xml_path, json_path, model, data = load_sample_for_export(dataset, i)
            export_dir = os.path.join(EXPORT_FOLDER, sample.relative_path)
            export_voxels_from_mujoco(model, data, sample.sample_name, export_dir=export_dir)
        except Exception as e:
            print(f"ERROR: Failed to export {sample.relative_path}. Reason: {e}")
    print("\nBulk export completed.")

def export_voxels_from_mujoco(
    model,
    data,
    sample_name,
    voxel_size=0.05,
    export_folder="mujoco_exports",
    export_dir=None,
):
    """Export a deterministic solid-ish voxelization of MuJoCo mesh geoms."""
    mujoco.mj_forward(model, data)
    meshes = []
    combined_mesh = o3d.geometry.TriangleMesh()

    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[i]
        v_start = model.mesh_vertadr[mesh_id]
        v_num = model.mesh_vertnum[mesh_id]
        f_start = model.mesh_faceadr[mesh_id]
        f_num = model.mesh_facenum[mesh_id]

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(
            model.mesh_vert[v_start:v_start + v_num] @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
        )
        sub_mesh.triangles = o3d.utility.Vector3iVector(model.mesh_face[f_start:f_start + f_num])
        sub_mesh = sub_mesh.subdivide_midpoint(number_of_iterations=2)
        meshes.append(sub_mesh)
        combined_mesh += sub_mesh

    if not meshes:
        print(f"CRITICAL: No meshes found for {sample_name}.")
        return None, None

    # 4. Create Voxel Grid (Solid Volume via Raycasting)
    # We query the interior to ensure the loss function sees a solid ground truth.

    room_origin_o3d = np.array([-decomposition.default_room_dimensions[0]/2, 0.0, -decomposition.default_room_dimensions[2]/2])
    
    # Get mesh bounds and snap to the global grid to prevent sub-voxel drift
    min_b = combined_mesh.get_min_bound() - voxel_size
    max_b = combined_mesh.get_max_bound() + voxel_size
    
    # Map MJ bounds to O3D bounds correctly
    min_b_o3d = mujoco_to_open3d_vector(min_b)
    max_b_o3d = mujoco_to_open3d_vector(max_b)
    
    # Use min/max properly since O3D mapping might flip axes
    true_min_o3d = np.minimum(min_b_o3d, max_b_o3d)
    true_max_o3d = np.maximum(min_b_o3d, max_b_o3d)
    
    min_idx = np.floor((true_min_o3d - room_origin_o3d) / voxel_size).astype(int)
    max_idx = np.ceil((true_max_o3d - room_origin_o3d) / voxel_size).astype(int)

    xi = np.arange(min_idx[0], max_idx[0] + 1)
    yi = np.arange(min_idx[1], max_idx[1] + 1)
    zi = np.arange(min_idx[2], max_idx[2] + 1)
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    
    centers_o3d = np.stack([
        gx.ravel() * voxel_size + room_origin_o3d[0] + voxel_size * 0.5,
        gy.ravel() * voxel_size + room_origin_o3d[1] + voxel_size * 0.5,
        gz.ravel() * voxel_size + room_origin_o3d[2] + voxel_size * 0.5,
    ], axis=1).astype(np.float64)

    # Map the exact Open3D room grid into MuJoCo space for testing
    centers_mj = np.array([open3d_vector_to_mujoco(c) for c in centers_o3d])
    
    is_filled = np.zeros(len(centers_mj), dtype=bool)
    touch_margin = (voxel_size * np.sqrt(3) / 2) + 0.002  # half-diagonal + epsilon

    # Fix: Evaluate raycasting per-mesh instead of one combined scene.
    # Open3D's raycasting computes winding numbers which can cancel out and create hollow
    # empty cores when multiple convex hulls intersect each other.
    for sub_mesh in meshes:
        min_b_sub = sub_mesh.get_min_bound() - touch_margin
        max_b_sub = sub_mesh.get_max_bound() + touch_margin
        
        in_box = (
            (centers_mj[:, 0] >= min_b_sub[0]) & (centers_mj[:, 0] <= max_b_sub[0]) &
            (centers_mj[:, 1] >= min_b_sub[1]) & (centers_mj[:, 1] <= max_b_sub[1]) &
            (centers_mj[:, 2] >= min_b_sub[2]) & (centers_mj[:, 2] <= max_b_sub[2])
        )
        
        if not np.any(in_box):
            continue
            
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sub_mesh))
        
        box_queries = o3d.core.Tensor(centers_mj[in_box].astype(np.float32))
        occupancy = scene.compute_occupancy(box_queries).numpy().astype(bool)
        signed_distance = scene.compute_signed_distance(box_queries).numpy()
        
        is_filled[in_box] |= occupancy | (signed_distance <= touch_margin)

    inside_points_o3d = centers_o3d[is_filled] + MANUAL_DIAGNOSTIC_SHIFT
    
    num_voxels = len(inside_points_o3d)
    
    if num_voxels == 0:
        print(f"ERROR: Voxelizer produced 0 voxels for {sample_name}.")
        return None, None

    if export_dir is None:
        export_dir = os.path.join(export_folder, sample_name)
    os.makedirs(export_dir, exist_ok=True)

    # Re-calculate indices relative to the fixed global Room Origin
    indices = np.floor((inside_points_o3d - room_origin_o3d) / voxel_size).astype(np.int32)
    
    np.save(os.path.join(export_dir, f"{sample_name}_voxel_indices.npy"), indices)

    pcd_export = o3d.geometry.PointCloud()
    pcd_export.points = o3d.utility.Vector3dVector(inside_points_o3d)
    
    success = o3d.io.write_point_cloud(os.path.join(export_dir, f"{sample_name}_voxels.ply"), pcd_export)
    
    if success:
        print(f"Successfully exported {len(inside_points_o3d)} voxels to {export_dir}")
    else:
        print("ERROR: Failed to write .ply file.")

    return indices, None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    configure_decomposition_for_export()

    dataset = decomposition.Dataset(DATASET_FOLDER)
    if not dataset.List_of_Samples:
        raise RuntimeError("no sample folders found")
    
    if EXPORT_ALL:
        run_export_all(dataset)
        return


    if os.environ.get("MUJOCO_EXPORT_NO_VIEW") == "1":
        run_export_only(dataset)
    else:
        run_mujoco_viewer_loop(dataset)


if __name__ == "__main__":
    main()
