"""Canonical Unity capture dataset helpers for the Azurion Python pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Iterable

import numpy as np


DEFAULT_CAPTURE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DepthCaptures")
DEFAULT_NEAR_CLIP = 0.3
DEFAULT_FAR_CLIP = 8.3
FAR_CLIP_EPSILON = 0.01
SAMPLE_RE = re.compile(r"^sample_(\d+)$")
ROBOT_JOINT_NAMES = {"Long", "Z1Rot", "Z2Rot", "Prop", "CArc"}
_DEPTH_CLIP_CACHE: dict[str, tuple[float, float]] = {}


@dataclass(frozen=True)
class SampleRef:
    path: str
    sample_name: str
    sample_index: int | None
    rig_id: str | None
    layout: str | None
    relative_path: str


@dataclass(frozen=True)
class DepthCamera:
    index: int
    name: str
    depth_path: str
    fx: float
    fy: float
    cx: float
    cy: float
    near_clip: float
    far_clip: float
    intrinsic: object
    extrinsic: np.ndarray
    metadata: dict


def default_capture_root() -> str:
    return os.environ.get("AZURION_CAPTURE_FOLDER", DEFAULT_CAPTURE_ROOT)


def _sample_index_from_name(name: str) -> int | None:
    match = SAMPLE_RE.match(name)
    if match:
        return int(match.group(1))
    return None


def is_sample_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "depth_metadata.json"))
    )


def _metadata_json(path: str, filename: str) -> dict:
    with open(os.path.join(path, filename), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_rig_layout(root: str, sample_path: str, metadata: dict | None = None) -> tuple[str | None, str | None]:
    rig_id = metadata.get("cameraRigId") if metadata else None
    layout = None
    rel_parts = os.path.relpath(sample_path, root).split(os.sep)
    if len(rel_parts) >= 3:
        rig_id = rig_id or rel_parts[-3]
        layout = rel_parts[-2]
    elif len(rel_parts) >= 2:
        layout = rel_parts[-2]
    else:
        parent = os.path.basename(os.path.dirname(sample_path))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(sample_path)))
        if parent and not parent.startswith("R"):
            layout = parent
        if not rig_id and grandparent.startswith("R"):
            rig_id = grandparent
                
    RIG_NAME_MAP = {
        "R0_LegacyCorner4": "4CamClassic",
        "R1_MinimalTriad3": "3Cam",
        "R2_HighWallCross4": "4CamAsym",
        "R3_HybridDense5": "5Cam",
    }
    rig_id = RIG_NAME_MAP.get(rig_id, rig_id)

    return rig_id, layout


def _sample_sort_key(sample: SampleRef):
    return (
        sample.rig_id or "",
        sample.layout or "",
        sample.sample_index if sample.sample_index is not None else 10**9,
        sample.relative_path,
    )


def discover_samples(
    root: str | None = None,
    rig_id: str | None = None,
    layout: str | None = None,
) -> list[SampleRef]:
    """Discover sample folders under the final nested Unity export structure.

    The input may be the full DepthCaptures root, a rig folder, a layout folder,
    or one specific sample folder. Environment filters are applied after
    discovery so old scripts can be pointed at the full root safely.
    """
    root = os.path.abspath(root or default_capture_root())
    if not os.path.isdir(root):
        raise NotADirectoryError(root)

    env_rig = os.environ.get("AZURION_RIG_ID")
    env_layout = os.environ.get("AZURION_LAYOUT")
    rig_filter = rig_id or env_rig
    layout_filter = layout or env_layout

    sample_paths: list[str] = []
    if is_sample_dir(root):
        sample_paths = [root]
        discovery_root = os.path.dirname(root)
    else:
        discovery_root = root
        for current_root, dirnames, _ in os.walk(root):
            for dirname in dirnames:
                candidate = os.path.join(current_root, dirname)
                if dirname.startswith("sample_") and is_sample_dir(candidate):
                    sample_paths.append(candidate)

    samples: list[SampleRef] = []
    for sample_path in sample_paths:
        metadata = _metadata_json(sample_path, "depth_metadata.json")
        rig, sample_layout = _infer_rig_layout(discovery_root, sample_path, metadata)
        if rig_filter and rig != rig_filter:
            continue
        if layout_filter and sample_layout != layout_filter:
            continue

        sample_name = os.path.basename(sample_path)
        samples.append(SampleRef(
            path=sample_path,
            sample_name=sample_name,
            sample_index=_sample_index_from_name(sample_name),
            rig_id=rig,
            layout=sample_layout,
            relative_path=os.path.relpath(sample_path, discovery_root),
        ))

    samples.sort(key=_sample_sort_key)
    return samples


def iter_samples(root: str | None = None, **filters) -> Iterable[SampleRef]:
    return iter(discover_samples(root, **filters))


def get_sample(index_or_ref=0, root: str | None = None, **filters) -> SampleRef:
    samples = discover_samples(root, **filters)
    if not samples:
        raise FileNotFoundError(f"no Unity sample folders found under {os.path.abspath(root or default_capture_root())}")
    if isinstance(index_or_ref, SampleRef):
        return index_or_ref
    if isinstance(index_or_ref, int):
        return samples[index_or_ref]
    requested = str(index_or_ref)
    for sample in samples:
        if requested in {sample.path, sample.sample_name, sample.relative_path}:
            return sample
    raise KeyError(f"sample not found: {index_or_ref}")


def _matrix_from_camera(camera: dict) -> np.ndarray:
    if "worldToCameraMatrix" in camera:
        extrinsic = np.array(camera["worldToCameraMatrix"], dtype=np.float64).reshape(4, 4)
        if "cameraToWorldMatrix" in camera:
            camera_to_world = np.array(camera["cameraToWorldMatrix"], dtype=np.float64).reshape(4, 4)
            if not np.allclose(extrinsic @ camera_to_world, np.eye(4), atol=1e-3):
                print(f"warning: camera matrices are not exact inverses for {camera.get('name', camera.get('index'))}")
        return extrinsic
    if "cameraToWorldMatrix" in camera:
        return np.linalg.inv(np.array(camera["cameraToWorldMatrix"], dtype=np.float64).reshape(4, 4))
    raise KeyError("camera metadata needs worldToCameraMatrix or cameraToWorldMatrix")


def load_depth_metadata(sample: SampleRef | str) -> dict:
    path = sample.path if isinstance(sample, SampleRef) else sample
    return _metadata_json(path, "depth_metadata.json")


def depth_clip_for_raw_path(raw_path: str) -> tuple[float, float]:
    """Return Unity near/far clip values for one depth raw.

    Depth pixels at the far clip are no-hit background. They must not become
    point-cloud geometry or they create camera-facing walls.
    """
    abs_raw = os.path.abspath(raw_path)
    cached = _DEPTH_CLIP_CACHE.get(abs_raw)
    if cached is not None:
        return cached

    near_clip = DEFAULT_NEAR_CLIP
    far_clip = DEFAULT_FAR_CLIP
    sample_dir = os.path.dirname(abs_raw)
    try:
        metadata = load_depth_metadata(sample_dir)
        near_clip = float(metadata.get("nearClip", near_clip))
        far_clip = float(metadata.get("farClip", far_clip))
        raw_name = os.path.basename(abs_raw)
        raw_files = [os.path.basename(path) for path in metadata.get("fullDepthRawFiles", [])]
        camera_index = raw_files.index(raw_name) if raw_name in raw_files else None
        if camera_index is not None:
            for camera in metadata.get("cameras", []):
                if int(camera.get("index", -1)) == camera_index:
                    near_clip = float(camera.get("nearClip", near_clip))
                    far_clip = float(camera.get("farClip", far_clip))
                    break
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass

    result = (near_clip, far_clip)
    _DEPTH_CLIP_CACHE[abs_raw] = result
    return result


def valid_depth_mask(depth: np.ndarray, near_clip: float, far_clip: float) -> np.ndarray:
    max_depth = max(float(near_clip), float(far_clip) - FAR_CLIP_EPSILON)
    return np.isfinite(depth) & (depth > float(near_clip)) & (depth < max_depth)


def load_depth_cameras(sample: SampleRef | str) -> list[DepthCamera]:
    """Load camera paths, Open3D intrinsics, and Unity-exported extrinsics."""
    import open3d as o3d

    path = sample.path if isinstance(sample, SampleRef) else sample
    metadata = load_depth_metadata(path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    raw_files = metadata.get("fullDepthRawFiles", [])

    cameras: list[DepthCamera] = []
    for camera in metadata.get("cameras", []):
        index = int(camera["index"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        near_clip = float(camera.get("nearClip", metadata.get("nearClip", DEFAULT_NEAR_CLIP)))
        far_clip = float(camera.get("farClip", metadata.get("farClip", DEFAULT_FAR_CLIP)))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        depth_file = raw_files[index] if index < len(raw_files) else f"cam{index}_depth.raw"
        cameras.append(DepthCamera(
            index=index,
            name=str(camera.get("name", f"cam{index}")),
            depth_path=os.path.join(path, depth_file),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near_clip=near_clip,
            far_clip=far_clip,
            intrinsic=intrinsic,
            extrinsic=_matrix_from_camera(camera),
            metadata=camera,
        ))
    return cameras


def load_robot_state(sample: SampleRef | str) -> dict:
    path = sample.path if isinstance(sample, SampleRef) else sample
    state_path = os.path.join(path, "robot_state.json")
    if os.path.exists(state_path):
        return _metadata_json(path, "robot_state.json")
    legacy_path = os.path.join(path, "robot_pose.json")
    if os.path.exists(legacy_path):
        return _metadata_json(path, "robot_pose.json")
    raise FileNotFoundError(f"robot_state.json not found under {path}")


def load_scene_objects(sample: SampleRef | str) -> dict:
    path = sample.path if isinstance(sample, SampleRef) else sample
    return _metadata_json(path, "scene_objects.json")


def load_unity_voxels(sample: SampleRef | str, kind: str = "scene") -> tuple[dict, np.ndarray]:
    path = sample.path if isinstance(sample, SampleRef) else sample
    if kind not in {"scene", "props", "robot"}:
        raise ValueError("kind must be one of: scene, props, robot")

    if kind == "scene":
        metadata_name = "voxel_scene_metadata.json"
        raw_name = "voxel_scene_occupancy.raw"
    elif kind == "props":
        metadata_name = "voxel_metadata.json"
        raw_name = "voxel_props_occupancy.raw"
    else:
        metadata_name = "voxel_robot_metadata.json"
        raw_name = "voxel_robot_occupancy.raw"

    metadata = _metadata_json(path, metadata_name)
    raw_path = os.path.join(path, metadata.get("fileName", raw_name))
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
    # Unity writes x fastest, then z, then y. Internal Python grids use (x, y, z).
    return flat.reshape((size_x, size_z, size_y), order="F").transpose(0, 2, 1)
