"""Beginner-facing command line entrypoint for the pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_SAMPLE_IDS = (3, 9, 21, 27, 36)


def _capture_root(mode: str) -> Path:
    if mode == "demo":
        return PROJECT_ROOT / "DepthCaptures_demo"
    if mode == "full":
        return PROJECT_ROOT / "DepthCaptures"
    raise ValueError("mode must be 'demo' or 'full'")


def _default_run_id(mode: str, noise: str) -> str:
    if mode == "demo":
        return f"demo_{noise}_20"
    return f"full_{noise}_160"


def _sample_indices_arg(sample: str | None, mode: str) -> str | None:
    if sample:
        return str(int(sample))
    if mode == "demo":
        return ",".join(str(index) for index in DEMO_SAMPLE_IDS)
    return None


def _pipeline_args(args: argparse.Namespace, mode: str) -> SimpleNamespace:
    noise = getattr(args, "noise", None) or "robust"
    return SimpleNamespace(
        capture_root=str(_capture_root(mode)),
        rig_id=getattr(args, "rig", None),
        layout=getattr(args, "layout", None),
        sample_indices=_sample_indices_arg(getattr(args, "sample", None), mode),
        run_id=getattr(args, "run_id", None) or _default_run_id(mode, noise),
        noise_profile=noise,
        noise_seed=getattr(args, "noise_seed", None),
        skip_camera_metrics=getattr(args, "skip_camera_metrics", False),
        skip_mujoco_metrics=False,
        skip_planning=True,
        skip_figures=getattr(args, "skip_figures", False),
        no_plan_smoke=True,
        ordinal=0,
    )


def _build_config(args: argparse.Namespace, mode: str):
    from src.config import default_config
    from src.pipeline import apply_cli_overrides

    config = apply_cli_overrides(default_config(), _pipeline_args(args, mode))
    config.metrics_camera_rig.enabled = False
    config.planning.enabled = False
    return config


def _discover_one_record(config):
    from src import dataset_io

    records = dataset_io.discover_samples(config)
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one sample after filters, found {len(records)}")
    dataset = dataset_io.make_source_dataset(config)
    return dataset, records[0]


def _load_manifest(run_id: str) -> tuple[Path, dict]:
    run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return run_dir, json.load(handle)


def _find_manifest_sample(manifest: dict, rig: str, sample: str) -> tuple[int, dict]:
    sample_id = int(sample)
    matches = []
    for ordinal, result in enumerate(manifest.get("sample_results", [])):
        sample_info = result.get("sample", {})
        if sample_info.get("rig_id") == rig and int(sample_info.get("sample_index", -1)) == sample_id:
            matches.append((ordinal, result))
    if not matches:
        raise FileNotFoundError(f"sample {sample_id:04d} with rig {rig!r} is not in the selected run")
    if len(matches) > 1:
        raise RuntimeError(f"run manifest has multiple matches for sample {sample_id:04d} and rig {rig!r}")
    return matches[0]


def _exported_paths(run_dir: Path, sample_result: dict) -> tuple[Path, Path, Path]:
    sample = sample_result["sample"]
    export_dir = run_dir / "mujoco_exports" / sample["relative_path"]
    sample_name = sample["sample_name"]
    return (
        export_dir / f"{sample_name}_mujoco.xml",
        export_dir / f"{sample_name}_boxes.json",
        export_dir / f"{sample_name}_voxels.ply",
    )


def _manifest_noise(manifest: dict, default: str) -> str:
    noise = manifest.get("config", {}).get("noise", {})
    profile = noise.get("profile")
    return str(profile or default)


def _decomposition_view_path(run_dir: Path, sample: dict, sample_result: dict) -> Path:
    from src import visualization_artifacts

    stored = sample_result.get("decomposition_view_path")
    if stored:
        return Path(stored)
    return visualization_artifacts.decomposition_view_path(
        run_dir,
        sample["relative_path"],
        sample["sample_name"],
    )


def _recompute_decomposition_view(args: argparse.Namespace, manifest: dict, run_dir: Path, sample: dict, view_path: Path) -> Path:
    from src import decomposition, visualization_artifacts

    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.noise = _manifest_noise(manifest, getattr(args, "noise", "robust"))
    config = _build_config(fallback_args, args.mode)
    try:
        dataset, record = _discover_one_record(config)
    except Exception as exc:
        raise RuntimeError(
            "This run does not contain a saved decomposition view artifact, and the source "
            "DepthCaptures data needed to recreate it is unavailable or does not match the selected sample."
        ) from exc
    if record.relative_path != sample["relative_path"]:
        raise RuntimeError(
            f"fallback recompute selected {record.relative_path}, but the run sample is {sample['relative_path']}"
        )
    result = decomposition.process_sample(dataset, record.index, config)
    visualization_artifacts.save_decomposition_view(view_path, result.get("geometries", []) or [])
    print(f"Recreated missing decomposition view artifact: {view_path.relative_to(run_dir)}")
    return view_path


def command_run_demo(args: argparse.Namespace) -> int:
    from src import pipeline

    return pipeline.command_bulk(_pipeline_args(args, "demo"))


def command_run_full(args: argparse.Namespace) -> int:
    from src import pipeline

    return pipeline.command_bulk(_pipeline_args(args, "full"))


def command_inspect_config(args: argparse.Namespace) -> int:
    from src import pipeline

    return pipeline.command_inspect_config(_pipeline_args(args, getattr(args, "mode", "demo")))


def command_view_voxels(args: argparse.Namespace) -> int:
    import open3d as o3d

    from src import visualization_artifacts

    run_id = args.run_id or _default_run_id(args.mode, args.noise)
    run_dir, manifest = _load_manifest(run_id)
    _ordinal, sample_result = _find_manifest_sample(manifest, args.rig, args.sample)
    sample = sample_result["sample"]
    view_path = _decomposition_view_path(run_dir, sample, sample_result)
    if not view_path.exists():
        view_path = _recompute_decomposition_view(args, manifest, run_dir, sample, view_path)
    geoms = visualization_artifacts.load_decomposition_view(view_path)
    if args.no_occlusions:
        geoms = visualization_artifacts.hide_occlusion_geometries(geoms)
    if not geoms:
        raise RuntimeError(f"no decomposition view geometries were found in {view_path}")
    print(f"Opening voxel/decomposition view: {run_id} / {sample['relative_path']}")
    o3d.visualization.draw_geometries(geoms, window_name=f"{run_id}: {sample['relative_path']}")
    return 0


def command_view_pointcloud(args: argparse.Namespace) -> int:
    from src import decomposition, pointcloud_viewer

    config = _build_config(args, args.mode)
    dataset, record = _discover_one_record(config)
    result = decomposition.process_sample(dataset, record.index, config)
    point_cloud = result.get("room_point_cloud")
    if point_cloud is None or point_cloud.is_empty():
        raise RuntimeError("the reconstructed point cloud is empty")
    print(f"Opening point cloud view: {record.relative_path}")
    pointcloud_viewer.draw_height_colored(point_cloud, window_name=f"Point cloud: {record.relative_path}")
    return 0


def _find_mjpython() -> str | None:
    sibling = Path(sys.executable).with_name("mjpython")
    if sibling.exists():
        return str(sibling)
    return shutil.which("mjpython")


def _relaunch_macos_mujoco_viewer_with_mjpython() -> None:
    if sys.platform != "darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    if os.environ.get("ENV_AWARENESS_MJPYTHON_REEXEC") == "1":
        raise RuntimeError(
            "MuJoCo's macOS viewer still is not running under mjpython after relaunch. "
            "Run `python .venv/bin/mjpython main.py view-mujoco ...` from the repository root."
        )

    mjpython = _find_mjpython()
    if mjpython is None:
        raise RuntimeError(
            "MuJoCo's macOS viewer requires mjpython, but it was not found. "
            "Activate the virtual environment, reinstall requirements, and rerun this command. "
            "Direct fallback: `python .venv/bin/mjpython main.py view-mujoco ...`."
        )

    command = [sys.executable, mjpython, str(Path(__file__).resolve()), *sys.argv[1:]]
    env = os.environ.copy()
    env["ENV_AWARENESS_MJPYTHON_REEXEC"] = "1"
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"macOS MuJoCo viewer requires mjpython; relaunching with: {printable}", flush=True)
    os.execve(sys.executable, command, env)


def _open_mujoco(xml_path: Path) -> None:
    _relaunch_macos_mujoco_viewer_with_mjpython()

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        print("MuJoCo viewer open. Close the viewer window to return to the terminal.")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


def command_view_mujoco(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id(args.mode, args.noise)
    run_dir, manifest = _load_manifest(run_id)
    _ordinal, sample_result = _find_manifest_sample(manifest, args.rig, args.sample)
    xml_path, _json_path, _ply_path = _exported_paths(run_dir, sample_result)
    if not xml_path.exists():
        raise FileNotFoundError(f"missing MuJoCo XML: {xml_path}")

    print("Planner attempt: skipped unless this run was exported with INCLUDE_ROBOT=True.")
    _open_mujoco(xml_path)
    return 0


def command_recompute_temp(args: argparse.Namespace) -> int:
    import open3d as o3d

    from src import decomposition, pipeline, pointcloud_viewer, visualization_artifacts

    config = _build_config(args, args.mode)
    dataset, record = _discover_one_record(config)
    with tempfile.TemporaryDirectory(prefix="env_awareness_") as tmp:
        tmp_path = Path(tmp)
        if args.stage == "pointcloud":
            result = decomposition.process_sample(dataset, record.index, config)
            point_cloud = result.get("room_point_cloud")
            if args.view and point_cloud is not None and not point_cloud.is_empty():
                pointcloud_viewer.draw_height_colored(
                    point_cloud,
                    window_name=f"Temporary point cloud: {record.relative_path}",
                )
            print(f"Temporary point-cloud recompute complete for {record.relative_path}")
            return 0

        sample_result = pipeline.export_one_sample(dataset, record, config, tmp_path / "mujoco_exports")
        if sample_result.status != "success":
            raise RuntimeError(sample_result.error or "temporary export failed")
        print(f"Temporary export complete for {record.relative_path}")
        print(f"Stage counts: {sample_result.stage_counts}")
        if args.stage == "voxels" and args.view:
            view_path = sample_result.decomposition_view_path or visualization_artifacts.decomposition_view_path(
                tmp_path,
                record.relative_path,
                record.sample_name,
            )
            geoms = visualization_artifacts.load_decomposition_view(Path(view_path))
            if args.no_occlusions:
                geoms = visualization_artifacts.hide_occlusion_geometries(geoms)
            if geoms:
                o3d.visualization.draw_geometries(geoms, window_name=f"Temporary voxels: {record.relative_path}")
        if args.stage == "mujoco" and args.view and sample_result.xml_path is not None:
            _open_mujoco(sample_result.xml_path)
    return 0


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rig", default=None, help="optional rig filter, for example 4CamAsym")
    parser.add_argument("--sample", default=None, help="optional sample id, for example 0027")
    parser.add_argument("--layout", default=None, help="optional layout filter")
    parser.add_argument("--noise", choices=("robust", "none"), default="robust")
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-camera-metrics", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")


def _add_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rig", required=True, help="camera rig id, for example 4CamAsym")
    parser.add_argument("--sample", required=True, help="sample id, for example 0027")
    parser.add_argument("--mode", choices=("demo", "full"), default="full")
    parser.add_argument("--noise", choices=("robust", "none"), default="robust")
    parser.add_argument("--run-id", default=None)


def _add_occlusion_view_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-occlusions",
        action="store_true",
        help="hide occlusion and blind-spot geometry in voxel/decomposition viewers only",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Environmental Awareness Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_demo = subparsers.add_parser("run-demo", help="run the committed 20-sample demo dataset")
    _add_common_run_args(run_demo)
    run_demo.set_defaults(func=command_run_demo)

    run_full = subparsers.add_parser("run-full", help="run the full local DepthCaptures dataset")
    _add_common_run_args(run_full)
    run_full.set_defaults(func=command_run_full)

    inspect = subparsers.add_parser("inspect-config", help="print effective pipeline config")
    inspect.add_argument("--mode", choices=("demo", "full"), default="demo")
    _add_common_run_args(inspect)
    inspect.set_defaults(func=command_inspect_config)

    view_pointcloud = subparsers.add_parser("view-pointcloud", help="reconstruct and view a point cloud for one sample")
    _add_view_args(view_pointcloud)
    view_pointcloud.set_defaults(func=command_view_pointcloud)

    view_voxels = subparsers.add_parser("view-voxels", help="view existing voxel and hull export for one sample")
    _add_view_args(view_voxels)
    _add_occlusion_view_arg(view_voxels)
    view_voxels.set_defaults(func=command_view_voxels)

    view_mujoco = subparsers.add_parser("view-mujoco", help="open an existing MuJoCo XML export for one sample")
    _add_view_args(view_mujoco)
    view_mujoco.set_defaults(func=command_view_mujoco)

    recompute = subparsers.add_parser("recompute-temp", help="temporarily recompute one sample without saving outputs")
    _add_view_args(recompute)
    recompute.add_argument("--stage", choices=("pointcloud", "voxels", "mujoco"), required=True)
    recompute.add_argument("--view", action="store_true", help="open the relevant native viewer after recomputing")
    _add_occlusion_view_arg(recompute)
    recompute.set_defaults(func=command_recompute_temp)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
