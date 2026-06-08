"""CLI orchestration for the modular noise-aware pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import PipelineConfig, default_config
from .types import PlanningRunResult, RunManifest, SampleRecord, SampleRunResult


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_run_log(run_dir: Path):
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, handle)
        sys.stderr = Tee(original_stderr, handle)
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _run_id(label: str, config: PipelineConfig) -> str:
    if config.output.run_id:
        return config.output.run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    return f"{timestamp}_{clean_label}"


def make_run_dir(config: PipelineConfig, label: str) -> Path:
    run_id = _run_id(label, config)
    config.output.run_id = run_id
    run_dir = config.output.root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if getattr(args, "capture_root", None):
        config.dataset.capture_root = Path(args.capture_root).resolve()
    if getattr(args, "rig_id", None):
        config.dataset.rig_id = args.rig_id
    if getattr(args, "layout", None):
        config.dataset.layout = args.layout
    if hasattr(args, "sample_indices"):
        config.dataset.sample_indices = _parse_indices(args.sample_indices)
    if getattr(args, "run_id", None):
        config.output.run_id = args.run_id
    if getattr(args, "noise_profile", None):
        config.noise.profile = args.noise_profile
    if getattr(args, "noise_seed", None) is not None:
        config.noise.global_seed = int(args.noise_seed)
    if getattr(args, "skip_camera_metrics", False):
        config.metrics_camera_rig.enabled = False
    # Deprecated compatibility flag. Collision-geometry metrics are generated
    # separately from exported PLY files to preserve Robert2 semantics.
    _ = getattr(args, "skip_mujoco_metrics", False)
    if getattr(args, "skip_planning", False):
        config.planning.enabled = False
    if getattr(args, "skip_figures", False):
        config.metrics_camera_rig.skip_figures = True
    return config


def config_to_json(config: PipelineConfig) -> str:
    return json.dumps(config.to_dict(), indent=2)


def select_sample(records: list[SampleRecord], ordinal: int) -> SampleRecord:
    if not records:
        raise FileNotFoundError("no samples matched the configured dataset filters")
    if ordinal < 0 or ordinal >= len(records):
        raise IndexError(f"sample ordinal {ordinal} out of range 0..{len(records) - 1}")
    return records[ordinal]


def export_one_sample(
    dataset,
    record: SampleRecord,
    config: PipelineConfig,
    export_root: Path,
) -> SampleRunResult:
    from . import decomposition, mujoco_export, visualization_artifacts

    try:
        result, xml_path, json_path, voxel_indices_path, voxel_ply_path, _, _ = mujoco_export.export_sample(
            dataset,
            record.index,
            config,
            export_root,
        )
        run_dir = export_root.parent
        decomposition_view_path = visualization_artifacts.decomposition_view_path(
            run_dir,
            record.relative_path,
            record.sample_name,
        )
        visualization_artifacts.save_decomposition_view(
            decomposition_view_path,
            result.get("geometries", []) or [],
        )
        return SampleRunResult(
            sample=record,
            status="success",
            export_dir=Path(xml_path).parent,
            xml_path=Path(xml_path),
            json_path=Path(json_path),
            voxel_indices_path=voxel_indices_path,
            voxel_ply_path=voxel_ply_path,
            decomposition_view_path=decomposition_view_path,
            noise=result.get("noise", {}) or {},
            stage_counts=decomposition.key_counts(result),
        )
    except Exception as exc:
        return SampleRunResult(
            sample=record,
            status="error",
            error=repr(exc),
        )


def write_noise_stats(run_dir: Path, sample_result: SampleRunResult) -> Path | None:
    if not sample_result.noise.get("enabled"):
        return None
    sample = sample_result.sample
    rig = sample.rig_id or "unknown_rig"
    layout = sample.layout or "unknown_layout"
    path = run_dir / "noise_stats" / rig / layout / f"{sample.sample_name}.json"
    _write_json(path, sample_result.noise)
    sample_result.noise["stats_path"] = str(path)
    return path


def run_manifest(run_id: str, run_dir: Path, config: PipelineConfig) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        run_dir=run_dir,
        config=config.to_dict(),
        source_refs=config.source_refs.__dict__.copy(),
    )


def write_manifest(manifest: RunManifest) -> Path:
    path = manifest.run_dir / "run_manifest.json"
    _write_json(path, manifest.to_dict())
    return path


def command_inspect_config(args: argparse.Namespace) -> int:
    config = apply_cli_overrides(default_config(), args)
    print(config_to_json(config))
    return 0


def command_sample(args: argparse.Namespace) -> int:
    from . import dataset_io, planning

    config = apply_cli_overrides(default_config(), args)
    os.chdir(config.output.root.parents[0])
    run_dir = make_run_dir(config, "sample")
    run_id = config.output.run_id or run_dir.name
    manifest = run_manifest(run_id, run_dir, config)

    with tee_run_log(run_dir) as log_path:
        records = dataset_io.discover_samples(config)
        dataset = dataset_io.make_source_dataset(config)
        record = select_sample(records, args.ordinal)
        sample_result = export_one_sample(dataset, record, config, run_dir / "mujoco_exports")
        write_noise_stats(run_dir, sample_result)

        manifest.sample_results.append(sample_result.to_dict())
        if sample_result.status != "success":
            manifest.failures.append({"sample": record.to_dict(), "error": sample_result.error})
        if config.planning.enabled and not args.no_plan_smoke:
            plan_result = planning.run_headless(sample_result, config)
            plan_path = run_dir / "planning" / "plan_smoke.json"
            _write_json(plan_path, plan_result.to_dict())
            manifest.planning = {"status_path": str(plan_path), **plan_result.to_dict()}

        manifest.metric_outputs["log"] = str(log_path)
        manifest_path = write_manifest(manifest)
        print(f"sample run written to {run_dir}")
        print(f"manifest: {manifest_path}")
    return 0 if sample_result.status == "success" else 1


def command_bulk(args: argparse.Namespace) -> int:
    from . import dataset_io, metrics_camera_rig, planning

    config = apply_cli_overrides(default_config(), args)
    os.chdir(config.output.root.parents[0])
    run_dir = make_run_dir(config, "bulk")
    run_id = config.output.run_id or run_dir.name
    manifest = run_manifest(run_id, run_dir, config)

    with tee_run_log(run_dir) as log_path:
        records = dataset_io.discover_samples(config)
        dataset = dataset_io.make_source_dataset(config)
        first_success: SampleRunResult | None = None
        for number, record in enumerate(records, start=1):
            print(f"[{number:03d}/{len(records):03d}] exporting {record.relative_path}")
            sample_result = export_one_sample(dataset, record, config, run_dir / "mujoco_exports")
            if sample_result.status == "success":
                if first_success is None:
                    first_success = sample_result
            else:
                manifest.failures.append({"sample": record.to_dict(), "error": sample_result.error})
                if not config.output.continue_on_sample_failure:
                    manifest.sample_results.append(sample_result.to_dict())
                    break
            write_noise_stats(run_dir, sample_result)
            manifest.sample_results.append(sample_result.to_dict())
            write_manifest(manifest)

        if config.metrics_camera_rig.enabled:
            manifest.metric_outputs["camera_rig"] = metrics_camera_rig.run(config, run_dir)
        if config.planning.enabled and first_success is not None:
            plan_result = planning.run_headless(first_success, config)
            plan_path = run_dir / "planning" / "plan_smoke.json"
            _write_json(plan_path, plan_result.to_dict())
            manifest.planning = {"status_path": str(plan_path), **plan_result.to_dict()}

        manifest.metric_outputs["log"] = str(log_path)
        manifest_path = write_manifest(manifest)
        print(f"bulk run written to {run_dir}")
        print(f"manifest: {manifest_path}")
    return 0 if not manifest.failures else 1


def command_plan_smoke(args: argparse.Namespace) -> int:
    from . import dataset_io, planning

    config = apply_cli_overrides(default_config(), args)
    os.chdir(config.output.root.parents[0])
    run_dir = make_run_dir(config, "plan_smoke")
    run_id = config.output.run_id or run_dir.name
    manifest = run_manifest(run_id, run_dir, config)

    with tee_run_log(run_dir) as log_path:
        records = dataset_io.discover_samples(config)
        dataset = dataset_io.make_source_dataset(config)
        record = select_sample(records, args.ordinal)
        sample_result = export_one_sample(dataset, record, config, run_dir / "mujoco_exports")
        plan_result: PlanningRunResult
        if sample_result.status == "success":
            plan_result = planning.run_headless(sample_result, config)
        else:
            plan_result = PlanningRunResult(
                status="skipped",
                sample=record,
                goal_poi=config.planning.goal_poi,
                message="sample export failed",
                error=sample_result.error,
            )

        plan_path = run_dir / "planning" / "plan_smoke.json"
        _write_json(plan_path, plan_result.to_dict())
        write_noise_stats(run_dir, sample_result)
        manifest.sample_results.append(sample_result.to_dict())
        manifest.planning = {"status_path": str(plan_path), **plan_result.to_dict()}
        manifest.metric_outputs["log"] = str(log_path)
        if sample_result.status != "success":
            manifest.failures.append({"sample": record.to_dict(), "error": sample_result.error})
        manifest_path = write_manifest(manifest)
        print(f"planning smoke written to {plan_path}")
        print(f"manifest: {manifest_path}")
    return 0 if plan_result.status in {"success", "failed"} else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capture-root", default=None, help="override the configured capture root")
    parser.add_argument("--rig-id", default=None, help="filter by camera rig")
    parser.add_argument("--layout", default=None, help="filter by clinical layout")
    parser.add_argument("--sample-indices", default=None, help="comma-separated Unity sample indices to include")
    parser.add_argument("--run-id", default=None, help="explicit output run id")
    parser.add_argument(
        "--noise-profile",
        default=None,
        choices=("none", "robust"),
        help="optional Helios2 Wide noise mode; default config is robust",
    )
    parser.add_argument("--noise-seed", type=int, default=None, help="override deterministic noise seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="modular noise-aware Azurion pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-config", help="print the effective config JSON")
    add_common_args(inspect_parser)
    inspect_parser.set_defaults(func=command_inspect_config)

    sample_parser = subparsers.add_parser("sample", help="export and evaluate one sample")
    add_common_args(sample_parser)
    sample_parser.add_argument("--ordinal", type=int, default=0, help="ordinal within the filtered sample list")
    sample_parser.add_argument("--skip-mujoco-metrics", action="store_true", help=argparse.SUPPRESS)
    sample_parser.add_argument("--skip-camera-metrics", action="store_true")
    sample_parser.add_argument("--skip-planning", action="store_true")
    sample_parser.add_argument("--no-plan-smoke", action="store_true")
    sample_parser.set_defaults(func=command_sample)

    bulk_parser = subparsers.add_parser("bulk", help="run all filtered samples")
    add_common_args(bulk_parser)
    bulk_parser.add_argument("--skip-mujoco-metrics", action="store_true", help=argparse.SUPPRESS)
    bulk_parser.add_argument("--skip-camera-metrics", action="store_true")
    bulk_parser.add_argument("--skip-figures", action="store_true")
    bulk_parser.add_argument("--skip-planning", action="store_true")
    bulk_parser.set_defaults(func=command_bulk)

    plan_parser = subparsers.add_parser("plan-smoke", help="export one sample and run a headless planner attempt")
    add_common_args(plan_parser)
    plan_parser.add_argument("--ordinal", type=int, default=0, help="ordinal within the filtered sample list")
    plan_parser.add_argument("--skip-camera-metrics", action="store_true")
    plan_parser.add_argument("--skip-mujoco-metrics", action="store_true", help=argparse.SUPPRESS)
    plan_parser.set_defaults(func=command_plan_smoke)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
