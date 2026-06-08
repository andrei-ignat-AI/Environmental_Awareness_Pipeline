"""Generate Robert-style collision-geometry metric files for pipeline runs.

This script intentionally recomputes the non-camera-rig MuJoCo metrics from existing
run manifests and exported ``*_voxels.ply`` files. It does not rerun
reconstruction, decomposition, MuJoCo export, camera-rig metrics, or planning.

The metric implementation below copies Robert2's relevant
``evaluate_bulk_mujoco.py``, ``test_2.py``, and ``Evaluation.py`` behavior:

- Unity GT is loaded from ``voxel_props_occupancy.raw``.
- Unity raw occupancy is reshaped as ``(size_y, size_z, size_x)``, transposed
  to ``(x, y, z)``, then flipped on z.
- Unity and MuJoCo predictions are wrapped in Robert's Open3D
  ``RoomVoxelGrid`` before comparison.
- Sparse containment uses Robert's weighted coverage / overfill / loss logic.
  The inner membership checks are vectorized so the handoff can be regenerated
  in minutes instead of spending ages in Python object loops.

Generated files are intended for ``metrics/values`` or another explicit output
folder chosen by the caller.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "outputs" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "metrics" / "values"

DEFAULT_VOXEL_SIZE = 0.05
DEFAULT_ROOM_DIMENSIONS = (9.9, 2.7, 5.9)

RIG_ORDER = ["4CamClassic", "3Cam", "4CamAsym", "5Cam"]
RIG_TO_ROBERT_LABEL = {
    "4CamClassic": "R0_LegacyCorner4",
    "3Cam": "R1_MinimalTriad3",
    "4CamAsym": "R2_HighWallCross4",
    "5Cam": "R3_HybridDense5",
}

ROBERT_METRICS = [
    ("coverage", "cov", "higher"),
    ("overfill_ratio", "over", "lower"),
    ("loss", "loss", "lower"),
]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    profile: str
    label: str
    description: str


DEFAULT_RUNS = [
    RunSpec(
        run_id="full_none_160",
        profile="none",
        label="baseline_no_noise",
        description="Clean modular baseline with no injected depth noise.",
    ),
    RunSpec(
        run_id="full_robust_160",
        profile="robust",
        label="robust_noise",
        description=(
            "Robust profile with axial range perturbation, per-camera bias, "
            "lateral jitter, edge/far dropout, sparse flying pixels, and rare "
            "mostly-deletion outliers."
        ),
    ),
]


class RoomVoxelGrid:
    """Robert2 ``test_2.RoomVoxelGrid`` subset used by bulk evaluation."""

    def __init__(
        self,
        resolution: float = DEFAULT_VOXEL_SIZE,
        width: float = DEFAULT_ROOM_DIMENSIONS[0],
        length: float = DEFAULT_ROOM_DIMENSIONS[1],
        height: float = DEFAULT_ROOM_DIMENSIONS[2],
    ):
        self.resolution = resolution
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.width = width
        self.length = length
        self.height = height

    @property
    def voxel_size(self) -> float:
        return self.resolution

    @property
    def origin(self) -> np.ndarray:
        return np.array(self.voxel_grid.origin)

    @property
    def is_empty(self) -> bool:
        if self.voxel_grid is None:
            return True
        return len(self.voxel_grid.get_voxels()) == 0

    def get_voxel_grid(self):
        return self.voxel_grid.get_voxels()

    def add_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=self.resolution,
        )

    def from_numpy(self, voxel_array: np.ndarray) -> None:
        origin = np.array([-self.width / 2, 0.0, -self.height / 2])
        occupied = np.argwhere(voxel_array)
        centers = occupied * self.resolution + origin + self.resolution * 0.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=self.resolution,
        )


class SparseVoxelEvaluator:
    """Robert2 ``Evaluation.SparseVoxelEvaluator`` containment metric."""

    def __init__(
        self,
        room_center: tuple[float, float, float] = (0, 0, 0),
        sigma_fraction: float = 0.3,
        room_size: tuple[float, float, float] = DEFAULT_ROOM_DIMENSIONS,
    ):
        self.center = np.array(room_center)
        self.sigmas = np.array(room_size) * sigma_fraction

    def _get_weight(self, world_pos: np.ndarray) -> float:
        dist_sq = np.sum(((world_pos - self.center) / self.sigmas) ** 2)
        return float(np.exp(-dist_sq / 2))

    def _get_weights(self, world_pos: np.ndarray) -> np.ndarray:
        if world_pos.size == 0:
            return np.asarray([], dtype=np.float64)
        dist_sq = np.sum(((world_pos - self.center) / self.sigmas) ** 2, axis=1)
        return np.exp(-dist_sq / 2)

    @staticmethod
    def _voxel_indices(grid: RoomVoxelGrid) -> np.ndarray:
        voxels = grid.get_voxel_grid()
        if not voxels:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray([v.grid_index for v in voxels], dtype=np.int64)

    @staticmethod
    def _row_view(values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.int64)
        if values.size == 0:
            return np.asarray([], dtype=np.dtype("V24"))
        return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).ravel()

    @classmethod
    def _isin_rows(cls, values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.asarray([], dtype=bool)
        if len(candidates) == 0:
            return np.zeros(len(values), dtype=bool)
        return np.isin(cls._row_view(values), cls._row_view(candidates))

    @staticmethod
    def _world_positions_and_keys(
        indices: np.ndarray,
        voxel_size: float,
        origin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        world_pos = (indices + 0.5) * voxel_size + origin
        keys = np.floor(world_pos / voxel_size).astype(np.int64)
        return world_pos, keys

    def evaluate_containment(
        self,
        src_grid: RoomVoxelGrid,
        hull_grid: RoomVoxelGrid,
    ) -> dict[str, float]:
        src_is_empty = src_grid is None or src_grid.is_empty
        hull_is_empty = hull_grid is None or hull_grid.is_empty

        if src_is_empty or hull_is_empty:
            return {
                "loss": float("inf"),
                "coverage": 0.0,
                "overfill_ratio": 0.0,
                "weighted_extra": 0.0,
                "weighted_missing": 0.0,
            }

        src_indices = self._voxel_indices(src_grid)
        hull_indices = self._voxel_indices(hull_grid)
        src_world_pos, src_keys = self._world_positions_and_keys(
            src_indices,
            src_grid.voxel_size,
            src_grid.origin,
        )
        hull_world_pos, hull_keys = self._world_positions_and_keys(
            hull_indices,
            hull_grid.voxel_size,
            hull_grid.origin,
        )

        src_weights = self._get_weights(src_world_pos)
        hull_weights = self._get_weights(hull_world_pos)

        weighted_sum_src = float(np.sum(src_weights))

        missing_mask = (
            ~self._isin_rows(src_indices, hull_indices)
            & ~self._isin_rows(src_keys, hull_keys)
        )
        extra_mask = (
            ~self._isin_rows(hull_indices, src_indices)
            & ~self._isin_rows(hull_keys, src_keys)
        )

        # Robert2's source code performs the world-key missing/extra checks twice.
        weighted_sum_missing = float(2.0 * np.sum(src_weights[missing_mask]))
        weighted_sum_extra = float(2.0 * np.sum(hull_weights[extra_mask]))

        coverage = (
            1.0 - (weighted_sum_missing / weighted_sum_src)
            if weighted_sum_src > 0
            else 1.0
        )
        overfill_ratio = (
            weighted_sum_extra / weighted_sum_src if weighted_sum_src > 0 else 0.0
        )
        total_loss = (
            (10.0 * weighted_sum_missing) + weighted_sum_extra
        ) / (weighted_sum_src + 1e-7)

        return {
            "loss": float(total_loss),
            "coverage": float(coverage),
            "overfill_ratio": float(overfill_ratio),
            "weighted_extra": float(weighted_sum_extra),
            "weighted_missing": float(weighted_sum_missing),
        }


def robert_label_for_rig(rig_id: str) -> str:
    return RIG_TO_ROBERT_LABEL.get(rig_id, rig_id)


def robert_sample_path(sample: dict[str, Any]) -> str:
    rig_label = robert_label_for_rig(str(sample.get("rig_id") or "Unknown_Rig"))
    layout = str(sample.get("layout") or "")
    sample_name = str(sample.get("sample_name") or Path(sample["path"]).name)
    return f"{rig_label}/{layout}/{sample_name}" if layout else f"{rig_label}/{sample_name}"


def load_manifest(run: RunSpec) -> dict[str, Any]:
    manifest_path = RUNS_ROOT / run.run_id / "run_manifest.json"
    if not manifest_path.exists():
        print(f"WARNING: Missing run manifest for {run.run_id}: {manifest_path}")
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_unity_voxel_grid_robert(sample_path: Path) -> np.ndarray:
    voxel_grid_path = sample_path / "voxel_props_occupancy.raw"
    voxel_metadata_path = sample_path / "voxel_metadata.json"
    if not voxel_grid_path.exists() or not voxel_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing Unity voxel files: {voxel_grid_path} or {voxel_metadata_path}"
        )

    voxel_grid = np.fromfile(voxel_grid_path, dtype=np.uint8) > 0
    metadata = json.loads(voxel_metadata_path.read_text(encoding="utf-8"))
    size_x = int(metadata["sizeX"])
    size_y = int(metadata["sizeY"])
    size_z = int(metadata["sizeZ"])

    voxel_grid = voxel_grid.reshape((size_y, size_z, size_x))
    voxel_grid_aligned = np.transpose(voxel_grid, (2, 0, 1))
    return np.flip(voxel_grid_aligned, axis=2)


def get_room_dimensions_robert(sample_path: Path) -> tuple[float, float, float]:
    scene_objects_path = sample_path / "scene_objects.json"
    if scene_objects_path.exists():
        metadata = json.loads(scene_objects_path.read_text(encoding="utf-8"))
        for item in metadata.get("objects", []):
            if item.get("objectId") == "room_interior_bounds":
                size = item.get("boundsSize", {})
                return (
                    float(size.get("x", DEFAULT_ROOM_DIMENSIONS[0])),
                    float(size.get("y", DEFAULT_ROOM_DIMENSIONS[1])),
                    float(size.get("z", DEFAULT_ROOM_DIMENSIONS[2])),
                )

    voxel_metadata_path = sample_path / "voxel_metadata.json"
    if voxel_metadata_path.exists():
        metadata = json.loads(voxel_metadata_path.read_text(encoding="utf-8"))
        voxel_size = metadata.get("voxelSize", {})
        return (
            int(metadata.get("sizeX", 0)) * float(voxel_size.get("x", DEFAULT_VOXEL_SIZE)),
            int(metadata.get("sizeY", 0)) * float(voxel_size.get("y", DEFAULT_VOXEL_SIZE)),
            int(metadata.get("sizeZ", 0)) * float(voxel_size.get("z", DEFAULT_VOXEL_SIZE)),
        )

    return DEFAULT_ROOM_DIMENSIONS


def load_mujoco_voxel_grid_robert(
    ply_path: Path,
    resolution: float,
    room_dimensions: tuple[float, float, float],
) -> RoomVoxelGrid | None:
    if not ply_path.exists():
        return None

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return None

    voxel_grid = RoomVoxelGrid(
        resolution=resolution,
        width=room_dimensions[0],
        length=room_dimensions[1],
        height=room_dimensions[2],
    )
    voxel_grid.add_point_cloud(pcd)
    return voxel_grid


def evaluate_sample_robert(result: dict[str, Any]) -> dict[str, Any]:
    sample = result["sample"]
    sample_path = Path(sample["path"])
    ply_path = Path(result["voxel_ply_path"])
    room_dimensions = get_room_dimensions_robert(sample_path)

    unity_voxel_grid = load_unity_voxel_grid_robert(sample_path)
    unity_grid_obj = RoomVoxelGrid(
        resolution=DEFAULT_VOXEL_SIZE,
        width=room_dimensions[0],
        length=room_dimensions[1],
        height=room_dimensions[2],
    )
    unity_grid_obj.from_numpy(unity_voxel_grid)

    mujoco_grid_obj = load_mujoco_voxel_grid_robert(
        ply_path,
        DEFAULT_VOXEL_SIZE,
        room_dimensions,
    )
    if mujoco_grid_obj is None:
        raise FileNotFoundError(f"Missing or empty MuJoCo voxel PLY: {ply_path}")

    evaluator = SparseVoxelEvaluator(room_size=room_dimensions)
    score = evaluator.evaluate_containment(unity_grid_obj, mujoco_grid_obj)
    score["rig_id"] = robert_label_for_rig(str(sample.get("rig_id") or "Unknown_Rig"))
    score["sample_path"] = robert_sample_path(sample)
    score["actual_rig_id"] = sample.get("rig_id")
    score["actual_sample_path"] = sample.get("relative_path")
    score["source_sample_path"] = str(sample_path)
    score["source_ply_path"] = str(ply_path)
    return score


def evaluate_run(run: RunSpec, output_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(run)
    if not manifest:
        return []
    export_include_robot = bool(
        manifest.get("config", {}).get("mujoco_export", {}).get("include_robot", False)
    )
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, result in enumerate(manifest.get("sample_results", []), start=1):
        if result.get("status") != "success":
            failures.append(f"{index}: skipped non-success sample")
            continue
        try:
            score = evaluate_sample_robert(result)
        except Exception as exc:  # noqa: BLE001 - handoff should report all failures.
            sample = result.get("sample", {})
            failures.append(f"{sample.get('relative_path', index)}: {exc}")
            continue

        score["run_id"] = run.run_id
        score["noise_profile"] = run.profile
        score["run_label"] = run.label
        score["export_include_robot"] = export_include_robot
        rows.append(score)

        if index % 40 == 0:
            print(f"{run.run_id}: evaluated {index} manifest entries", flush=True)

    if failures:
        failure_path = output_dir / f"{run.label}_failures.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{run.run_id}: {len(failures)} failures written to {failure_path}", flush=True)
    return rows


def get_stats(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_scores = [
        score
        for score in scores
        if np.isfinite(score["coverage"])
        and np.isfinite(score["overfill_ratio"])
        and np.isfinite(score["loss"])
    ]
    if not valid_scores:
        return None

    coverages = [score["coverage"] for score in valid_scores]
    overfills = [score["overfill_ratio"] for score in valid_scores]
    losses = [score["loss"] for score in valid_scores]

    def get_dist_metrics(values: list[float]) -> dict[str, float]:
        return {
            "avg": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "q25": np.percentile(values, 25),
            "median": np.median(values),
            "q75": np.percentile(values, 75),
            "max": np.max(values),
        }

    cov_stats = get_dist_metrics(coverages)
    over_stats = get_dist_metrics(overfills)
    loss_stats = get_dist_metrics(losses)

    min_loss_sample = min(valid_scores, key=lambda x: x["loss"]).get(
        "sample_path",
        "unknown",
    )
    max_loss_sample = max(valid_scores, key=lambda x: x["loss"]).get(
        "sample_path",
        "unknown",
    )
    min_cov_sample = min(valid_scores, key=lambda x: x["coverage"]).get(
        "sample_path",
        "unknown",
    )
    max_cov_sample = max(valid_scores, key=lambda x: x["coverage"]).get(
        "sample_path",
        "unknown",
    )

    return {
        "count": len(valid_scores),
        "cov_avg": cov_stats["avg"],
        "cov_std": cov_stats["std"],
        "cov_min": cov_stats["min"],
        "cov_q25": cov_stats["q25"],
        "cov_median": cov_stats["median"],
        "cov_q75": cov_stats["q75"],
        "cov_max": cov_stats["max"],
        "cov_min_sample": min_cov_sample,
        "cov_max_sample": max_cov_sample,
        "over_avg": over_stats["avg"],
        "over_std": over_stats["std"],
        "over_min": over_stats["min"],
        "over_q25": over_stats["q25"],
        "over_median": over_stats["median"],
        "over_q75": over_stats["q75"],
        "over_max": over_stats["max"],
        "loss_avg": loss_stats["avg"],
        "loss_std": loss_stats["std"],
        "loss_min": loss_stats["min"],
        "loss_q25": loss_stats["q25"],
        "loss_median": loss_stats["median"],
        "loss_q75": loss_stats["q75"],
        "loss_max": loss_stats["max"],
        "loss_min_sample": min_loss_sample,
        "loss_max_sample": max_loss_sample,
    }


def robert_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results_per_rig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        results_per_rig[row["rig_id"]].append(row)

    output: list[dict[str, Any]] = []
    for rig in sorted(results_per_rig.keys()):
        stats = get_stats(results_per_rig[rig])
        if stats:
            output.append({"Rig ID": rig, **stats})

    global_stats = get_stats(rows)
    if global_stats:
        output.append({"Rig ID": "Global Average", **global_stats})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_robert_exact_files(run: RunSpec, rows: list[dict[str, Any]], output_dir: Path) -> None:
    run_dir = output_dir / run.label
    detailed_fields = [
        "rig_id",
        "sample_path",
        "coverage",
        "overfill_ratio",
        "loss",
        "weighted_extra",
        "weighted_missing",
    ]
    summary_fields = [
        "Rig ID",
        "count",
        "cov_avg",
        "cov_std",
        "cov_min",
        "cov_q25",
        "cov_median",
        "cov_q75",
        "cov_max",
        "cov_min_sample",
        "cov_max_sample",
        "over_avg",
        "over_std",
        "over_min",
        "over_q25",
        "over_median",
        "over_q75",
        "over_max",
        "loss_avg",
        "loss_std",
        "loss_min",
        "loss_q25",
        "loss_median",
        "loss_q75",
        "loss_max",
        "loss_min_sample",
        "loss_max_sample",
    ]

    write_csv(
        run_dir / "bulk_mujoco_metrics_detailed.csv",
        [{field: row.get(field, "") for field in detailed_fields} for row in rows],
        detailed_fields,
    )
    summary_rows = robert_summary_rows(rows)
    write_csv(run_dir / "bulk_mujoco_metrics_summary.csv", summary_rows, summary_fields)
    (run_dir / "bulk_mujoco_metrics.tex").write_text(
        build_robert_latex_table(summary_rows),
        encoding="utf-8",
    )


def build_robert_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[hbt!]",
        r"\centering",
        r"\caption{Bulk Evaluation Metrics by Camera Rig}",
        r"\begin{tabular}{|l|cccc|cccc|cccc|}",
        r"\hline",
        r"Rig ID & \multicolumn{4}{c|}{Coverage} & \multicolumn{4}{c|}{Overfill Ratio} & \multicolumn{4}{c|}{Total Loss} \\",
        r"\cline{2-13}",
        r" & Avg & Std & Min & Max & Avg & Std & Min & Max & Avg & Std & Min & Max \\",
        r"\hline",
    ]
    for row in rows:
        name = str(row["Rig ID"]).replace("_", r"\_")
        if row["Rig ID"] == "Global Average":
            lines.append(r"\hline")
        lines.append(
            f"{name} & {row['cov_avg']:.4f} & {row['cov_std']:.4f} & "
            f"{row['cov_min']:.4f} & {row['cov_max']:.4f} & "
            f"{row['over_avg']:.4f} & {row['over_std']:.4f} & "
            f"{row['over_min']:.4f} & {row['over_max']:.4f} & "
            f"{row['loss_avg']:.4f} & {row['loss_std']:.4f} & "
            f"{row['loss_min']:.4f} & {row['loss_max']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_combined_summary(
    runs: list[RunSpec],
    all_rows_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        for row in robert_summary_rows(all_rows_by_run[run.run_id]):
            output.append(
                {
                    "run_id": run.run_id,
                    "noise_profile": run.profile,
                    "run_label": run.label,
                    **row,
                }
            )
    return output


def quality_hint(metric: str, delta: float) -> str:
    if abs(delta) < 1e-12:
        return "unchanged"
    direction = next(direction for name, _, direction in ROBERT_METRICS if name == metric)
    if direction == "higher":
        return "better" if delta > 0 else "worse"
    return "worse" if delta > 0 else "better"


def pct_delta(new_value: float, base_value: float) -> float | str:
    if base_value == 0:
        return ""
    return 100.0 * (new_value - base_value) / base_value


def summary_by_rig(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["Rig ID"]: row for row in robert_summary_rows(rows)}


def build_delta_summary(
    runs: list[RunSpec],
    all_rows_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not all_rows_by_run.get(runs[0].run_id):
        return []
    baseline = summary_by_rig(all_rows_by_run[runs[0].run_id])
    output: list[dict[str, Any]] = []

    for run in runs[1:]:
        current = summary_by_rig(all_rows_by_run[run.run_id])
        for rig_id, current_row in current.items():
            baseline_row = baseline[rig_id]
            out = {
                "noise_run_id": run.run_id,
                "noise_profile": run.profile,
                "run_label": run.label,
                "Rig ID": rig_id,
                "count": current_row["count"],
            }
            worse_signals = []
            for metric, prefix, _ in ROBERT_METRICS:
                baseline_key = f"{prefix}_avg"
                base_value = float(baseline_row[baseline_key])
                noise_value = float(current_row[baseline_key])
                delta = noise_value - base_value
                hint = quality_hint(metric, delta)
                out[f"baseline_{prefix}_avg"] = base_value
                out[f"noise_{prefix}_avg"] = noise_value
                out[f"{prefix}_delta"] = delta
                out[f"{prefix}_delta_pct"] = pct_delta(noise_value, base_value)
                out[f"{prefix}_quality_hint"] = hint
                if hint == "worse":
                    worse_signals.append(prefix)
            out["overall_quality_hint"] = (
                "worse" if {"over", "loss"} & set(worse_signals) else "mixed"
            )
            out["interpretation"] = (
                "Robert2-exact metric recomputation from existing exported PLYs; "
                "lower overfill and loss are better."
            )
            output.append(out)
    return output


def build_paired_sample_deltas(
    runs: list[RunSpec],
    all_rows_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not all_rows_by_run.get(runs[0].run_id):
        return []
    baseline_by_sample = {
        row["actual_sample_path"]: row
        for row in all_rows_by_run[runs[0].run_id]
    }
    output: list[dict[str, Any]] = []

    for run in runs[1:]:
        for row in all_rows_by_run[run.run_id]:
            base = baseline_by_sample[row["actual_sample_path"]]
            out = {
                "noise_run_id": run.run_id,
                "noise_profile": run.profile,
                "run_label": run.label,
                "Rig ID": row["rig_id"],
                "sample_path": row["sample_path"],
                "actual_rig_id": row["actual_rig_id"],
                "actual_sample_path": row["actual_sample_path"],
            }
            for metric, prefix, _ in ROBERT_METRICS:
                base_value = float(base[metric])
                noise_value = float(row[metric])
                delta = noise_value - base_value
                out[f"baseline_{prefix}"] = base_value
                out[f"noise_{prefix}"] = noise_value
                out[f"{prefix}_delta"] = delta
                out[f"{prefix}_delta_pct"] = pct_delta(noise_value, base_value)
                out[f"{prefix}_quality_hint"] = quality_hint(metric, delta)
            output.append(out)
    return output


def format_float(value: object, digits: int = 4) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def build_noise_latex_table(delta_rows: list[dict[str, Any]]) -> str:
    global_rows = [row for row in delta_rows if row["Rig ID"] == "Global Average"]
    lines = [
        r"\begin{table}[hbt!]",
        r"\centering",
        r"\caption{Effect of robust depth noise on Robert2 MuJoCo metrics. Lower overfill and loss are better.}",
        r"\begin{tabular}{|l|ccc|ccc|ccc|}",
        r"\hline",
        r"Profile & \multicolumn{3}{c|}{Coverage} & \multicolumn{3}{c|}{Overfill Ratio} & \multicolumn{3}{c|}{Total Loss} \\",
        r"\cline{2-10}",
        r" & Baseline & Noise & Delta & Baseline & Noise & Delta & Baseline & Noise & Delta \\",
        r"\hline",
    ]
    profile_label = {
        "robust": "Robust",
    }
    for row in global_rows:
        lines.append(
            " & ".join(
                [
                    profile_label.get(str(row["noise_profile"]), str(row["noise_profile"])),
                    format_float(row["baseline_cov_avg"]),
                    format_float(row["noise_cov_avg"]),
                    format_float(row["cov_delta"]),
                    format_float(row["baseline_over_avg"]),
                    format_float(row["noise_over_avg"]),
                    format_float(row["over_delta"]),
                    format_float(row["baseline_loss_avg"]),
                    format_float(row["noise_loss_avg"]),
                    format_float(row["loss_delta"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_markdown_summary(
    runs: list[RunSpec],
    delta_rows: list[dict[str, Any]],
    combined_summary: list[dict[str, Any]],
) -> str:
    global_rows = [row for row in delta_rows if row["Rig ID"] == "Global Average"]
    baseline_global = next(
        (row for row in combined_summary if row["run_id"] == runs[0].run_id and row["Rig ID"] == "Global Average"),
        {"cov_avg": 0.0, "over_avg": 0.0, "loss_avg": 0.0}
    )
    lines = [
        "# Collision Geometry Metrics",
        "",
        "This handoff excludes camera-rig metrics. It recomputes MuJoCo metrics from existing run manifests and exported PLYs using the Robert2 metric path.",
        "",
        "Robert2 behavior copied here:",
        "",
        "- Unity GT: `voxel_props_occupancy.raw` plus `voxel_metadata.json`.",
        "- Unity reshape: `(size_y, size_z, size_x)`, transpose to `(x, y, z)`, flip z.",
        "- Grid wrapper: Robert's Open3D `RoomVoxelGrid` with room dimensions from `scene_objects.json`.",
        "- Evaluator: Robert's weighted sparse containment metric from `Evaluation.SparseVoxelEvaluator`.",
        "",
        "Robot scope: these runs use `include_robot=False` in the MuJoCo export. That matches Robert-facing evaluation because the Unity ground truth is `voxel_props_occupancy.raw`, Philips knows the robot state, and robot detection is not in scope for these metrics.",
        "",
        "## Profiles",
        "",
        "- `none`: no injected depth noise.",
        "- `robust`: active range `0.30-8.30 m`, axial range perturbation from an 8300 mm precision curve, per-camera axial bias, `0.03 px` lateral jitter, edge dropout, far-range dropout, sparse flying pixels, and rare mostly-deletion outliers.",
        "",
        "## Main Result",
        "",
        "| Profile | Coverage Mean | Coverage Delta | Overfill Mean | Overfill Delta | Loss Mean | Loss Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in global_rows:
        lines.append(
            "| {profile} | {cov:.4f} | {cov_delta:+.4f} | {over:.4f} | {over_delta:+.4f} | {loss:.4f} | {loss_delta:+.4f} |".format(
                profile=row["noise_profile"],
                cov=float(row["noise_cov_avg"]),
                cov_delta=float(row["cov_delta"]),
                over=float(row["noise_over_avg"]),
                over_delta=float(row["over_delta"]),
                loss=float(row["noise_loss_avg"]),
                loss_delta=float(row["loss_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "Baseline global means for reference:",
            "",
            "- coverage `{}`, overfill `{}`, loss `{}`".format(
                format_float(baseline_global["cov_avg"]),
                format_float(baseline_global["over_avg"]),
                format_float(baseline_global["loss_avg"]),
            ),
            "",
            "## Files",
            "",
            "- `baseline_no_noise/`, `robust_noise/`: Robert-shaped `bulk_mujoco_metrics_detailed.csv`, `bulk_mujoco_metrics_summary.csv`, and LaTeX table for each run.",
            "- `bulk_mujoco_noise_metrics_detailed.csv`: combined per-sample metric rows for baseline and robust runs.",
            "- `bulk_mujoco_noise_metrics_summary.csv`: combined Robert summary rows with run/profile columns.",
            "- `bulk_mujoco_noise_delta_vs_baseline.csv`: rig/global deltas against no-noise baseline.",
            "- `bulk_mujoco_noise_paired_sample_deltas.csv`: per-sample deltas against no-noise baseline.",
            "- `bulk_mujoco_noise_metrics.tex`: compact global noise-effect LaTeX table.",
            "- `metadata.json`: source refs, profile descriptions, method notes, and robot-scope statement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robert2-style metric handoff generator")
    parser.add_argument("--baseline-run-id", default=DEFAULT_RUNS[0].run_id)
    parser.add_argument("--robust-run-id", default=DEFAULT_RUNS[1].run_id)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def run_specs_from_args(args: argparse.Namespace) -> list[RunSpec]:
    return [
        RunSpec(
            run_id=args.baseline_run_id,
            profile="none",
            label="baseline_no_noise",
            description="Clean modular baseline with no injected depth noise.",
        ),
        RunSpec(
            run_id=args.robust_run_id,
            profile="robust",
            label="robust_noise",
            description=(
                "Robust profile with axial range perturbation, per-camera bias, "
                "lateral jitter, edge/far dropout, sparse flying pixels, and rare "
                "mostly-deletion outliers."
            ),
        ),
    ]


def output_dir_from_arg(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    runs = run_specs_from_args(args)
    output_dir = output_dir_from_arg(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows_by_run = {run.run_id: evaluate_run(run, output_dir) for run in runs}
    for run in runs:
        if all_rows_by_run.get(run.run_id):
            write_robert_exact_files(run, all_rows_by_run[run.run_id], output_dir)

    detailed_rows = [row for run in runs for row in all_rows_by_run[run.run_id]]
    combined_summary = build_combined_summary(runs, all_rows_by_run)
    delta_rows = build_delta_summary(runs, all_rows_by_run)
    paired_delta_rows = build_paired_sample_deltas(runs, all_rows_by_run)

    write_csv(
        output_dir / "bulk_mujoco_noise_metrics_detailed.csv",
        detailed_rows,
        [
            "run_id",
            "noise_profile",
            "run_label",
            "rig_id",
            "sample_path",
            "actual_rig_id",
            "actual_sample_path",
            "coverage",
            "overfill_ratio",
            "loss",
            "weighted_extra",
            "weighted_missing",
            "source_sample_path",
            "source_ply_path",
            "export_include_robot",
        ],
    )
    write_csv(
        output_dir / "bulk_mujoco_noise_metrics_summary.csv",
        combined_summary,
        [
            "run_id",
            "noise_profile",
            "run_label",
            "Rig ID",
            "count",
            "cov_avg",
            "cov_std",
            "cov_min",
            "cov_q25",
            "cov_median",
            "cov_q75",
            "cov_max",
            "cov_min_sample",
            "cov_max_sample",
            "over_avg",
            "over_std",
            "over_min",
            "over_q25",
            "over_median",
            "over_q75",
            "over_max",
            "loss_avg",
            "loss_std",
            "loss_min",
            "loss_q25",
            "loss_median",
            "loss_q75",
            "loss_max",
            "loss_min_sample",
            "loss_max_sample",
        ],
    )
    write_csv(
        output_dir / "bulk_mujoco_noise_metrics.csv",
        [
            {
                "run_id": row["run_id"],
                "noise_profile": row["noise_profile"],
                "run_label": row["run_label"],
                "Rig ID": row["Rig ID"],
                "count": row["count"],
                "cov_avg": row["cov_avg"],
                "cov_min": row["cov_min"],
                "cov_max": row["cov_max"],
                "over_avg": row["over_avg"],
                "over_min": row["over_min"],
                "over_max": row["over_max"],
                "loss_avg": row["loss_avg"],
                "loss_min": row["loss_min"],
                "loss_max": row["loss_max"],
            }
            for row in combined_summary
        ],
        [
            "run_id",
            "noise_profile",
            "run_label",
            "Rig ID",
            "count",
            "cov_avg",
            "cov_min",
            "cov_max",
            "over_avg",
            "over_min",
            "over_max",
            "loss_avg",
            "loss_min",
            "loss_max",
        ],
    )
    write_csv(
        output_dir / "bulk_mujoco_noise_delta_vs_baseline.csv",
        delta_rows,
        [
            "noise_run_id",
            "noise_profile",
            "run_label",
            "Rig ID",
            "count",
            "baseline_cov_avg",
            "noise_cov_avg",
            "cov_delta",
            "cov_delta_pct",
            "cov_quality_hint",
            "baseline_over_avg",
            "noise_over_avg",
            "over_delta",
            "over_delta_pct",
            "over_quality_hint",
            "baseline_loss_avg",
            "noise_loss_avg",
            "loss_delta",
            "loss_delta_pct",
            "loss_quality_hint",
            "overall_quality_hint",
            "interpretation",
        ],
    )
    write_csv(
        output_dir / "bulk_mujoco_noise_paired_sample_deltas.csv",
        paired_delta_rows,
        [
            "noise_run_id",
            "noise_profile",
            "run_label",
            "Rig ID",
            "sample_path",
            "actual_rig_id",
            "actual_sample_path",
            "baseline_cov",
            "noise_cov",
            "cov_delta",
            "cov_delta_pct",
            "cov_quality_hint",
            "baseline_over",
            "noise_over",
            "over_delta",
            "over_delta_pct",
            "over_quality_hint",
            "baseline_loss",
            "noise_loss",
            "loss_delta",
            "loss_delta_pct",
            "loss_quality_hint",
        ],
    )

    (output_dir / "bulk_mujoco_noise_metrics.tex").write_text(
        build_noise_latex_table(delta_rows),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        build_markdown_summary(runs, delta_rows, combined_summary),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "purpose": "Robert-style collision-geometry metrics for pipeline exports.",
                "generated_from_runs": [run.__dict__ for run in runs],
                "rig_label_map": RIG_TO_ROBERT_LABEL,
                "metric_method": (
                    "Copied Robert2 evaluate_bulk_mujoco.py/test_2.py/"
                    "Evaluation.py behavior into this generator."
                ),
                "source_inputs": [
                    "outputs/runs/<run_id>/run_manifest.json",
                    "Unity sample voxel_props_occupancy.raw",
                    "Unity sample voxel_metadata.json",
                    "Unity sample scene_objects.json",
                    "Existing exported sample *_voxels.ply",
                ],
                "excluded": ["camera-rig metrics", "pipeline regeneration"],
                "robot_scope": (
                    "These Robert-facing runs are expected to use include_robot=False. "
                    "Robot geometry is kept only for separate planning-oriented exports."
                ),
                "output_dir": str(output_dir.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Robert noise handoff written to: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
