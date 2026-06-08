"""Small data containers shared by the modular pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SampleRecord:
    index: int
    path: Path
    relative_path: str
    sample_name: str
    rig_id: str | None
    layout: str | None
    sample_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class SampleRunResult:
    sample: SampleRecord
    status: str
    export_dir: Path | None = None
    xml_path: Path | None = None
    json_path: Path | None = None
    voxel_indices_path: Path | None = None
    voxel_ply_path: Path | None = None
    decomposition_view_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    stage_counts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanningRunResult:
    status: str
    sample: SampleRecord | None = None
    goal_poi: tuple[float, float, float] | None = None
    path: list[list[float]] = field(default_factory=list)
    poi_path: list[list[float]] = field(default_factory=list)
    message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class RunManifest:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    source_refs: dict[str, Any]
    sample_results: list[dict[str, Any]] = field(default_factory=list)
    metric_outputs: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
