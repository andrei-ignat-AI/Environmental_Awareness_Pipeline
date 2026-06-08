"""Dataset discovery wrappers for the canonical Unity capture export."""

from __future__ import annotations

from pathlib import Path

from . import azurion_dataset

from .config import PipelineConfig
from .types import SampleRecord


def discover_samples(config: PipelineConfig) -> list[SampleRecord]:
    samples = azurion_dataset.discover_samples(
        str(config.dataset.capture_root),
        rig_id=config.dataset.rig_id,
        layout=config.dataset.layout,
    )
    allowed_indices = set(config.dataset.sample_indices) if config.dataset.sample_indices else None
    records: list[SampleRecord] = []
    for index, sample in enumerate(samples):
        if allowed_indices is not None and sample.sample_index not in allowed_indices:
            continue
        records.append(
            SampleRecord(
                index=index,
                path=Path(sample.path),
                relative_path=sample.relative_path,
                sample_name=sample.sample_name,
                rig_id=sample.rig_id,
                layout=sample.layout,
                sample_index=sample.sample_index,
            )
        )
    return records


def make_source_dataset(config: PipelineConfig):
    from . import source_robert2_decomposition as source

    dataset = source.Dataset(str(config.dataset.capture_root))
    filtered_samples = azurion_dataset.discover_samples(
        str(config.dataset.capture_root),
        rig_id=config.dataset.rig_id,
        layout=config.dataset.layout,
    )
    dataset.samples = filtered_samples
    dataset.sample_paths = [sample.path for sample in filtered_samples]
    dataset.List_of_Samples = [sample.relative_path for sample in filtered_samples]
    return dataset
