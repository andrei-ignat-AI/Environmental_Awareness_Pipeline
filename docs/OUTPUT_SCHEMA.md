# Output Schema

## Input Data

Demo data:

```text
DepthCaptures_demo/<rig>/<layout>/sample_XXXX/
```

Full data:

```text
DepthCaptures/<rig>/<layout>/sample_XXXX/
```

Each sample folder is copied from the Unity export without modification.

## Generated Runs

Run outputs are written to:

```text
outputs/runs/<run_id>/
```

Important files:

```text
outputs/runs/<run_id>/run_manifest.json
outputs/runs/<run_id>/logs/pipeline.log
outputs/runs/<run_id>/noise_stats/<rig>/<layout>/sample_XXXX.json
outputs/runs/<run_id>/mujoco_exports/<rig>/<layout>/sample_XXXX/
outputs/runs/<run_id>/visualizations/decomposition/<rig>/<layout>/sample_XXXX/
outputs/runs/<run_id>/camera_rig_metrics/
```

Per-sample MuJoCo export folder:

```text
sample_XXXX_mujoco.xml
sample_XXXX_boxes.json
sample_XXXX_voxel_indices.npy
sample_XXXX_voxels.ply
```

Per-sample decomposition visualization folder:

```text
sample_XXXX_decomposition_view.npz
```

This artifact stores the colorful Open3D decomposition view used by `view-voxels`: separated component voxels, convex-hull wireframes, blind spots, and other enabled decomposition viewer geometry. It is separate from the uncolored MuJoCo export voxel PLY.

`run_manifest.json` records:

- effective config
- sample list
- status per sample
- exported XML/JSON/voxel paths
- decomposition view path
- stage counts
- noise stats path
- metrics output paths

## Curated Metrics

Tracked summary metrics live outside generated runs:

```text
metrics/values/
metrics/figures/
```

Values:

```text
SUMMARY_camera_rig_visibility.csv
SUMMARY_camera_rig_tradeoff.csv
SUMMARY_free_space_certainty.csv
SUMMARY_collision_geometry.csv
SUMMARY_collision_geometry_compact.csv
SUMMARY_collision_geometry_robert_style.csv
BULK_camera_rig_per_sample.csv
BULK_collision_geometry_per_sample.csv
BULK_collision_geometry_robert_style_per_sample.csv
```

Figures:

```text
camera_rig_floorplan.png/pdf
free_space_certainty_horizontal.png/pdf
layout_robustness_horizontal_heatmap.png/pdf
navigation_score_by_rig.png/pdf
redundant_and_blind_surface_by_rig.png/pdf
robot_relevant_object_visibility.png/pdf
three_to_four_camera_tradeoff.png/pdf
visibility_risk_horizontal.png/pdf
```

`SUMMARY_` files are intended for quick review and report support. `BULK_` files contain per-sample or detailed diagnostic rows.

## Temporary Outputs

`recompute-temp` uses a system temporary directory. Files are deleted automatically when the command finishes.
