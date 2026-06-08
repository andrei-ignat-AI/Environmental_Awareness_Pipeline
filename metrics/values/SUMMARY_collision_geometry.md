# Collision Geometry Metrics

These files report Robert-style MuJoCo collision-geometry metrics for the full robust run `full_robust_160`.

The implementation compares Unity ground-truth prop occupancy against the exported MuJoCo voxel geometry using the copied Robert2 weighted containment method:

- coverage: higher is better
- overfill ratio: lower is better
- loss: lower is better

The default reported scope is robot-off (`INCLUDE_ROBOT = False`) because the collision-geometry validation evaluates reconstruction/export of scene obstacles, not robot-state estimation.

Tracked files:

- `SUMMARY_collision_geometry.csv`: rig-level and global summary.
- `SUMMARY_collision_geometry_compact.csv`: compact summary table from the handoff bundle.
- `BULK_collision_geometry_per_sample.csv`: per-sample rows for all 160 full-run exports.
- `SUMMARY_collision_geometry_robert_style.csv`: Robert-shaped summary table.
- `BULK_collision_geometry_robert_style_per_sample.csv`: Robert-shaped per-sample table.
