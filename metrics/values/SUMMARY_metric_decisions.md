# Results Decisions

This file documents the stakeholder-grade metric choices implemented by `source_camera_rig_metrics.py`.

## Scope

- writes to `camera_rig_metrics/` inside each run directory.
- The headline story is camera-rig support for autonomous robot navigation through observability and redundancy, not reconstruction overlap.
- MuJoCo remains a pipeline handoff and demonstration target until a controlled planning benchmark is defined.

## Dataset Design

- The paired camera-rig design is valid because the audit found `40` matched scene indices and `160` sample folders.
- Scene, robot, props, and robot voxel files are required to hash-match across rigs for each scene index.
- The paired design isolates the effect of camera placement because every rig sees the same accepted scene.

## Headline Metrics

- Active metric scope for this run: `roi`.
- ROI bounds are `X [-4.0, 4.0]`, `Y [0.0, 2.35]`, and `Z [-2.0, 2.0]` meters.
- Observable surface is the fraction of ROI GT surface seen by at least one camera.
- Redundant surface is the fraction seen by at least two cameras.
- Strong redundancy is the fraction seen by at least three cameras.
- Single-view surface is visible but fragile because it depends on one camera.
- Blind surface is not observable by the rig.
- Navigation Visibility Score is `100 * (0.35 * redundant surface + 0.20 * strong redundancy + 0.10 * observable surface + 0.10 * non-blind surface + 0.10 * certified free volume + 0.15 * redundantly certified free volume)`.
- These weights reflect a clinical/corporate safety preference: robust multi-view obstacle evidence first, certified motion-space evidence second, and simple one-camera coverage last.

## Semantic Metrics

- Stakeholder-facing semantic outputs exclude patient and ceiling lamp.
- Patient visibility is available in the full semantic diagnostic CSV, but it is not a headline metric because the autonomous robot should move around the room rather than over the patient.
- Object-region visibility uses exported AABB regions and aggregate prop voxels; it is not exact mesh segmentation.

## Free-Space Certainty And Occlusion Artifacts

- recomputes free-space certainty from the current depth captures on every non-audit run.
- Stored artifacts live under `camera_rig_metrics/occlusion_zones/<Rig>/<Layout>/sample_XXXX.npz` with companion JSON metadata.
- Certified free volume means GT-empty voxels that are in front of the measured depth surface for at least one camera.
- Unknown free volume means GT-empty voxels that no camera could certify as free; this is conservative unknown space, not occupied obstacle geometry.
- Far/no-hit depth pixels can certify free space up to far clip, but they are never treated as occupied surfaces.

## Reconstruction Diagnostics

- Distance-tolerant and exact voxel-overlap reconstruction metrics are written to `SUMMARY_diagnostic_reconstruction.*` only.
- They are representation-alignment diagnostics because the reconstruction is a depth-derived first-hit surface while Unity GT is an occupancy-derived surface band.

## Figure style

- Figures are written as both `.pdf` and `.png` under `camera_rig_metrics/figures`.
- Plot style file: `not bundled`.
- Figures use IEEE single-column or double-column canvases, no plot titles, grid where useful, and horizontal bar-style layouts where possible.
- PNG figures are exported at 600 dpi, while PDF figures keep embedded TrueType-compatible fonts for report/presentation use.

## Runtime settings

- capture root: `DepthCaptures`
- results dir: `outputs/runs/full_robust_160/camera_rig_metrics`
- depth stride: `1`
- metric scope: `roi`
- metrics rows: `160`
- semantic rows: `1760`
