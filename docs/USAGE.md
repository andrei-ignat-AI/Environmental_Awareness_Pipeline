# Usage

Run every command from the repository root, with the virtual environment activated.

```bash
python main.py --help
```

The public interface is `main.py`. The CLI is meant for running the demo or full pipeline, inspecting the effective configuration, opening saved visualizations, and temporarily recomputing one sample for inspection.

## Command Overview

```bash
python main.py inspect-config
python main.py run-demo
python main.py run-full
python main.py view-pointcloud --rig 4CamAsym --sample 0027 --mode demo
python main.py view-voxels --rig 4CamAsym --sample 0027 --mode demo
python main.py view-mujoco --rig 4CamAsym --sample 0027 --mode demo
python main.py recompute-temp --stage voxels --rig 4CamAsym --sample 0027 --mode demo --view
```

## CLI-Controlled Options

CLI flags change the current command only. They do not edit `config.py`.

Common run flags:

- `--rig`: process or view one camera rig, for example `3Cam` or `4CamAsym`.
- `--sample`: process or view one sample, for example `0027`.
- `--layout`: process one internal dataset layout folder.
- `--noise robust|none`: use robust simulated depth noise, or disable noise.
- `--noise-seed`: override the configured noise seed for the command.
- `--run-id`: choose the run folder name under `outputs/runs/`.
- `--skip-camera-metrics`: skip camera-rig metric generation for run commands.
- `--skip-figures`: skip camera-rig figure generation for run commands.

Viewer and temporary recompute flags:

- `--mode demo|full`: choose the dataset/run family when `--run-id` is not provided.
- `--stage pointcloud|voxels|mujoco`: choose the temporary recompute stage.
- `--view`: open the relevant viewer after temporary recompute.
- `--no-occlusions`: hide occlusion and blind-spot geometry in voxel/decomposition viewers only.

`--no-occlusions` is visual only. It does not change saved outputs, metrics, MuJoCo exports, or the default pipeline behaviour. Occlusions and blind spots remain included unless the corresponding settings are changed in `config.py`.

## Config-Controlled Options

Use `config.py` for persistent defaults and deeper pipeline parameters.

Project paths:

- `DATASET_MODE`
- `CAPTURE_ROOT`
- `OUTPUT_ROOT`
- `ROBOT_HULLS_FOLDER`

Run selection:

- `RUN_ID`
- `RIG_ID`
- `LAYOUT`
- `SAMPLE_INDICES`
- `CONTINUE_ON_SAMPLE_FAILURE`

Noise:

- `NOISE_MODE`
- `NOISE_GLOBAL_SEED`

Reconstruction and decomposition:

- robot removal
- table removal
- voxel size
- object filtering thresholds
- occlusion-zone construction
- blind-spot construction
- convex decomposition settings

MuJoCo export:

- robot inclusion
- occlusion and blind-spot export
- collision geometry style
- hull margins
- XML and voxel artifact export

Metrics:

- camera-rig metrics
- collision-geometry metrics
- summary value and figure outputs

Planning:

- planning target
- planning iteration/time limits
- internal planner settings

The default runtime path is CPU/Open3D/MuJoCo. No CUDA, MPS, or Torch device path is used.

## Inspect Configuration

Use this before running a long job:

```bash
python main.py inspect-config
python main.py inspect-config --mode full
python main.py inspect-config --mode demo --noise none
```

This prints the effective typed configuration after loading `config.py` and applying the command-line overrides.

## Run Demo Dataset

The committed demo dataset contains five scene indices across four camera rigs, so the default demo run processes 20 samples.

```bash
python main.py run-demo
```

Default output run:

```text
outputs/runs/demo_robust_20/
```

Useful variants:

```bash
python main.py run-demo --noise none
python main.py run-demo --rig 4CamAsym --sample 0027
python main.py run-demo --rig 4CamAsym --sample 0027 --run-id my_demo_check
```

## Run Full Dataset

Place the complete Unity export in `DepthCaptures/`, then run:

```bash
python main.py run-full
```

Default output run:

```text
outputs/runs/full_robust_160/
```

Useful variants:

```bash
python main.py run-full --noise none
python main.py run-full --rig 4CamAsym --sample 0027
python main.py run-full --skip-camera-metrics --skip-figures
```

## View Saved Outputs

These commands read existing runs under `outputs/runs/`.

```bash
python main.py view-voxels --rig 4CamAsym --sample 0027 --mode full
python main.py view-mujoco --rig 4CamAsym --sample 0027 --mode full
```

`view-voxels` opens the saved decomposition visualization: colored components, voxel grids, exported hull wireframes, occlusion volumes, and blind spots. Add `--no-occlusions` to hide occlusion and blind-spot geometry in the viewer only:

```bash
python main.py view-voxels --rig 4CamAsym --sample 0027 --mode demo --no-occlusions
```

If an older run is missing the saved decomposition-view artifact, `view-voxels` tries to recreate that artifact from the matching source data.

`view-mujoco` opens the saved MuJoCo XML export. On macOS, the command automatically relaunches through `mjpython` when needed. On Windows, it opens directly from normal `python`.

Use `--run-id` when you want a specific non-default run:

```bash
python main.py view-voxels --rig 4CamAsym --sample 0027 --run-id demo_robust_20 --mode demo
python main.py view-mujoco --rig 4CamAsym --sample 0027 --run-id full_robust_160 --mode full
```

## View Point Clouds

Point-cloud viewing reconstructs the selected sample on demand and does not save outputs. The viewer colors points by height on a white background.

```bash
python main.py view-pointcloud --rig 4CamAsym --sample 0027 --mode demo
python main.py view-pointcloud --rig 4CamAsym --sample 0027 --mode demo --noise none
```

## Temporary Recomputation

Temporary recompute commands write into a system temporary directory and delete it after the command exits.

```bash
python main.py recompute-temp --stage pointcloud --rig 4CamAsym --sample 0027 --mode demo --view
python main.py recompute-temp --stage voxels --rig 4CamAsym --sample 0027 --mode demo --view
python main.py recompute-temp --stage voxels --rig 4CamAsym --sample 0027 --mode demo --view --no-occlusions
python main.py recompute-temp --stage mujoco --rig 4CamAsym --sample 0027 --mode demo --view
```

Without `--view`, the command recomputes and prints stage counts only.

## Output Lookup

Run outputs are stored under:

```text
outputs/runs/<run_id>/
```

Default run IDs are:

```text
demo_robust_20
demo_none_20
full_robust_160
full_none_160
```

Metrics are written separately from run outputs:

```text
metrics/values/
metrics/figures/
```

For the complete folder layout, see `docs/OUTPUT_SCHEMA.md`.
