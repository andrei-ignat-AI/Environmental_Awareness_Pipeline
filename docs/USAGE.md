# Usage

Run commands from the repository root.

## Inspect Configuration

```bash
python main.py inspect-config
```

This prints the effective typed config after reading `config.py`.

## Run Demo Dataset

```bash
python main.py run-demo
```

This processes the committed demo samples for all four rigs:

- `0003`
- `0009`
- `0021`
- `0027`
- `0036`

Default output run:

```text
outputs/runs/demo_robust_20/
```

Run without noise:

```bash
python main.py run-demo --noise none
```

Run one demo sample and rig:

```bash
python main.py run-demo --rig 4CamAsym --sample 0027
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

Run without noise:

```bash
python main.py run-full --noise none
```

Run one full sample and rig:

```bash
python main.py run-full --rig 4CamAsym --sample 0027
```

## View Existing Outputs

These commands read an existing run under `outputs/runs/`.

```bash
python main.py view-voxels --rig 4CamAsym --sample 0027 --mode full
python main.py view-mujoco --rig 4CamAsym --sample 0027 --mode full
```

`view-voxels` opens the saved colorful decomposition view: separated components, voxel colors, and wireframe hulls. If an older run is missing this artifact, the command recomputes the decomposition view from the source `DepthCaptures` data and saves it into that run.

Use `--run-id` for a non-default run:

```bash
python main.py view-voxels --rig 4CamAsym --sample 0027 --run-id demo_robust_20 --mode demo
```

## View Reconstructed Point Cloud

Point-cloud viewing reconstructs the selected sample on demand and does not save outputs. The viewer colors points by height on a white background so the room structure is easier to inspect:

```bash
python main.py view-pointcloud --rig 4CamAsym --sample 0027 --mode demo
```

## Temporary Recomputation

Temporary recompute commands write into a system temporary directory and delete it after the command exits.

```bash
python main.py recompute-temp --stage pointcloud --rig 4CamAsym --sample 0027 --mode demo --view
python main.py recompute-temp --stage voxels --rig 4CamAsym --sample 0027 --mode demo --view
python main.py recompute-temp --stage mujoco --rig 4CamAsym --sample 0027 --mode demo --view
```

Without `--view`, the command recomputes and prints stage counts only.

## Configuration

Edit `config.py` for persistent defaults:

- `DATASET_MODE`: `demo` or `full`
- `NOISE_MODE`: `robust` or `none`
- `INCLUDE_ROBOT`: default `False`; robot-inclusive export is for planning-oriented experiments
- `PLANNING_TARGET_M`: default `(3.0, 2.0, 1.0)`
- reconstruction, decomposition, MuJoCo export, metrics, and planning parameters

The default path is CPU/Open3D/MuJoCo. No CUDA or MPS path is used.
