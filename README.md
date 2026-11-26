# Environmental Awareness Pipeline

Python pipeline that turns simulated multi-camera depth captures of an interventional cath lab (Philips Azurion 7 robotic C-arm) into 3D obstacle geometry for collision-aware motion planning. Built for an MSc team internship with Philips (Feb–Jul 2026). Simulation data comes from the companion Unity project, [Environmental_Awareness_Simulation_Environment](https://github.com/andrei-ignat-AI/Environmental_Awareness_Simulation_Environment).

## Pipeline

```text
Unity depth captures (depth + calibration + scene metadata, 4 camera rigs)
  -> dataset I/O and validation
  -> sensor-noise model ("robust" or "none")
  -> robot removal and 3D reconstruction (point clouds, voxels; Open3D)
  -> occlusion analysis and geometry decomposition
  -> MuJoCo collision-geometry export
  -> collision-aware planning check and metrics (camera rig, collision geometry)
  -> run manifest, logs and artifacts under outputs/runs/<run_id>/
```

Everything runs from one CLI (`main.py`). Each run is reproducible from its config and writes a JSON run manifest and a log. Input and output formats are documented in [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md).

## Quick start


1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python main.py run-demo`.

The committed demo dataset is `DepthCaptures_demo/` and contains samples `0003`, `0009`, `0021`, `0027`, and `0036` for all four camera rigs. The full dataset belongs in `DepthCaptures/`.

Common commands:

```bash
python main.py run-demo
python main.py run-full
python main.py view-pointcloud --rig 4CamAsym --sample 0027 --mode demo
python main.py view-voxels --rig 4CamAsym --sample 0027 --mode full
python main.py view-mujoco --rig 4CamAsym --sample 0027 --mode full
python main.py recompute-temp --stage voxels --rig 4CamAsym --sample 0027 --mode demo --view
```

On macOS, `view-mujoco` automatically relaunches through the active virtual environment's `mjpython` viewer launcher when needed. Windows uses the normal `python` command.

Documentation:

- `docs/INSTALLATION.md`
- `docs/USAGE.md`
- `docs/OUTPUT_SCHEMA.md`

Configuration:

- Edit `config.py` for default run options, paths, reconstruction parameters, MuJoCo export settings, metrics, and planning target.
- Default noise mode is `robust`; `none` is also supported.
- Computation is CPU/Open3D/MuJoCo to preserve the report behavior.
