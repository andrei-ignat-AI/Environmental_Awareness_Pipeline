# Environmental Awareness Pipeline

Clean, transferable Python pipeline for reconstructing Unity depth captures, exporting MuJoCo collision geometry, visualizing intermediate artifacts, and reading the report metrics.

Start here:

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

Documentation:

- `docs/INSTALLATION.md`
- `docs/USAGE.md`
- `docs/OUTPUT_SCHEMA.md`

Configuration:

- Edit `config.py` for default run options, paths, reconstruction parameters, MuJoCo export settings, metrics, and planning target.
- Default noise mode is `robust`; `none` is also supported.
- Computation is CPU/Open3D/MuJoCo to preserve the report behavior.
