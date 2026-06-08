# Installation

Use Python 3.11.

Copy only the command lines shown below. Do not paste the Markdown fence markers
such as ````bash`, ````powershell`, or closing ``` lines into the terminal.

## Windows

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py inspect-config
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## macOS

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py inspect-config
```

MuJoCo native viewer windows use the same command on macOS and Windows:

```bash
python main.py view-mujoco --rig 4CamAsym --sample 0027 --mode full
```

On macOS, the CLI automatically relaunches this viewer command through the
`mjpython` executable installed in the active `.venv`. If it reports that
`mjpython` is missing, activate the virtual environment and reinstall
`requirements.txt`. The direct fallback is:

```bash
python .venv/bin/mjpython main.py view-mujoco --rig 4CamAsym --sample 0027 --mode full
```

## Data Folders

`DepthCaptures_demo/` is included for quick validation.

For a full run, place the complete Unity export folder at:

```text
DepthCaptures/
```

The folder should contain rig folders such as:

```text
DepthCaptures/
  3Cam/
  4CamClassic/
  4CamAsym/
  5Cam/
```

Do not change the sample files inside `DepthCaptures/`.
