"""Exercise job sequencing without running GPU training or submitting to Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_restart_uses_new_checkpoint_and_stops_on_failed_smoke(tmp_path, smoke_fails):
    raw = tmp_path / "raw"
    team1 = raw / "team1"
    team1.mkdir(parents=True)
    split = dict(train=["0000000001"], val=["0000000002"], test=[])
    (raw / "data_split.json").write_text(json.dumps(split))
    (team1 / "lesions.csv").write_text("patient,lesion,split\n0000000002,1,val\n")
    (team1 / "centres.csv").write_text("patient,lesion,x_mm,y_mm,z_mm\n0000000002,1,10,20,30\n")
    data, aug = raw / "prepared", raw / "aug"
    keys = dict.fromkeys(["bl", "fu", "bl_mask", "fu_mask", "fu_big", "fu_mask_big", "target_mm", "has_propagated"], 0)
    for root, patients in ((data, ["0000000001", "0000000002"]), (aug, ["0000000001"])):
        (root / "lesions").mkdir(parents=True)
        for pid in patients:
            np.savez(root / "lesions" / f"{pid}_1.npz", **keys)
    (data / "split.json").write_text(json.dumps(split))
    venv, bin_dir = tmp_path / "venv", tmp_path / "bin"
    (venv / "bin").mkdir(parents=True)
    bin_dir.mkdir()
    (venv / "bin" / "activate").write_text("")
    # Preflight runs the real Python code. Only expensive training/inference
    # calls are stubbed, recording exactly which new checkpoint is handed over.
    runner = bin_dir / "python"
    runner.write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys
from pathlib import Path
if sys.argv[1] == '-':
    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
out = Path(args[args.index('--out') + 1])
if args[0] == 'scripts/train.py':
    out.mkdir(parents=True)
    (out / 'best.pt').write_text('new-test-checkpoint')
else:
    ckpt = Path(args[args.index('--ckpt') + 1])
    assert ckpt.read_text() == 'new-test-checkpoint'
    if '--limit' in args and os.environ['SMOKE_FAILS'] == '1':
        raise SystemExit(1)
''')
    runner.chmod(0o755)
    module = bin_dir / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    out, calls = tmp_path / "new-run", tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", RAW=str(raw), DATA=str(data),
               AUG_DATA=str(aug), VENV=str(venv), OUT=str(out), CALLS=str(calls),
               SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="1", SMOKE_FAILS=str(int(smoke_fails)))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "scripts/train_joint.sbatch"], cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == int(smoke_fails), result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == (2 if smoke_fails else 3)
    assert commands[0][0] == "scripts/train.py" and "--ckpt" not in commands[0]
    for command in commands[1:]:
        assert command[command.index("--ckpt") + 1] == str(out / "model" / "best.pt")
    if not smoke_fails:
        assert "--limit" not in commands[-1]
    again = subprocess.run(["bash", "scripts/train_joint.sbatch"], cwd=repo, env=env, capture_output=True, text=True)
    assert again.returncode != 0 and "Output already exists" in again.stderr
