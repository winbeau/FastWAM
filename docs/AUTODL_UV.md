# AutoDL uv reproduction

This path reproduces the frozen FastWAM environment on the measured AutoDL RTX 4090 host without Git push/pull.

## Run

Transfer the working tree without `.git`, `.venv`, `runs`, or `outputs`, then run:

```bash
cd /root/autodl-tmp/fastwam-p1
bash scripts/autodl_bootstrap.sh
```

The bootstrap command:

1. moves uv, Hugging Face, Torch, and XDG caches to `/root/autodl-tmp`;
2. runs `uv sync --frozen` from the committed `uv.lock`;
3. writes a machine/storage/CUDA preflight report;
4. runs import, Panthera LeRobot-v3 data, and tiny structural model/data smoke checks.

No bare `pip` or conda command is used.

## Pull one Pi 5 episode from Hugging Face

Pi 5 publishes complete collectord episodes to the Hugging Face dataset repo
`winbeau/fastwam-lerobot`. Always pin the 40-character Hub revision printed by the Pi upload;
do not ingest mutable `main` implicitly. Restore the matching Panthera-WAM source archive,
then run:

```bash
export PANTHERA_WAM_ROOT=/root/autodl-tmp/fastwam-work/Panthera-WAM
bash scripts/autodl_ingest_panthera_episode.sh \
  color-block-000001 \
  <40-character-hf-revision> \
  episodes
```

For a non-training diagnostic use `smoke` as the third argument. The script downloads exactly
three files for that episode, verifies the portable SHA-256 and manifest, rejects unsafe tar
members, extracts atomically, runs the pinned LeRobot 0.4.4/v3 producer from
`Panthera-WAM/tools/lerobot-v3`, then executes `scripts/validate_panthera_dataset.py` including
video decode probes. AutoDL may load `~/.bashrc` before this command when its configured
Hugging Face mirror is required; credentials and proxy values must not be logged.

## Expected artifacts

The command writes:

- `artifacts/autodl/preflight.json`
- `artifacts/autodl/import.json`
- `artifacts/autodl/data.json`
- `artifacts/autodl/structural.json`

The structural stage uses the real `WanVideoDiT`, `ActionDiT`, and `MoT` implementations with tiny random weights and Panthera fixture actions. It performs one CUDA forward/backward pass and records wall time and peak allocated VRAM. It does **not** download Wan weights and is not a full pretrained FastWAM training result.

## Measured P1 host

The recorded host had one RTX 4090 (49,140 MiB as reported by `nvidia-smi`), driver 580.105.08, PyTorch 2.7.1+cu128, and a CUDA 11.8 system toolkit. PyTorch CUDA 12.8 wheels ran successfully through the installed driver. The frozen environment and caches used about 14 GiB of the 50 GiB `/root/autodl-tmp` volume.

The host had no Tailscale binary or `/dev/net/tun`, so it is not eligible for the Pi closed loop. P1 evidence is offline environment/data/structural smoke only.
