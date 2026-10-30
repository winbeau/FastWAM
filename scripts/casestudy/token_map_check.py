"""diff-select self-check C3 (token map): a synthetic marker in a known 32 x 32 cell of one camera's 224 x 224 model view,
through the model's own image composition, VAE and patch grid, must change the token computed for that cell most.

Loads the model through the action-eval adapter with a run's ``resolved.yaml`` options (as latency.py). For each test
cell (camera, h, w): a flat gray observation pair and the same pair with a white square covering the cell's pixels of
the 256 x 256 observation; both through ``compose_model_image``, the VAE (``_encode_input_image_latents_tensor``) and
the (1, 2, 2) patch grid of the latents ([h, w] tokens, w = 0..6 agentview, 7..13 wrist). Prints the token with the
largest latent change against the expected one.

    python scripts/casestudy/token_map_check.py --resolved $W/runs/<FastWAM-Joint run>/resolved.yaml --out check.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CELLS = [("agentview", 0, 0), ("agentview", 3, 5), ("agentview", 6, 2), ("wrist", 1, 1), ("wrist", 5, 6), ("agentview", 2, 6)]
OBS, MODEL, TOKEN = 256, 224, 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pol = yaml.safe_load(Path(args.resolved).read_text())["policy"]
    options = {"model_root": pol["repo_root"], "model_config": pol["model_config"], "checkpoint": pol["checkpoint"],
               **{k: v for k, v in pol["options"].items() if k != "casestudy_file"},
               "casestudy": {"method": "dense", "num_inference_steps": 10}}
    spec = importlib.util.spec_from_file_location("fastwam_action_eval_infer", ROOT / "evaluation/action_eval/infer.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    policy = adapter.create_policy(options, "cuda:0")
    torch = policy._torch
    from einops import rearrange

    def tokens(cams: dict) -> np.ndarray:
        rgb = policy._io.compose_model_image(cams["agentview"], cams["wrist"], processor=policy.processor,
                                             concatenation=policy.concatenation, width=policy.input_w, height=policy.input_h)
        image = policy._io.image_to_model_tensor(rgb, device=policy.device, dtype=policy.model.torch_dtype)
        with torch.no_grad():
            z = policy.model._encode_input_image_latents_tensor(input_image=image, tiled=False)  # [1, C, 1, h, w]
        z = rearrange(z[0].float(), "c (f a) (h b) (w e) -> f h w (c a b e)", a=1, b=2, e=2)[0]
        return z.cpu().numpy()  # [7, 14, D]

    blank = {c: np.full((OBS, OBS, 3), 128, dtype=np.uint8) for c in ("agentview", "wrist")}
    base = tokens(blank)
    rows = []
    for cam, h, w in CELLS:
        cams = {c: v.copy() for c, v in blank.items()}
        lo_y, hi_y = int(np.ceil(h * TOKEN * OBS / MODEL)), int(np.floor((h + 1) * TOKEN * OBS / MODEL))
        lo_x, hi_x = int(np.ceil(w * TOKEN * OBS / MODEL)), int(np.floor((w + 1) * TOKEN * OBS / MODEL))
        cams[cam][lo_y:hi_y, lo_x:hi_x] = 255
        change = np.linalg.norm(tokens(cams) - base, axis=-1)  # [7, 14]
        got = np.unravel_index(int(change.argmax()), change.shape)
        expected = (h, w + (7 if cam == "wrist" else 0))
        rows.append({"camera": cam, "cell": [h, w], "expected_token": list(expected), "max_change_token": [int(got[0]), int(got[1])],
                     "ok": tuple(int(x) for x in got) == expected, "max_change": float(change.max()),
                     "second_change": float(np.sort(change.ravel())[-2])})
        print(json.dumps(rows[-1]), flush=True)
    out = {"cells": rows, "pass": all(r["ok"] for r in rows)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("pass", out["pass"])


if __name__ == "__main__":
    main()
