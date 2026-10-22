"""Precompute umt5 prompt encodings for a fixed instruction set.

The case-study runner reuses prompt encodings anyway (exact: ``encode_prompt`` is
deterministic), so storing them once lets the eval workers skip loading the 11 GB
text encoder. Encodings are produced by FastWAM's own ``encode_prompt`` with the same
tokenizer, max length and text encoder the model would load
(``tokenizer_max_len`` from configs/model/fastwam_joint.yaml), on the GPU given.

    python scripts/casestudy/precompute_prompts.py --instructions libero_instructions.json \
        --diffsynth-base <ckpt>/diffsynth --out prompts.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", required=True, help="JSON: {suite: [instruction, ...]} or a list")
    ap.add_argument("--diffsynth-base", required=True)
    ap.add_argument("--tokenizer-max-len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.diffsynth_base
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    import torch

    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    raw = json.loads(Path(args.instructions).read_text())
    instructions = sorted({x for v in raw.values() for x in v} if isinstance(raw, dict) else set(raw))

    _dit, text_config, _vae, tokenizer_config = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B")
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    encoder = _load_registered_model(text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=args.device)
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=args.tokenizer_max_len, clean="whitespace")
    owner = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer, device=torch.device(args.device))

    store = {}
    for instruction in instructions:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, mask = FastWAM.encode_prompt(owner, prompt)
        _ids, raw_mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        store[prompt] = {
            "context": context.detach().to("cpu"),
            "mask": mask.detach().to("cpu"),
            "num_text": int(raw_mask.gt(0).sum().item()),
        }
    torch.save(
        {
            "prompts": store,
            "text_encoder": str(text_config.path),
            "tokenizer": str(tokenizer_config.path),
            "tokenizer_max_len": args.tokenizer_max_len,
            "dtype": "bfloat16",
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        },
        args.out,
    )
    shapes = {tuple(v["context"].shape) for v in store.values()}
    print(f"wrote {len(store)} prompts to {args.out}; context shapes {shapes}")


if __name__ == "__main__":
    main()
