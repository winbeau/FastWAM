#!/usr/bin/env python3
"""Bounded AutoDL smoke checks for the frozen FastWAM environment.

These checks intentionally do not download Wan weights.  The structural stage
runs the real FastWAM video/action expert and MoT implementations at tiny
shapes, using actions read through the Panthera LeRobot-v3 adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/panthera_lerobot_v3_minimal"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def base_report(stage: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in (
                "fastwam",
                "torch",
                "torchvision",
                "accelerate",
                "datasets",
                "transformers",
                "torchcodec",
            )
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    if cuda_available:
        report["gpu"] = {
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    return report


def run_import() -> dict[str, Any]:
    started = time.perf_counter()
    from fastwam.datasets.lerobot.lerobot_v3_adapter import PantheraLeRobotV3Dataset
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    assert PantheraLeRobotV3Dataset and ActionDiT and MoT and WanVideoDiT
    report = base_report("import")
    report["wall_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    return report


def shape_meta() -> dict[str, Any]:
    return {
        "images": [
            {"key": "overhead_rgb", "raw_shape": [3, 64, 64], "shape": [3, 128, 128]},
            {"key": "wrist_rgb", "raw_shape": [3, 64, 64], "shape": [3, 128, 128]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    }


def build_dataset(work_dir: Path):
    import hashlib

    from torchvision.transforms import Resize

    from fastwam.datasets.lerobot.lerobot_v3_adapter import PantheraLeRobotV3Dataset
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from fastwam.datasets.lerobot.transforms.image import ToTensor
    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
    from fastwam.utils import misc

    misc.register_work_dir(work_dir)
    processor = FastWAMProcessor(
        shape_meta=shape_meta(),
        num_obs_steps=5,
        num_output_cameras=2,
        action_output_dim=7,
        proprio_output_dim=7,
        delta_action_dim_mask={"default": [False] * 7},
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor(), Resize(size=[128, 128])],
        val_transforms=[ToTensor(), Resize(size=[128, 128])],
    )
    raw = PantheraLeRobotV3Dataset(
        [str(FIXTURE)],
        shape_meta(),
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    stats_path = work_dir / "dataset_stats.json"
    save_dataset_stats_to_json(raw.get_dataset_stats(processor), stats_path)

    prompt = DEFAULT_PROMPT.format(task="Move the red block from the start area to the target area.")
    cache_dir = work_dir / "text-cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len128.wan22ti2v5b.pt"
    torch.save(
        {"context": torch.zeros(128, 4096), "mask": torch.ones(128, dtype=torch.bool)},
        cache_path,
    )
    return RobotVideoDataset(
        dataset_dirs=[str(FIXTURE)],
        dataset_format="panthera_v3",
        video_backend="pyav",
        shape_meta=shape_meta(),
        num_frames=5,
        video_size=[128, 256],
        processor=processor,
        text_embedding_cache_dir=str(cache_dir),
        context_len=128,
        pretrained_norm_stats=str(stats_path),
        val_set_proportion=0.0,
        is_training_set=True,
        action_video_freq_ratio=1,
        concat_multi_camera="horizontal",
        random_fallback_on_error=False,
    )


def run_data() -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fastwam-autodl-data-") as tmp:
        dataset = build_dataset(Path(tmp))
        sample = dataset[0]
    expected = {
        "video": (3, 5, 128, 256),
        "action": (4, 7),
        "proprio": (4, 7),
        "action_is_pad": (4,),
        "image_is_pad": (5,),
    }
    shapes = {key: tuple(sample[key].shape) for key in expected}
    if shapes != expected:
        raise AssertionError(f"dataset shape mismatch: expected={expected}, got={shapes}")
    report = base_report("data")
    report.update(
        {
            "status": "passed",
            "fixture": str(FIXTURE.relative_to(ROOT)),
            "dataset_length": len(dataset),
            "sample_shapes": {key: list(value) for key, value in shapes.items()},
            "video_value_range": [float(sample["video"].min()), float(sample["video"].max())],
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return report


def run_structural() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("structural AutoDL smoke requires CUDA")

    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    started = time.perf_counter()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with tempfile.TemporaryDirectory(prefix="fastwam-autodl-structural-") as tmp:
        sample = build_dataset(Path(tmp))[0]

    video_expert = WanVideoDiT(
        hidden_dim=32,
        in_dim=4,
        ffn_dim=64,
        out_dim=4,
        text_dim=16,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=24,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_clip_embedding=False,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    ).to(device=device, dtype=dtype)
    action_expert = ActionDiT(
        hidden_dim=32,
        action_dim=7,
        ffn_dim=64,
        text_dim=16,
        freq_dim=32,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=24,
        num_layers=1,
    ).to(device=device, dtype=dtype)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    ).to(device=device, dtype=dtype)
    mot.train()
    optimizer = torch.optim.AdamW(mot.parameters(), lr=1.0e-3)

    context = torch.randn(1, 4, 16, device=device, dtype=dtype)
    context_mask = torch.ones(1, 4, device=device, dtype=torch.bool)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    latents = torch.randn(1, 4, 3, 4, 4, device=device, dtype=dtype)
    action = sample["action"][:8].unsqueeze(0).to(device=device, dtype=dtype)

    def forward_loss():
        video_pre = video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        video_len = video_pre["tokens"].shape[1]
        action_len = action_pre["tokens"].shape[1]
        attention_mask = torch.ones(
            video_len + action_len,
            video_len + action_len,
            device=device,
            dtype=torch.bool,
        )
        first_frame_tokens = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask[:first_frame_tokens, first_frame_tokens:video_len] = False
        attention_mask[:video_len, video_len:] = False
        attention_mask[video_len:, first_frame_tokens:video_len] = False
        outputs = mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        video_out = video_expert.post_dit(outputs["video"], video_pre)
        action_out = action_expert.post_dit(outputs["action"], action_pre)
        loss = video_out.float().square().mean() + action_out.float().square().mean()
        return loss, video_out, action_out

    loss_history = []
    gradient_norm = torch.tensor(float("nan"))
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss, video_out, action_out = forward_loss()
        loss.backward()
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in mot.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = torch.stack(gradients).norm() if gradients else gradient_norm
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise AssertionError(
                "non-finite structural optimization values: "
                f"loss={loss.item()}, grad_norm={gradient_norm.item()}"
            )
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    if loss_history[-1] >= loss_history[0]:
        raise AssertionError(
            f"tiny overfit loss did not decrease: initial={loss_history[0]}, final={loss_history[-1]}"
        )

    first_parameter = next(mot.parameters())
    expected_parameter = first_parameter.detach().clone()
    checkpoint_path = ROOT / "artifacts/autodl/tiny-structural.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": mot.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_bytes = checkpoint_path.stat().st_size
    with torch.no_grad():
        first_parameter.add_(1.0)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mot.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.testing.assert_close(first_parameter, expected_parameter)
    torch.cuda.synchronize()
    report = base_report("structural")
    report.update(
        {
            "status": "passed",
            "scope": "40-step tiny overfit with real WanVideoDiT + ActionDiT + MoT and checkpoint reload; no pretrained weights",
            "input_action_source": str(FIXTURE.relative_to(ROOT)),
            "video_output_shape": list(video_out.shape),
            "action_output_shape": list(action_out.shape),
            "loss": float(loss.detach().cpu()),
            "loss_initial": loss_history[0],
            "loss_final": loss_history[-1],
            "loss_ratio": loss_history[-1] / loss_history[0],
            "loss_trace": [loss_history[index] for index in (0, 1, 4, 9, 19, 39)],
            "overfit_steps": len(loss_history),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "optimizer": "AdamW",
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_reload_verified": True,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("import", "data", "structural"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runners = {"import": run_import, "data": run_data, "structural": run_structural}
    report = runners[args.stage]()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
