"""Padding-aware offline action metrics and synchronized latency measurement."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch


def offline_action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> dict[str, float | int | list[float]]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,T,D] shape")
    if padding_mask.shape != prediction.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean with shape [B,T]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("metric inputs must be finite")
    valid_steps = ~padding_mask
    if not bool(valid_steps.any()):
        raise ValueError("metrics require at least one valid action step")
    error = (prediction - target).double()
    selected = error[valid_steps]
    per_axis_mae = selected.abs().mean(dim=0)
    per_axis_rmse = selected.square().mean(dim=0).sqrt()
    pair_valid = valid_steps[:, 1:] & valid_steps[:, :-1]
    deltas = prediction[:, 1:] - prediction[:, :-1]
    smooth = deltas[pair_valid].double()
    return {
        "valid_steps": int(valid_steps.sum().item()),
        "valid_elements": int(valid_steps.sum().item() * prediction.shape[-1]),
        "mae": float(selected.abs().mean().item()),
        "rmse": float(selected.square().mean().sqrt().item()),
        "max_abs": float(selected.abs().max().item()),
        "per_axis_mae": per_axis_mae.tolist(),
        "per_axis_rmse": per_axis_rmse.tolist(),
        "temporal_delta_l2": 0.0
        if smooth.numel() == 0
        else float(smooth.square().sum(dim=-1).sqrt().mean().item()),
    }


def benchmark_latency(
    function: Callable[[], object],
    *,
    warmup: int = 2,
    iterations: int = 10,
    device: str | torch.device = "cpu",
    dtype: str = "float32",
    scope: str = "model_forward",
    structural_only: bool = False,
) -> dict[str, float | int | str | bool]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    measured_device = torch.device(device)

    def synchronize() -> None:
        if measured_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA latency requested but CUDA is unavailable")
            torch.cuda.synchronize(measured_device)

    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize()
        samples = []
        for _ in range(iterations):
            synchronize()
            started = time.perf_counter_ns()
            function()
            synchronize()
            samples.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(samples, dtype=np.float64)
    result: dict[str, float | int | str | bool] = {
        "scope": scope,
        "device": str(measured_device),
        "dtype": dtype,
        "warmup": warmup,
        "iterations": iterations,
        "structural_only": structural_only,
        "latency_ms_mean": float(values.mean()),
    }
    for percentile in (50, 90, 95, 99):
        result[f"latency_ms_p{percentile}"] = float(np.percentile(values, percentile))
    return result
