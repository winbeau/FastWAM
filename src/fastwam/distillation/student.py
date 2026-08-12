"""Small action-only behavior-cloning student models and strict checkpoints."""

from __future__ import annotations

import os
import pickle
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .manifest import ActionStudentServingManifest

CHECKPOINT_VERSION = 3


@dataclass(frozen=True)
class DirectBCConfig:
    vision_dim: int = 32
    text_dim: int = 16
    hidden_dim: int = 64
    action_horizon: int = 32
    action_dim: int = 7
    state_dim: int = 7
    freeze_vision: bool = True

    def validate(self) -> None:
        if any(not isinstance(value, int) or value <= 0 for value in (self.vision_dim, self.text_dim, self.hidden_dim)):
            raise ValueError("DirectBC feature dimensions must be positive integers")
        if (self.action_horizon, self.action_dim, self.state_dim) != (32, 7, 7):
            raise ValueError("DirectBC geometry requires action_horizon/action_dim/state_dim [32,7,7]")

    def validate_manifest(self, manifest: ActionStudentServingManifest) -> None:
        self.validate()
        if (self.action_horizon, self.action_dim, self.state_dim) != (
            manifest.action_horizon, manifest.action_dim, manifest.state_dim
        ):
            raise ValueError("DirectBC config geometry does not match serving manifest")


class TinyVisionEncoder(nn.Module):
    """Download-free encoder intended only for fixtures and structural smoke tests."""

    encoder_id = "tiny-vision-fixture-v1"

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class DirectBCStudent(nn.Module):
    """Two-camera, state-and-text direct action chunk regressor."""

    model_type = "direct_bc_action_student"

    def __init__(self, vision_encoder: nn.Module, config: DirectBCConfig = DirectBCConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vision_encoder = vision_encoder
        if config.freeze_vision:
            self.vision_encoder.requires_grad_(False)
        fused_dim = config.vision_dim * 2 + config.text_dim + config.state_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_horizon * config.action_dim),
        )

    def train(self, mode: bool = True) -> "DirectBCStudent":
        super().train(mode)
        if self.config.freeze_vision:
            self.vision_encoder.eval()
        return self

    @staticmethod
    def masked_text_pool(context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or mask.shape != context.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("text context/mask must be [B,L,D], boolean [B,L]")
        if not torch.isfinite(context).all() or not bool(mask.any(dim=1).all()):
            raise ValueError("text context must be finite and every mask nonempty")
        valid = mask.to(dtype=context.dtype).unsqueeze(-1)
        return (context * valid).sum(dim=1) / valid.sum(dim=1)

    def forward(
        self,
        overhead_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        state: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if overhead_rgb.ndim != 4 or wrist_rgb.shape != overhead_rgb.shape:
            raise ValueError("camera tensors must have matching [B,C,H,W] shapes")
        if state.shape != (overhead_rgb.shape[0], self.config.state_dim):
            raise ValueError(f"state must have shape [B,{self.config.state_dim}]")
        if not all(torch.isfinite(value).all() for value in (overhead_rgb, wrist_rgb, state)):
            raise ValueError("model inputs must be finite")
        overhead = self.vision_encoder(overhead_rgb)
        wrist = self.vision_encoder(wrist_rgb)
        if overhead.shape[-1] != self.config.vision_dim or wrist.shape[-1] != self.config.vision_dim:
            raise ValueError("vision encoder output does not match config.vision_dim")
        text = self.masked_text_pool(text_context, text_mask)
        if text.shape[-1] != self.config.text_dim:
            raise ValueError("pooled text does not match config.text_dim")
        fused = torch.cat((overhead, wrist, state, text), dim=-1)
        return self.head(fused).reshape(
            overhead_rgb.shape[0], self.config.action_horizon, self.config.action_dim
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        serving_manifest: ActionStudentServingManifest,
        deployment_manifest_path: str | Path,
        stats_path: str | Path,
        schema_path: str | Path,
        task_registry_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> ActionStudentServingManifest:
        contract = serving_manifest.checkpoint_contract()
        contract.validate()
        self.config.validate_manifest(contract)
        encoder_id = getattr(self.vision_encoder, "encoder_id", None)
        if encoder_id != contract.vision_encoder_id:
            raise ValueError("vision encoder id does not match serving manifest")
        contract.validate_assets(
            stats_path=stats_path, schema_path=schema_path, task_registry_path=task_registry_path
        )
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "model_type": self.model_type,
            "vision_encoder_id": encoder_id,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "serving_contract": contract.to_dict(),
            "serving_contract_sha256": contract.digest(),
            "metadata": metadata or {},
        }
        path = Path(path)
        _atomic_torch_save(path, payload)
        deployment = contract.bind_checkpoint(path)
        deployment.validate_files(
            checkpoint_path=path, stats_path=stats_path, schema_path=schema_path,
            task_registry_path=task_registry_path,
        )
        deployment.write(deployment_manifest_path)
        return deployment

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        vision_encoder: nn.Module,
        deployment_manifest_path: str | Path,
        stats_path: str | Path,
        schema_path: str | Path,
        task_registry_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "DirectBCStudent":
        expected_manifest = ActionStudentServingManifest.read(deployment_manifest_path)
        expected_manifest.validate_files(
            checkpoint_path=path, stats_path=stats_path, schema_path=schema_path,
            task_registry_path=task_registry_path,
        )
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except (OSError, RuntimeError, EOFError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            raise ValueError(f"malformed or truncated action student checkpoint: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("malformed action student checkpoint payload")
        if payload.get("format_version") != CHECKPOINT_VERSION or payload.get("model_type") != cls.model_type:
            raise ValueError("unsupported action student checkpoint")
        try:
            actual_manifest = ActionStudentServingManifest.from_dict(payload["serving_contract"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed checkpoint serving contract: {exc}") from exc
        if actual_manifest.checkpoint_sha256 is not None:
            raise ValueError("checkpoint serving contract must not contain a self-hash")
        if payload.get("serving_contract_sha256") != actual_manifest.digest():
            raise ValueError("checkpoint serving contract digest mismatch")
        if actual_manifest != expected_manifest.checkpoint_contract():
            raise ValueError("checkpoint serving lineage mismatch")
        encoder_id = getattr(vision_encoder, "encoder_id", None)
        if payload.get("vision_encoder_id") != encoder_id or encoder_id != expected_manifest.vision_encoder_id:
            raise ValueError("checkpoint vision encoder id mismatch")
        try:
            config = DirectBCConfig(**payload["config"])
            config.validate_manifest(expected_manifest)
            model = cls(vision_encoder, config)
            model.load_state_dict(payload["state_dict"], strict=True)
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"malformed checkpoint model payload: {exc}") from exc
        model.serving_manifest = expected_manifest
        model.serving_manifest_digest = expected_manifest.digest()
        model.checkpoint_path = Path(path)
        model.deployment_manifest_path = Path(deployment_manifest_path)
        model.dataset_lineage_sha256 = expected_manifest.dataset_lineage_sha256
        return model


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
