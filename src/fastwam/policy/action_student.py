"""Strict preprocessing and PolicyEngine adapter for an action-only student."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.image import fastwam_validation_image_transforms
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.distillation.manifest import ActionStudentServingManifest, sha256_file

from .server import PolicyPrediction, PolicyRequest

PANTHERA_AXES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
IMAGE_PREPROCESSING = "FastWAM ToTensor(uint8/255 float32) then torchvision.transforms.Resize(size=[H,W])"


class ActionStudentPreprocessor:
    """Serving transform matching FastWAM global min/max ``LinearNormalizer``."""

    def __init__(
        self,
        *,
        stats_path: str | Path,
        axes: tuple[str, ...] = PANTHERA_AXES,
        camera_order: tuple[str, ...] = ("overhead_rgb", "wrist_rgb"),
        image_sizes: tuple[tuple[int, int], ...] = ((224, 224), (224, 224)),
    ) -> None:
        if axes != PANTHERA_AXES:
            raise ValueError("Panthera axes/order mismatch")
        if camera_order != ("overhead_rgb", "wrist_rgb"):
            raise ValueError("camera order must be overhead_rgb then wrist_rgb")
        if len(image_sizes) != 2 or any(len(size) != 2 or min(size) <= 0 for size in image_sizes):
            raise ValueError("two positive per-camera image sizes are required")
        self.stats_path = Path(stats_path)
        self.axes = axes
        self.camera_order = camera_order
        self.image_sizes = image_sizes
        self.image_transforms = tuple(fastwam_validation_image_transforms(size) for size in image_sizes)
        try:
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
            state_stats = _global_stats(payload, "state")
            action_stats = _global_stats(payload, "action")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed dataset_stats JSON: {exc}") from exc
        self.state_normalizer = SingleFieldLinearNormalizer(state_stats, mode="min/max")
        self.action_normalizer = SingleFieldLinearNormalizer(action_stats, mode="min/max")

    def normalize_state(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = _finite_tensor(value, "state")
        return self.state_normalizer.forward(tensor)

    def normalize_action(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape[-1:] != (7,) or not torch.isfinite(tensor).all():
            raise ValueError("action must be finite with final dimension 7")
        return self.action_normalizer.forward(tensor)

    def denormalize_action(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape[-1:] != (7,) or not torch.isfinite(tensor).all():
            raise ValueError("normalized action must be finite with final dimension 7")
        result = self.action_normalizer.backward(tensor)
        if not torch.isfinite(result).all():
            raise ValueError("denormalized action is non-finite")
        return result

    def process_images(self, overhead: Image.Image, wrist: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self._image(overhead, 0), self._image(wrist, 1)

    def _image(self, image: Image.Image, camera_index: int) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().unsqueeze(0)
        for transform in self.image_transforms[camera_index]:
            tensor = transform(tensor)
        return tensor[0]


class ActionStudentPolicyModel:
    def __init__(
        self,
        *,
        model: nn.Module,
        text_cache_dir: str | Path,
        preprocessor: ActionStudentPreprocessor,
        serving_manifest: ActionStudentServingManifest,
        checkpoint_path: str | Path,
        deployment_manifest_path: str | Path,
        schema_path: str | Path,
        task_registry_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        detached = ActionStudentServingManifest.read(deployment_manifest_path)
        if detached != serving_manifest:
            raise ValueError("provided serving manifest does not match detached deployment manifest")
        task_registry = serving_manifest.validate_files(
            checkpoint_path=checkpoint_path,
            stats_path=preprocessor.stats_path,
            schema_path=schema_path,
            task_registry_path=task_registry_path,
        )
        model_type = getattr(model, "model_type", None)
        if model_type != "direct_bc_action_student":
            raise ValueError("serving model type mismatch")
        model_encoder_id = getattr(getattr(model, "vision_encoder", None), "encoder_id", None)
        if model_encoder_id != serving_manifest.vision_encoder_id:
            raise ValueError("serving vision encoder id mismatch")
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("serving model is missing DirectBC config")
        config.validate_manifest(serving_manifest)
        loaded_digest = getattr(model, "serving_manifest_digest", None)
        if loaded_digest != serving_manifest.digest():
            raise ValueError("loaded checkpoint/deployment manifest digest mismatch")
        if Path(getattr(model, "checkpoint_path", "")) != Path(checkpoint_path):
            raise ValueError("loaded model checkpoint path mismatch")
        if serving_manifest.stats_sha256 != sha256_file(preprocessor.stats_path):
            raise ValueError("serving preprocessing stats lineage mismatch")
        if serving_manifest.axes != preprocessor.axes:
            raise ValueError("serving preprocessing axes mismatch")
        if serving_manifest.camera_order != preprocessor.camera_order:
            raise ValueError("serving preprocessing camera order mismatch")
        if serving_manifest.image_sizes != preprocessor.image_sizes:
            raise ValueError("serving preprocessing image geometry mismatch")
        if serving_manifest.prompt_template != DEFAULT_PROMPT:
            raise ValueError("serving prompt template mismatch")
        if serving_manifest.image_preprocessing != IMAGE_PREPROCESSING:
            raise ValueError("serving image preprocessing mismatch")
        self.model = model.to(device).eval()
        self.text_cache_dir = Path(text_cache_dir)
        self.preprocessor = preprocessor
        self.manifest = serving_manifest
        self.task_registry = task_registry
        self.device = torch.device(device)
        self._text_context: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_text(
        self,
        task_id: str,
        canonical_task: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_registry.get(task_id) != canonical_task:
            raise ValueError("task id and canonical task do not match the validated task registry")
        cached = self._text_context.get(canonical_task)
        if cached is not None:
            return cached
        prompt = DEFAULT_PROMPT.format(task=canonical_task)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = (
            f"{digest}.t5_len{self.manifest.context_length}."
            f"{self.manifest.text_encoder_id}.pt"
        )
        path = self.text_cache_dir / filename
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context = payload["context"]
            mask = payload["mask"]
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid cached text context {path}: {exc}") from exc
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("text cache context/mask must be tensors")
        if context.ndim != 2 or context.shape[0] != self.manifest.context_length:
            raise ValueError("text context has incompatible shape")
        if mask.dtype != torch.bool or mask.shape != (self.manifest.context_length,):
            raise ValueError("text mask must be boolean with exact context length")
        if not torch.isfinite(context).all() or not bool(mask.any()):
            raise ValueError("text context must be finite with a nonempty mask")
        cached = (context.to(dtype=torch.float32).unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device))
        self._text_context[canonical_task] = cached
        return cached

    @torch.inference_mode()
    def infer(self, request: PolicyRequest) -> PolicyPrediction:
        overhead, wrist = self.preprocessor.process_images(request.overhead_rgb, request.wrist_rgb)
        state = self.preprocessor.normalize_state(request.state_position)
        context, mask = self._load_text(request.task_id, request.canonical_prompt)
        normalized = self.model(
            overhead.unsqueeze(0).to(self.device),
            wrist.unsqueeze(0).to(self.device),
            state.unsqueeze(0).to(self.device),
            context,
            mask,
        )[0]
        physical = self.preprocessor.denormalize_action(normalized.detach().cpu()).numpy()
        return PolicyPrediction(waypoint_positions=physical)


def _global_stats(payload: dict, kind: str) -> dict[str, torch.Tensor]:
    entry = payload[kind]["default"]
    result = {}
    for key in ("min", "max"):
        array = torch.as_tensor(entry[f"global_{key}"], dtype=torch.float32)
        if array.shape != (7,) or not torch.isfinite(array).all():
            raise ValueError(f"{kind} global_{key} must be finite with shape [7]")
        result[key] = array
    # Constant dimensions are intentionally supported exactly as LinearNormalizer supports them.
    return result


def _finite_tensor(value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (7,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite with shape [7]")
    return tensor
