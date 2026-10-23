"""FastWAM-Joint inference for the sparse-acceleration case study.

``JointCaseStudyRunner.infer_action`` reproduces ``FastWAMJoint.infer_action`` with two
exact restructurings shared by every arm, then applies the arm's method:

1. Frame-0 reuse. Frame-0 latents are pinned to the encoded observation, carry timestep
   0 and (``first_frame_causal``) attend only to themselves, so their hidden states are
   identical at every denoising step. They are run once per call and their per-layer
   K/V reused; only future video tokens and action tokens are recomputed per step.
2. Prompt reuse. The umt5 encoding of an instruction is cached for the episode.

Per step the joint sequence is ``[future video (kept) | action]`` with keys
``[frame-0 | future (kept) | action]``. Video rows never see action keys and action
rows see everything, exactly as ``FastWAMJoint._build_mot_attention_mask``; the two row
groups are run as two unmasked attention calls.

Methods (see ``config.py``): token pruning of future video tokens after warm-up (pruned
tokens leave the sequence and their latents keep moving with their last velocity),
TeaCache step skipping of the whole joint stack, and attention-output noise on the
action stream. ``frame0_cache=false`` (dense only) calls the model's native
``infer_action`` unchanged, as the reference for equivalence checks.

Execution options (``execution=`` of ``make_runner``; all off by default, each must reproduce
the default path bit for bit and is verified on paired episodes before use):

* ``cross_kv_cache``: the cross-attention K/V of the (constant) instruction context are
  computed once per call per layer instead of at every denoising step.
* ``defer_diagnostics``: diagnostic tensors stay on the GPU during the call and are converted
  after the final action copy (which synchronises anyway), instead of forcing mid-call syncs.
* ``cuda_graphs`` (needs ``cross_kv_cache``): the layer stack of a denoising step without
  attention capture, noise or TeaCache is replayed from a CUDA graph per (mode, token count,
  context length); graphs are built from a runner's second call on, so one-call override
  runners stay eager.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate

from .config import CaseStudyConfig
from . import selection as sel


def _f16_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float16).tobytes()).decode()


class JointCaseStudyRunner:
    def __init__(self, model, config: CaseStudyConfig, prompt_store: Optional[dict] = None,
                 execution: Optional[dict] = None):
        self.model = model
        self.cfg = config
        self.exec = {"cross_kv_cache": False, "defer_diagnostics": False, "cuda_graphs": False, **(execution or {})}
        unknown = set(self.exec) - {"cross_kv_cache", "defer_diagnostics", "cuda_graphs"}
        if unknown:
            raise ValueError(f"unknown execution options: {sorted(unknown)}")
        if self.exec["cuda_graphs"] and not self.exec["cross_kv_cache"]:
            raise ValueError("execution.cuda_graphs needs execution.cross_kv_cache")
        self._graphs: dict[tuple, "_StepGraph"] = {}
        self._calls_done = 0
        #: Precomputed ``encode_prompt`` outputs keyed by full prompt
        #: (``scripts/casestudy/precompute_prompts.py``); lets the model run without its
        #: text encoder. A prompt missing from the store is an error, never a fallback.
        self.prompt_store = prompt_store
        self.mot = model.mot
        self.video = model.video_expert
        self.action = model.action_expert
        self.num_heads = int(self.mot.num_heads)
        self.num_layers = int(self.mot.num_layers)
        if config.method != "dense" and not config.frame0_cache:
            raise ValueError("frame0_cache=false is the native reference path and supports method=dense only")
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._layers: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------ episode
    def reset_episode(self) -> None:
        """Per-episode state: prompt cache and the CF layer choices (frozen per episode)."""
        self._prompt_cache.clear()
        self._layers = None

    # ------------------------------------------------------------------ helpers
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.cfg.prompt_cache and prompt in self._prompt_cache:
            return self._prompt_cache[prompt]
        if self.prompt_store is not None:
            if prompt not in self.prompt_store:
                raise KeyError(f"prompt not in precomputed store: {prompt!r}")
            item = self.prompt_store[prompt]
            entry = (
                item["context"].to(device=self.model.device),
                item["mask"].to(device=self.model.device, dtype=torch.bool),
                int(item["num_text"]),
            )
            self._prompt_cache[prompt] = entry
            return entry
        context, context_mask = self.model.encode_prompt(prompt)
        _ids, raw_mask = self.model.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        num_text = int(raw_mask.gt(0).sum().item())
        entry = (context, context_mask, num_text)
        if self.cfg.prompt_cache:
            self._prompt_cache[prompt] = entry
        return entry

    def _attention_probs(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Head-averaged softmax(q k^T / sqrt(d)) for batch 1 -> [Sq, Sk] (fp32).

        Used only for scores/diagnostics; the forward pass keeps the SDPA output so
        capture does not change the numerics of the arm.
        """
        qh = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        kh = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        logits = (qh.float() @ kh.float().transpose(-1, -2)) * (qh.shape[-1] ** -0.5)
        return torch.softmax(logits, dim=-1).mean(dim=1)[0]

    def _post(self, block, io: tuple, mixed: torch.Tensor, context_payload: dict) -> torch.Tensor:
        _q, _k, _v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, _ckpt = io
        return self.mot._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=False,
            mixed_slice=mixed,
            context_payload=context_payload,
        )

    def _cross_kv(self, expert, context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer cross-attention K/V of a projected context (what every block computes
        from the same context at every step)."""
        out = []
        for layer in range(self.num_layers):
            ca = expert.blocks[layer].cross_attn
            out.append((ca.norm_k(ca.k(context)), ca.v(context)))
        return out

    def _post_cached(self, block, io: tuple, mixed: torch.Tensor, mask: Optional[torch.Tensor],
                     ckv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """``_post`` with the cross-attention K/V taken from ``ckv``; otherwise the same ops in
        the same order as ``MoT._apply_expert_post_block``."""
        _q, _k, _v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, _ckpt = io
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        ca = block.cross_attn
        q = ca.norm_q(ca.q(block.norm3(x)))
        x = x + ca.o(flash_attention(q=q, k=ckv[0], v=ckv[1], num_heads=ca.num_heads, ctx_mask=mask))
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        return block.gate(x, gate_mlp, block.ffn(mlp_input))

    def _stack(self, xv, xa, freqs_v, tmod_v, mask_v, freqs_a, tmod_a, mask_a, kv0, ckv_v, ckv_a):
        """One denoising step's layer stack without capture, noise or TeaCache (the loop of
        ``infer_action`` with cached cross K/V); the body of a step CUDA graph."""
        for layer in range(self.num_layers):
            bv = self.video.blocks[layer]
            ba = self.action.blocks[layer]
            io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=freqs_v, t_mod=tmod_v)
            io_a = self.mot._build_expert_attention_io(expert=self.action, block=ba, x=xa, freqs=freqs_a, t_mod=tmod_a)
            k_vid = torch.cat([kv0[layer]["k"], io_v[1]], dim=1)
            v_vid = torch.cat([kv0[layer]["v"], io_v[2]], dim=1)
            out_v = flash_attention(q=io_v[0], k=k_vid, v=v_vid, num_heads=self.num_heads)
            k_all = torch.cat([k_vid, io_a[1]], dim=1)
            v_all = torch.cat([v_vid, io_a[2]], dim=1)
            out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads)
            xv = self._post_cached(bv, io_v, out_v, mask_v, ckv_v[layer])
            xa = self._post_cached(ba, io_a, out_a, mask_a, ckv_a[layer])
        return xv, xa

    @staticmethod
    def _entropy_t(dist: torch.Tensor) -> torch.Tensor:
        """``selection.entropy`` without the host sync (same values)."""
        x = dist.float().clamp_min(0)
        total = x.sum()
        p = torch.where(total > 0, x / total, torch.full_like(x, 1.0 / x.numel()))
        return -(p * torch.log(p.clamp_min(1e-12))).sum()

    def _gate_entropies(self, p_act: torch.Tensor, grid) -> torch.Tensor:
        """[h_third, h_wrist, h_early, h_late] of ``selection.gate_score`` as one GPU tensor."""
        third, wrist = grid.future_view_masks(p_act.device)
        start = grid.per_frame
        fut = p_act[:, start : start + grid.num_future].float()
        mass = fut.sum(dim=0)
        half = p_act.shape[0] // 2
        return torch.stack([self._entropy_t(mass[third]), self._entropy_t(mass[wrist]),
                            self._entropy_t(fut[:half].sum(dim=0)), self._entropy_t(fut[half:].sum(dim=0))])

    @staticmethod
    def _gate_dict(h: list[float]) -> dict[str, float]:
        h_third, h_wrist, h_early, h_late = h
        return {"h_third": h_third, "h_wrist": h_wrist, "h_early": h_early, "h_late": h_late,
                "c_view": h_third - h_wrist, "c_temp": h_early - h_late, "c": (h_third - h_wrist) + (h_early - h_late)}

    def _prefill_frame0(self, pre0: dict, num_text: int, inst_layers: set[int],
                        ckv_v: Optional[list] = None) -> tuple[list, dict[int, torch.Tensor]]:
        """Run frame-0 tokens alone through the video expert; return per-layer K/V and,
        for ``inst_layers``, the cross-attention mass on real instruction tokens."""
        x = pre0["tokens"]
        payload = {"context": pre0["context"], "mask": pre0["context_mask"]}
        kv: list[dict[str, torch.Tensor]] = []
        inst_mass: dict[int, torch.Tensor] = {}
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            io = self.mot._build_expert_attention_io(
                expert=self.video, block=block, x=x, freqs=pre0["freqs"], t_mod=pre0["t_mod"]
            )
            q, k, v = io[0], io[1], io[2]
            mixed = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
            if layer in inst_layers:
                x_mid = block.gate(io[3], io[4], block.self_attn.o(mixed))
                cq = block.cross_attn.norm_q(block.cross_attn.q(block.norm3(x_mid)))
                ck = block.cross_attn.norm_k(block.cross_attn.k(pre0["context"]))
                probs = self._attention_probs(cq, ck)  # [Kv0, L]
                inst_mass[layer] = probs[:, :num_text].sum(dim=-1)
            if ckv_v is not None:
                x = self._post_cached(block, io, mixed, pre0["context_mask"], ckv_v[layer])
            else:
                x = self._post(block, io, mixed, payload)
            kv.append({"k": k, "v": v})
        return kv, inst_mass

    def _choose_layers(self, act_probs: dict[int, torch.Tensor], inst_mass: dict[int, torch.Tensor], grid: sel.Grid) -> dict[str, int]:
        prune = self.cfg.prune
        if prune is not None and prune.layer_rule == "fixed":
            return {"act": int(prune.layer), "inst": int(prune.layer)}
        # CF-Prune: the layer whose attention is most concentrated (minimum entropy) at the
        # first policy query of the episode, frozen for the rest of the episode.
        act = min(act_probs, key=lambda l: sel.entropy(sel.future_mass(act_probs[l], grid)))
        inst = min(inst_mass, key=lambda l: sel.entropy(inst_mass[l])) if inst_mass else act
        return {"act": int(act), "inst": int(inst)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def infer_action(
        self,
        prompt: str,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        action_horizon: int,
        num_video_frames: int,
        seed: int,
        sigma_shift: Optional[float] = None,
        rand_device: str = "cpu",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.cfg
        model = self.model
        steps = int(cfg.num_inference_steps)
        if not cfg.frame0_cache:
            text = {"prompt": prompt}
            if self.prompt_store is not None:
                context, context_mask, _ = self._encode_prompt(prompt)
                text = {"prompt": None, "context": context, "context_mask": context_mask}
            out = model.infer_action(
                **text,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                num_inference_steps=steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            return out["action"], {"path": "native", "seed": int(seed), "steps": steps}

        model.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        _, _, height, width = input_image.shape
        if proprio is not None:
            proprio = proprio.reshape(1, -1).to(device=model.device, dtype=model.torch_dtype)

        vae = model.vae
        latent_t = (num_video_frames - 1) // vae.temporal_downsample_factor + 1
        latent_h = height // vae.upsampling_factor
        latent_w = width // vae.upsampling_factor
        patch = self.video.patch_size
        grid = sel.Grid(frames=latent_t // int(patch[0]), height=latent_h // int(patch[1]), width=latent_w // int(patch[2]))
        f0 = grid.per_frame
        num_future = grid.num_future

        # Same noise draw as the native path: two CPU generators with the same seed.
        gen_v = torch.Generator(device=rand_device).manual_seed(int(seed))
        gen_a = torch.Generator(device=rand_device).manual_seed(int(seed))
        latents_video = torch.randn(
            (1, vae.model.z_dim, latent_t, latent_h, latent_w), generator=gen_v, device=rand_device, dtype=torch.float32
        ).to(device=model.device, dtype=model.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action.action_dim), generator=gen_a, device=rand_device, dtype=torch.float32
        ).to(device=model.device, dtype=model.torch_dtype)

        input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
        first = model._encode_input_image_latents_tensor(input_image=input_image, tiled=False)
        latents_video[:, :, 0:1] = first.clone()
        fuse = bool(getattr(self.video, "fuse_vae_embedding_in_latents", False))

        context, context_mask, num_text = self._encode_prompt(prompt)
        if proprio is not None:
            context, context_mask = model._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)

        t_video, d_video = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_video.dtype, shift_override=sigma_shift
        )
        t_action, d_action = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_action.dtype, shift_override=sigma_shift
        )

        prune = cfg.prune
        is_cf = prune is not None and prune.selection == "cf"
        first_call = self._layers is None
        want_scores = prune is not None and prune.selection in ("av", "cf")
        capture_first_step = cfg.record_attention or want_scores
        # Layer choice needs every layer on the episode's first call; afterwards one layer.
        all_layers = set(range(self.num_layers))
        if first_call:
            act_layers = all_layers if capture_first_step else set()
            inst_layers = all_layers if is_cf else set()
        else:
            act_layers = {self._layers["act"]} if capture_first_step else set()
            inst_layers = {self._layers["inst"]} if is_cf else set()

        # ---- frame-0 prefill (once per call)
        pre0 = self.video.pre_dit(
            x=first,
            timestep=t_video[0].unsqueeze(0).to(dtype=latents_video.dtype, device=model.device),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse,
        )
        ckv_v = self._cross_kv(self.video, pre0["context"]) if self.exec["cross_kv_cache"] else None
        ckv_a = None
        kv0, inst_mass = self._prefill_frame0(pre0, num_text, inst_layers, ckv_v)
        defer = self.exec["defer_diagnostics"]
        pending: dict[str, Any] = {}
        call_token = object()

        # ---- denoising
        all_future = torch.arange(num_future, device=model.device)
        kept = all_future
        gated_dense = False
        vel_tokens: Optional[torch.Tensor] = None
        noise_gen = None
        if cfg.method == "noise" and cfg.noise.sigma > 0:
            noise_gen = torch.Generator(device=model.device).manual_seed((int(seed) ^ 0x5EED) & 0x7FFFFFFF)
        tea = cfg.teacache
        tea_prev: Optional[torch.Tensor] = None
        tea_acc = 0.0
        tea_consecutive = 0
        tea_res_v: Optional[torch.Tensor] = None
        tea_res_a: Optional[torch.Tensor] = None
        tea_poly = np.poly1d(tea.coefficients) if tea is not None else None

        modes: list[str] = []
        tea_rel: list[float] = []
        diag: dict[str, Any] = {"path": "casestudy", "seed": int(seed), "steps": steps, "method": cfg.method}
        act_probs_step0: dict[int, torch.Tensor] = {}
        k_future_store: dict[int, torch.Tensor] = {}

        for s in range(steps):
            ts_v = t_video[s].unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
            ts_a = t_action[s].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
            pre_v = self.video.pre_dit(
                x=latents_video, timestep=ts_v, context=context, context_mask=context_mask,
                action=None, fuse_vae_embedding_in_latents=fuse,
            )
            pre_a = self.action.pre_dit(action_tokens=latents_action, timestep=ts_a, context=context, context_mask=context_mask)

            # ---- mode for this step
            mode = "full"
            if prune is not None and s >= prune.warmup_steps and not gated_dense and s not in prune.full_steps:
                mode = "pruned"
            if tea is not None:
                block0 = self.video.blocks[0]
                fut_tok = pre_v["tokens"][:, f0:]
                shift_msa, scale_msa, *_ = self.mot._split_modulation(block0, pre_v["t_mod"][:, f0:])
                signal = modulate(block0.norm1(fut_tok), shift_msa, scale_msa)
                if tea_prev is not None:
                    rel = float(((signal - tea_prev).abs().mean() / tea_prev.abs().mean().clamp_min(1e-12)).item())
                    tea_rel.append(rel)
                    if 0 < s < steps - 1:
                        tea_acc += float(tea_poly(rel))
                        if tea_acc < tea.thresh and tea_consecutive < tea.max_consecutive_skips and tea_res_v is not None:
                            mode = "skip"
                tea_prev = signal

            idx_rel = kept if mode == "pruned" else all_future
            idx_abs = idx_rel + f0
            xv = pre_v["tokens"][:, idx_abs]
            xa = pre_a["tokens"]
            xv_in, xa_in = xv, xa

            if mode == "skip":
                xv = xv_in + tea_res_v
                xa = xa_in + tea_res_a
                tea_consecutive += 1
            else:
                capture_layers = act_layers if (s == 0 and capture_first_step) else set()
                if mode == "pruned" and prune.reselect_every_step:
                    capture_layers = {self._layers["act"]}
                freqs_v = pre_v["freqs"][idx_abs]
                tmod_v = pre_v["t_mod"][:, idx_abs]
                payload_v = {"context": pre_v["context"], "mask": pre_v["context_mask"][:, idx_abs]}
                payload_a = {"context": pre_a["context"], "mask": pre_a["context_mask"]}
                step_probs: dict[int, torch.Tensor] = {}
                if ckv_v is not None and ckv_a is None:
                    ckv_a = self._cross_kv(self.action, pre_a["context"])
                use_graph = (self.exec["cuda_graphs"] and self._calls_done >= 1 and not capture_layers
                             and noise_gen is None and tea is None)
                if use_graph:
                    key = (mode, int(xv.shape[1]), int(ckv_v[0][0].shape[1]), int(ckv_a[0][0].shape[1]))
                    args = (xv, xa, freqs_v, tmod_v, payload_v["mask"], pre_a["freqs"], pre_a["t_mod"], payload_a["mask"])
                    graph = self._graphs.get(key)
                    if graph is None:
                        graph = self._graphs[key] = _StepGraph(self, args, kv0, ckv_v, ckv_a)
                    graph.set_call_constants(call_token, kv0, ckv_v, ckv_a)
                    xv, xa = graph(args)
                for layer in (range(0) if use_graph else range(self.num_layers)):
                    bv = self.video.blocks[layer]
                    ba = self.action.blocks[layer]
                    io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=freqs_v, t_mod=tmod_v)
                    io_a = self.mot._build_expert_attention_io(
                        expert=self.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
                    )
                    k_vid = torch.cat([kv0[layer]["k"], io_v[1]], dim=1)
                    v_vid = torch.cat([kv0[layer]["v"], io_v[2]], dim=1)
                    out_v = flash_attention(q=io_v[0], k=k_vid, v=v_vid, num_heads=self.num_heads)
                    k_all = torch.cat([k_vid, io_a[1]], dim=1)
                    v_all = torch.cat([v_vid, io_a[2]], dim=1)
                    out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads)
                    if layer in capture_layers:
                        if idx_rel.numel() != num_future:
                            # P2 reselection on a pruned step: score every future token, with
                            # fresh keys for kept tokens and warm-up keys for pruned ones.
                            k_future = k_future_store[layer]
                            k_future[:, idx_rel] = io_v[1]
                            probs = self._attention_probs(
                                io_a[0], torch.cat([kv0[layer]["k"], k_future, io_a[1]], dim=1)
                            )
                        else:
                            probs = self._attention_probs(io_a[0], k_all)  # [A, f0 + Kf + A]
                            if s == 0 and prune is not None and prune.reselect_every_step:
                                k_future_store[layer] = io_v[1].clone()
                        step_probs[layer] = probs
                    if noise_gen is not None:
                        rms = out_a.float().pow(2).mean().sqrt()
                        out_a = out_a + (cfg.noise.sigma * rms * torch.randn(
                            out_a.shape, generator=noise_gen, device=out_a.device, dtype=torch.float32
                        )).to(out_a.dtype)
                    if ckv_v is not None:
                        xv = self._post_cached(bv, io_v, out_v, payload_v["mask"], ckv_v[layer])
                        xa = self._post_cached(ba, io_a, out_a, payload_a["mask"], ckv_a[layer])
                    else:
                        xv = self._post(bv, io_v, out_v, payload_v)
                        xa = self._post(ba, io_a, out_a, payload_a)
                if tea is not None:
                    tea_res_v = xv - xv_in
                    tea_res_a = xa - xa_in
                    tea_acc = 0.0
                    tea_consecutive = 0
                if s == 0:
                    act_probs_step0 = step_probs

            # ---- heads and scheduler step
            vel_part = self.video.head(xv, pre_v["t"][:, idx_abs])
            if vel_tokens is None:
                vel_tokens = torch.zeros((1, f0 + num_future, vel_part.shape[-1]), device=vel_part.device, dtype=vel_part.dtype)
            vel_tokens[:, idx_abs] = vel_part  # pruned tokens keep their last velocity
            pred_video = self.video.unpatchify(vel_tokens, pre_v["meta"]["grid_size"])
            pred_action = self.action.post_dit(xa, pre_a)
            latents_video = model.infer_video_scheduler.step(pred_video, d_video[s], latents_video)
            latents_action = model.infer_action_scheduler.step(pred_action, d_action[s], latents_action)
            latents_video[:, :, 0:1] = first.clone()
            modes.append(mode)

            # ---- after warm-up: layer choice, gate, kept set
            if s == 0 and act_probs_step0:
                if first_call:
                    self._layers = self._choose_layers(act_probs_step0, inst_mass, grid)
                p_act = act_probs_step0[self._layers["act"]]
                diag["layer_act"] = self._layers["act"]
                av = sel.av_scores(p_act, grid)
                if defer:
                    pending["gate"] = self._gate_entropies(p_act, grid)
                    pending["av"] = av
                    pending["av_mass_obs"] = p_act[:, :f0].float().sum()
                else:
                    diag["gate_score"] = sel.gate_score(p_act, grid)
                    diag["av_mass_future"] = float(av.sum().item())
                    diag["av_mass_obs"] = float(p_act[:, :f0].float().sum().item())
                    diag["av_scores_f16"] = _f16_b64(av)
            elif s == 0 and first_call:
                self._layers = {"act": -1, "inst": -1}
            if prune is not None and (s == prune.warmup_steps - 1 or (mode == "pruned" and prune.reselect_every_step)):
                k = max(1, int(round(prune.keep_ratio * num_future)))
                if prune.selection == "random":
                    sel_gen = torch.Generator(device="cpu").manual_seed((int(seed) + 1) & 0x7FFFFFFF)
                    kept = sel.random_indices(num_future, k, sel_gen, model.device)
                else:
                    p_src = step_probs if (mode == "pruned" and prune.reselect_every_step) else act_probs_step0
                    p_act = p_src[self._layers["act"]]
                    if prune.selection == "av":
                        kept = sel.topk_indices(sel.av_scores(p_act, grid), k)
                    else:
                        s_act = sel.cf_action_scores(p_act, grid)
                        s_inst = sel.cf_instruction_scores(inst_mass[self._layers["inst"]], grid)
                        kept = sel.complementary_topk(s_act, s_inst, k)
                        diag["layer_inst"] = self._layers["inst"]
                    if prune.gate is not None and s == prune.warmup_steps - 1:
                        c = sel.gate_score(p_act, grid)["c"]
                        gated_dense = bool(c > prune.gate.tau)
                        diag["gated_dense"] = gated_dense
                if prune.extra_keep_ratio > 0:
                    dropped_mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    dropped_mask[kept] = False
                    dropped = torch.nonzero(dropped_mask).flatten()
                    m = int(round(prune.extra_keep_ratio * dropped.numel()))
                    if m > 0:
                        if prune.extra_selection == "random":
                            g = torch.Generator(device="cpu").manual_seed((int(seed) + 2) & 0x7FFFFFFF)
                            extra = dropped[torch.randperm(dropped.numel(), generator=g)[:m].to(dropped.device)]
                        else:
                            av_all = sel.av_scores(act_probs_step0[self._layers["act"]], grid)
                            extra = dropped[torch.topk(av_all[dropped], m).indices]
                        kept = torch.cat([kept, extra])
                        diag["extra_kept"] = int(m)
                kept = torch.sort(kept).values
                if s == prune.warmup_steps - 1:
                    av = sel.av_scores(act_probs_step0[self._layers["act"]], grid) if act_probs_step0 else None
                    diag["kept_count"] = int(kept.numel())
                    if defer:
                        pending["kept"] = kept
                        pending["kept_av"] = av
                    else:
                        diag["kept_index"] = kept.tolist()
                        if av is not None and float(av.sum()) > 0:
                            diag["kept_av_mass_fraction"] = float((av[kept].sum() / av.sum()).item())

        diag["modes"] = modes
        diag["computed_steps"] = sum(1 for m in modes if m != "skip")
        if prune is not None:
            pruned_steps = sum(1 for m in modes if m == "pruned")
            n_kept = int(kept.numel())
            diag["future_token_steps"] = (steps - pruned_steps) * num_future + pruned_steps * n_kept
            diag["future_token_steps_dense"] = steps * num_future
        if tea is not None:
            diag["teacache_rel_l1"] = [round(r, 6) for r in tea_rel]
        if noise_gen is not None:
            diag["noise_sigma"] = cfg.noise.sigma
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if defer:  # the copy above synchronised; convert in the order the default path writes
            ordered = {}
            for k_, v_ in diag.items():
                ordered[k_] = v_
                if k_ == "layer_act" and "gate" in pending:
                    ordered["gate_score"] = self._gate_dict(pending["gate"].tolist())
                    ordered["av_mass_future"] = float(pending["av"].sum().item())
                    ordered["av_mass_obs"] = float(pending["av_mass_obs"].item())
                    ordered["av_scores_f16"] = _f16_b64(pending["av"])
                if k_ == "kept_count" and "kept" in pending:
                    ordered["kept_index"] = pending["kept"].tolist()
                    av_ = pending["kept_av"]
                    if av_ is not None and float(av_.sum()) > 0:
                        ordered["kept_av_mass_fraction"] = float((av_[pending["kept"]].sum() / av_.sum()).item())
            diag = ordered
        self._calls_done += 1
        return action_out, diag


def timed(fn, *args, **kwargs):
    """Run ``fn`` with CUDA synchronisation on both sides; return (result, seconds)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


__all__ = ["JointCaseStudyRunner", "timed"]


def _static_copy(src: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """Static buffer for ``src`` plus a function giving a view with ``src``'s exact strides
    (broadcast dims are stored once and expanded, as in the eager tensors)."""
    idx = tuple(slice(0, 1) if (st == 0 and sz > 1) else slice(None) for st, sz in zip(src.stride(), src.shape))
    base = src[idx].clone()
    shape = tuple(src.shape)
    expanded = base.shape != src.shape
    return base, (lambda: base.expand(shape)) if expanded else (lambda: base)


class _StepGraph:
    """CUDA graph of ``JointCaseStudyRunner._stack`` for fixed shapes.

    Per-step inputs are copied into static buffers before each replay; per-call constants
    (frame-0 K/V, cross-attention K/V) live in static buffers refreshed once per call.
    """

    def __init__(self, runner: JointCaseStudyRunner, args: tuple, kv0: list, ckv_v: list, ckv_a: list):
        self.inp = [_static_copy(t) for t in args]
        self.kv0 = [{"k": d["k"].clone(), "v": d["v"].clone()} for d in kv0]
        self.ckv_v = [(k.clone(), v.clone()) for k, v in ckv_v]
        self.ckv_a = [(k.clone(), v.clone()) for k, v in ckv_a]
        self.token = None

        def run():
            views = [fn() for _base, fn in self.inp]
            return runner._stack(*views, self.kv0, self.ckv_v, self.ckv_a)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = run()

    def set_call_constants(self, token, kv0: list, ckv_v: list, ckv_a: list) -> None:
        if token is self.token:
            return
        for dst, src in zip(self.kv0, kv0):
            dst["k"].copy_(src["k"])
            dst["v"].copy_(src["v"])
        for (dk, dv), (sk, sv) in zip(self.ckv_v + self.ckv_a, ckv_v + ckv_a):
            dk.copy_(sk)
            dv.copy_(sv)
        self.token = token

    def __call__(self, args: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        for (base, fn), src in zip(self.inp, args):
            idx = tuple(slice(0, 1) if (st == 0 and sz > 1) else slice(None) for st, sz in zip(src.stride(), src.shape))
            base.copy_(src[idx])
        self.graph.replay()
        return self.out[0].clone(), self.out[1].clone()
