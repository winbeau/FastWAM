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
    def __init__(self, model, config: CaseStudyConfig, prompt_store: Optional[dict] = None):
        self.model = model
        self.cfg = config
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

    def _prefill_frame0(self, pre0: dict, num_text: int, inst_layers: set[int]) -> tuple[list, dict[int, torch.Tensor]]:
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
        want_scores = prune is not None and prune.selection in ("av", "vv", "cf")
        # V→V mass: dumped on dense score runs (record_vv) and needed by selection=vv.
        want_vv = cfg.record_vv or (prune is not None and prune.selection == "vv")
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
        kv0, inst_mass = self._prefill_frame0(pre0, num_text, inst_layers)

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
        diag: dict[str, Any] = {
            "path": "casestudy", "seed": int(seed), "steps": steps, "method": cfg.method,
            "grid": [int(grid.frames), int(grid.height), int(grid.width)],
        }
        act_probs_step0: dict[int, torch.Tensor] = {}
        vid_step_probs: dict[int, torch.Tensor] = {}
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
                for layer in range(self.num_layers):
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
                        if want_vv and mode == "full":
                            # Video branch attends with k = [frame-0 | future kept], no action keys.
                            vid_step_probs[layer] = self._attention_probs(io_v[0], k_vid)
                    if noise_gen is not None:
                        rms = out_a.float().pow(2).mean().sqrt()
                        out_a = out_a + (cfg.noise.sigma * rms * torch.randn(
                            out_a.shape, generator=noise_gen, device=out_a.device, dtype=torch.float32
                        )).to(out_a.dtype)
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
                diag["gate_score"] = sel.gate_score(p_act, grid)
                av = sel.av_scores(p_act, grid)
                diag["av_mass_future"] = float(av.sum().item())
                diag["av_mass_obs"] = float(p_act[:, :f0].float().sum().item())
                diag["av_scores_f16"] = _f16_b64(av)
                if want_vv and vid_step_probs:
                    vv_layer = self._layers.get("act", 0)
                    if vv_layer not in vid_step_probs:
                        vv_layer = int(min(vid_step_probs))
                    vv = sel.vv_scores(vid_step_probs[vv_layer], grid)
                    diag["vv_layer"] = int(vv_layer)
                    diag["vv_scores_f16"] = _f16_b64(vv)
                    diag["vv_mass_future"] = float(vv.sum().item())
            elif s == 0 and first_call:
                # Dense arms with record_attention log A→V/V→V scores on every call. The layer
                # is either fixed by score_layer or chosen by the pruning rule (min entropy).
                if cfg.score_layer is not None:
                    self._layers = {"act": int(cfg.score_layer), "inst": -1}
                elif cfg.record_attention and act_probs_step0:
                    self._layers = {
                        "act": int(min(act_probs_step0, key=lambda l: sel.entropy(sel.future_mass(act_probs_step0[l], grid)))),
                        "inst": -1,
                    }
                else:
                    self._layers = {"act": -1, "inst": -1}
            if prune is not None and (s == prune.warmup_steps - 1 or (mode == "pruned" and prune.reselect_every_step)):
                k = max(1, int(round(prune.keep_ratio * num_future)))
                if prune.kept_index is not None:
                    # Keep-set taken from a dense rollout: no score is computed here.
                    kept = torch.tensor(
                        sorted({int(i) for i in prune.kept_index if 0 <= int(i) < num_future}),
                        device=model.device, dtype=torch.long,
                    )
                    if kept.numel() == 0:
                        kept = all_future[:k]
                elif prune.selection == "random":
                    sel_gen = torch.Generator(device="cpu").manual_seed((int(seed) + 1) & 0x7FFFFFFF)
                    kept = sel.random_indices(num_future, k, sel_gen, model.device)
                else:
                    p_src = step_probs if (mode == "pruned" and prune.reselect_every_step) else act_probs_step0
                    p_act = p_src[self._layers["act"]]
                    if prune.selection == "vv":
                        p_vid = vid_step_probs.get(self._layers["act"])
                        kept = (sel.topk_indices(sel.vv_scores(p_vid, grid), k)
                                if p_vid is not None else all_future[:k])
                    elif prune.selection == "av":
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
        return latents_action[0].detach().to(device="cpu", dtype=torch.float32), diag


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
