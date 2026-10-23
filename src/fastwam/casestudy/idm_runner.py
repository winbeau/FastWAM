"""FastWAM-IDM (Optional IDM, ``action_infer_mode=idm``) inference for the case study.

Reproduces ``FastWAMIDM.infer_action``: stage 1 denoises the video alone, stage 2 freezes
the denoised video (timestep 0) as a K/V cache and denoises the action against it. Two
exact restructurings are shared by every arm:

1. Frame-0 reuse. Frame-0 latents are pinned to the observation with timestep 0 and
   (``first_frame_causal``) attend only to themselves, in stage 1 and in the stage-2
   prefill alike, so their per-layer K/V is computed once per call and reused.
2. Prompt reuse (or a precomputed prompt store).

Arms (``config.py``):
``prune``    future video tokens leave stage 1 after ``warmup_steps`` (latents keep their
             last velocity) and only the kept tokens enter the stage-2 cache, so action
             queries attend to frame 0 plus the kept tokens. Scores come from a probe:
             one action pass (initial action noise, first action timestep) against the
             stage-1 step-0 video K/V -- CF-Prune's "score on the first, noisiest pass".
``teacache`` residual reuse over the video stack on stage-1 steps.
``noise``    Gaussian noise on the stage-2 action attention output.
Fewer steps is ``dense`` with a smaller ``num_inference_steps`` (both stages).
``frame0_cache=false`` (dense only) calls the native ``infer_action`` unchanged.

Execution options (``execution=``; off by default, each verified bit for bit on paired
episodes before use): ``cross_kv_cache`` (cross-attention K/V of the constant context once per
call per layer and expert), ``defer_diagnostics`` (diagnostics converted after the final action
copy), ``cuda_graphs`` (needs ``cross_kv_cache``; CUDA-graph replay of stage-1 video stacks
without the probe and TeaCache and of stage-2 action stacks without noise, from a runner's
second call on).
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate

from .config import CaseStudyConfig
from . import selection as sel


def _f16_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float16).tobytes()).decode()


class IDMCaseStudyRunner:
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
        self._graphs: dict[tuple, "_GraphCall"] = {}
        self._calls_done = 0
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

    def reset_episode(self) -> None:
        self._prompt_cache.clear()
        self._layers = None

    def _encode_prompt(self, prompt: str):
        if self.cfg.prompt_cache and prompt in self._prompt_cache:
            return self._prompt_cache[prompt]
        if self.prompt_store is not None:
            if prompt not in self.prompt_store:
                raise KeyError(f"prompt not in precomputed store: {prompt!r}")
            item = self.prompt_store[prompt]
            entry = (item["context"].to(self.model.device), item["mask"].to(self.model.device, dtype=torch.bool), int(item["num_text"]))
        else:
            context, context_mask = self.model.encode_prompt(prompt)
            _ids, raw_mask = self.model.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            entry = (context, context_mask, int(raw_mask.gt(0).sum().item()))
        if self.cfg.prompt_cache:
            self._prompt_cache[prompt] = entry
        return entry

    def _probs(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qh = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        kh = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        logits = (qh.float() @ kh.float().transpose(-1, -2)) * (qh.shape[-1] ** -0.5)
        return torch.softmax(logits, dim=-1).mean(dim=1)[0]

    def _post(self, block, io, mixed, context, context_mask):
        return self.mot._apply_expert_post_block_tensor(
            block=block, residual_x=io[3], mixed_attn_out=mixed, gate_msa=io[4],
            shift_mlp=io[5], scale_mlp=io[6], gate_mlp=io[7], context=context, context_mask=context_mask,
        )

    def _cross_kv(self, expert, context):
        out = []
        for layer in range(self.num_layers):
            ca = expert.blocks[layer].cross_attn
            out.append((ca.norm_k(ca.k(context)), ca.v(context)))
        return out

    def _post_c(self, block, io, mixed, ctx_mask, ckv):
        """``_post`` with cached cross-attention K/V (same ops and order as
        ``MoT._apply_expert_post_block_tensor``)."""
        x = block.gate(io[3], io[4], block.self_attn.o(mixed))
        ca = block.cross_attn
        q = ca.norm_q(ca.q(block.norm3(x)))
        x = x + ca.o(flash_attention(q=q, k=ckv[0], v=ckv[1], num_heads=ca.num_heads, ctx_mask=ctx_mask.unsqueeze(1)))
        mlp_input = modulate(block.norm2(x), io[5], io[6])
        return block.gate(x, io[7], block.ffn(mlp_input))

    def _video_stack(self, x, freqs, t_mod, ctx_mask, kv0, ckv):
        """``_video_layers`` without K/V capture, with cached cross K/V (graph body)."""
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.video, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([kv0[layer][0], io[1]], dim=1)
            v = torch.cat([kv0[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads)
            x = self._post_c(block, io, out, ctx_mask, ckv[layer])
        return x

    def _action_stack(self, x, freqs, t_mod, a_ctx_mask, cache, ckv):
        """``_action_layers`` without capture or noise, with cached cross K/V (graph body)."""
        for layer in range(self.num_layers):
            block = self.action.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.action, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([cache[layer][0], io[1]], dim=1)
            v = torch.cat([cache[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads)
            x = self._post_c(block, io, out, a_ctx_mask, ckv[layer])
        return x

    def _graph(self, kind, fn, step_args, consts, token):
        key = (kind,) + tuple(tuple(t.shape) for t in step_args) + tuple(tuple(c[0][0].shape) for c in consts)
        g = self._graphs.get(key)
        if g is None:
            g = self._graphs[key] = _GraphCall(fn, step_args, consts)
        g.set_consts(token, consts)
        return g(step_args)

    def _video_layers(self, x, freqs, t_mod, ctx, ctx_mask, kv0, keep_kv: bool, ckv=None):
        """Video stack over the given (future) tokens with frame-0 K/V prepended."""
        kv = []
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.video, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([kv0[layer][0], io[1]], dim=1)
            v = torch.cat([kv0[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads)
            if keep_kv:
                kv.append((k, v))
            x = self._post_c(block, io, out, ctx_mask, ckv[layer]) if ckv is not None else self._post(block, io, out, ctx, ctx_mask)
        return x, kv

    def _action_layers(self, latents_action, ts_a, context, context_mask, cache, capture: set[int], noise_gen=None,
                       ckv=None, token=None):
        tokens, _t, t_mod, a_ctx, a_ctx_mask, freqs = self.action.prepare(
            action_tokens=latents_action, timestep=ts_a, context=context, context_mask=context_mask
        )
        x = tokens
        probs = {}
        if (self.exec["cuda_graphs"] and ckv is not None and self._calls_done >= 1 and not capture
                and noise_gen is None and token is not None):
            x = self._graph("action", self._action_stack, (tokens, freqs, t_mod, a_ctx_mask), [cache, ckv], token)
            return self.action.post(x), probs
        for layer in range(self.num_layers):
            block = self.action.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.action, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([cache[layer][0], io[1]], dim=1)
            v = torch.cat([cache[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads)
            if layer in capture:
                probs[layer] = self._probs(io[0], k)
            if noise_gen is not None:
                rms = out.float().pow(2).mean().sqrt()
                out = out + (self.cfg.noise.sigma * rms * torch.randn(
                    out.shape, generator=noise_gen, device=out.device, dtype=torch.float32)).to(out.dtype)
            x = self._post_c(block, io, out, a_ctx_mask, ckv[layer]) if ckv is not None else self._post(block, io, out, a_ctx, a_ctx_mask)
        return self.action.post(x), probs

    @torch.no_grad()
    def infer_action(self, prompt, input_image, proprio, action_horizon, num_video_frames, seed,
                     sigma_shift=None, rand_device="cpu") -> tuple[torch.Tensor, dict[str, Any]]:
        cfg, model = self.cfg, self.model
        steps = int(cfg.num_inference_steps)
        if not cfg.frame0_cache:
            text = {"prompt": prompt}
            if self.prompt_store is not None:
                c, m, _ = self._encode_prompt(prompt)
                text = {"prompt": None, "context": c, "context_mask": m}
            out = model.infer_action(**text, input_image=input_image, action_horizon=action_horizon,
                                     num_video_frames=num_video_frames, proprio=proprio, num_inference_steps=steps,
                                     sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, action_infer_mode="idm")
            return out["action"], {"path": "native", "seed": int(seed), "steps": steps}

        model.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        _, _, height, width = input_image.shape
        if proprio is not None:
            proprio = proprio.reshape(1, -1).to(device=model.device, dtype=model.torch_dtype)
        vae = model.vae
        latent_t = (num_video_frames - 1) // vae.temporal_downsample_factor + 1
        latent_h, latent_w = height // vae.upsampling_factor, width // vae.upsampling_factor
        patch = self.video.patch_size
        grid = sel.Grid(frames=latent_t // int(patch[0]), height=latent_h // int(patch[1]), width=latent_w // int(patch[2]))
        f0, num_future = grid.per_frame, grid.num_future

        gen_v = torch.Generator(device=rand_device).manual_seed(int(seed))
        gen_a = torch.Generator(device=rand_device).manual_seed(int(seed))
        latents_video = torch.randn((1, vae.model.z_dim, latent_t, latent_h, latent_w), generator=gen_v,
                                    device=rand_device, dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
        latents_action = torch.randn((1, action_horizon, self.action.action_dim), generator=gen_a,
                                     device=rand_device, dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
        input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
        first = model._encode_input_image_latents_tensor(input_image=input_image, tiled=False)
        latents_video[:, :, 0:1] = first.clone()
        fuse = bool(getattr(self.video, "fuse_vae_embedding_in_latents", False))
        context, context_mask, num_text = self._encode_prompt(prompt)
        if proprio is not None:
            context, context_mask = model._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)
        t_video, d_video = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_video.dtype, shift_override=sigma_shift)
        t_action, d_action = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_action.dtype, shift_override=sigma_shift)
        zero_t = torch.zeros((1,), dtype=latents_video.dtype, device=model.device)

        prune = cfg.prune
        is_cf = prune is not None and prune.selection == "cf"
        first_call = self._layers is None
        want_scores = prune is not None and prune.selection in ("av", "cf")
        probe = cfg.record_attention or want_scores
        act_layers = set(range(self.num_layers)) if (first_call and probe) else ({self._layers["act"]} if probe else set())
        inst_layers = (set(range(self.num_layers)) if first_call else {self._layers["inst"]}) if is_cf else set()

        # ---- frame-0 prefill (shared by stage 1 and the stage-2 cache)
        x0, _t0, tmod0, ctx0, ctxm0, freqs0, *_ = self.video.prepare(
            x=first, timestep=zero_t, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse)
        ckv_v = self._cross_kv(self.video, ctx0) if self.exec["cross_kv_cache"] else None
        ckv_a = None
        defer = self.exec["defer_diagnostics"]
        pending: dict[str, Any] = {}
        token = object()
        kv0, inst_mass, x = [], {}, x0
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.video, block=block, x=x, freqs=freqs0, t_mod=tmod0)
            out = flash_attention(q=io[0], k=io[1], v=io[2], num_heads=self.num_heads)
            if layer in inst_layers:
                x_mid = block.gate(io[3], io[4], block.self_attn.o(out))
                cq = block.cross_attn.norm_q(block.cross_attn.q(block.norm3(x_mid)))
                ck = block.cross_attn.norm_k(block.cross_attn.k(ctx0))
                inst_mass[layer] = self._probs(cq, ck)[:, :num_text].sum(dim=-1)
            kv0.append((io[1], io[2]))
            x = self._post_c(block, io, out, ctxm0, ckv_v[layer]) if ckv_v is not None else self._post(block, io, out, ctx0, ctxm0)

        # ---- stage 1: video denoising
        all_future = torch.arange(num_future, device=model.device)
        kept, gated_dense = all_future, False
        vel = None
        tea = cfg.teacache
        tea_prev, tea_acc, tea_cons, tea_res = None, 0.0, 0, None
        tea_poly = np.poly1d(tea.coefficients) if tea is not None else None
        modes, tea_rel = [], []
        diag: dict[str, Any] = {"path": "casestudy-idm", "seed": int(seed), "steps": steps, "method": cfg.method}
        probe_probs: dict[int, torch.Tensor] = {}
        for s in range(steps):
            ts_v = t_video[s].unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
            xt, tt, tmodt, ctxt, ctxmt, freqst, f, h, w, _tpf = self.video.prepare(
                x=latents_video, timestep=ts_v, context=context, context_mask=context_mask, action=None,
                fuse_vae_embedding_in_latents=fuse)
            mode = "full"
            if prune is not None and s >= prune.warmup_steps and not gated_dense and s not in prune.full_steps:
                mode = "pruned"
            if tea is not None:
                b0 = self.video.blocks[0]
                sh, sc, *_ = self.mot._split_modulation(b0, tmodt[:, f0:])
                signal = modulate(b0.norm1(xt[:, f0:]), sh, sc)
                if tea_prev is not None:
                    rel = float(((signal - tea_prev).abs().mean() / tea_prev.abs().mean().clamp_min(1e-12)).item())
                    tea_rel.append(rel)
                    if 0 < s < steps - 1:
                        tea_acc += float(tea_poly(rel))
                        if tea_acc < tea.thresh and tea_cons < tea.max_consecutive_skips and tea_res is not None:
                            mode = "skip"
                tea_prev = signal
            idx_rel = kept if mode == "pruned" else all_future
            idx_abs = idx_rel + f0
            xv_in = xt[:, idx_abs]
            if mode == "skip":
                xv = xv_in + tea_res
                tea_cons += 1
            else:
                need_kv = s == 0 and probe
                if (self.exec["cuda_graphs"] and ckv_v is not None and self._calls_done >= 1 and not need_kv and tea is None):
                    xv = self._graph("video", self._video_stack, (xv_in, freqst[idx_abs], tmodt[:, idx_abs], ctxmt[:, idx_abs]),
                                     [kv0, ckv_v], token)
                    kv_step = []
                else:
                    xv, kv_step = self._video_layers(xv_in, freqst[idx_abs], tmodt[:, idx_abs], ctxt, ctxmt[:, idx_abs], kv0,
                                                     keep_kv=need_kv, ckv=ckv_v)
                if tea is not None:
                    tea_res, tea_acc, tea_cons = xv - xv_in, 0.0, 0
                if need_kv:
                    # Probe: one action pass at the first action timestep against step-0 video K/V.
                    ts_a0 = t_action[0].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
                    if ckv_v is not None and ckv_a is None:
                        ckv_a = self._cross_kv(self.action, self.action.prepare(
                            action_tokens=latents_action, timestep=ts_a0, context=context, context_mask=context_mask)[3])
                    _pred, probe_probs = self._action_layers(latents_action, ts_a0, context, context_mask, kv_step, act_layers,
                                                             ckv=ckv_a)
            part = self.video.head(xv, tt[:, idx_abs])
            if vel is None:
                vel = torch.zeros((1, f0 + num_future, part.shape[-1]), device=part.device, dtype=part.dtype)
            vel[:, idx_abs] = part
            latents_video = model.infer_video_scheduler.step(self.video.unpatchify(vel, (f, h, w)), d_video[s], latents_video)
            latents_video[:, :, 0:1] = first.clone()
            modes.append(mode)
            if s == 0 and probe_probs:
                if first_call:
                    act = min(probe_probs, key=lambda l: sel.entropy(sel.future_mass(probe_probs[l], grid)))
                    inst = min(inst_mass, key=lambda l: sel.entropy(inst_mass[l])) if inst_mass else act
                    if prune is not None and prune.layer_rule == "fixed":
                        act = inst = int(prune.layer)
                    self._layers = {"act": int(act), "inst": int(inst)}
                p_act = probe_probs[self._layers["act"]]
                av = sel.av_scores(p_act, grid)
                if defer:
                    diag["layer_act"] = self._layers["act"]
                    pending.update(gate=_gate_entropies(p_act, grid), av=av, av_mass_obs=p_act[:, :f0].float().sum())
                else:
                    diag.update({"layer_act": self._layers["act"], "gate_score": sel.gate_score(p_act, grid),
                                 "av_mass_future": float(av.sum()), "av_mass_obs": float(p_act[:, :f0].float().sum()),
                                 "av_scores_f16": _f16_b64(av)})
            elif s == 0 and first_call:
                self._layers = {"act": -1, "inst": -1}
            if prune is not None and s == prune.warmup_steps - 1:
                k = max(1, int(round(prune.keep_ratio * num_future)))
                if prune.selection == "random":
                    g = torch.Generator(device="cpu").manual_seed((int(seed) + 1) & 0x7FFFFFFF)
                    kept = sel.random_indices(num_future, k, g, model.device)
                else:
                    p_act = probe_probs[self._layers["act"]]
                    if prune.selection == "av":
                        kept = sel.topk_indices(sel.av_scores(p_act, grid), k)
                    else:
                        kept = sel.complementary_topk(sel.cf_action_scores(p_act, grid),
                                                      sel.cf_instruction_scores(inst_mass[self._layers["inst"]], grid), k)
                        diag["layer_inst"] = self._layers["inst"]
                    if prune.gate is not None:
                        gated_dense = bool(sel.gate_score(p_act, grid)["c"] > prune.gate.tau)
                        diag["gated_dense"] = gated_dense
                if prune.extra_keep_ratio > 0:
                    mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    mask[kept] = False
                    dropped = torch.nonzero(mask).flatten()
                    m = int(round(prune.extra_keep_ratio * dropped.numel()))
                    if m > 0:
                        if prune.extra_selection == "random":
                            g = torch.Generator(device="cpu").manual_seed((int(seed) + 2) & 0x7FFFFFFF)
                            extra = dropped[torch.randperm(dropped.numel(), generator=g)[:m].to(dropped.device)]
                        else:
                            extra = dropped[torch.topk(sel.av_scores(probe_probs[self._layers["act"]], grid)[dropped], m).indices]
                        kept = torch.cat([kept, extra])
                kept = torch.sort(kept).values
                diag["kept_count"] = int(kept.numel())
                if defer:
                    pending["kept"] = kept
                    pending["kept_av"] = sel.av_scores(probe_probs[self._layers["act"]], grid) if probe_probs else None
                else:
                    diag["kept_index"] = kept.tolist()
                    if probe_probs:
                        av = sel.av_scores(probe_probs[self._layers["act"]], grid)
                        if float(av.sum()) > 0:
                            diag["kept_av_mass_fraction"] = float((av[kept].sum() / av.sum()).item())

        # ---- stage 2: clean video cache (frame 0 + kept future tokens), action denoising
        keep2 = kept if (prune is not None and not gated_dense) else all_future
        idx2 = keep2 + f0
        xc, _tc, tmodc, ctxc, ctxmc, freqsc, *_ = self.video.prepare(
            x=latents_video, timestep=zero_t, context=context, context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=fuse)
        _x, cache = self._video_layers(xc[:, idx2], freqsc[idx2], tmodc[:, idx2], ctxc, ctxmc[:, idx2], kv0, keep_kv=True,
                                       ckv=ckv_v)
        if ckv_v is not None and ckv_a is None:
            ckv_a = self._cross_kv(self.action, self.action.prepare(
                action_tokens=latents_action, timestep=t_action[0].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device),
                context=context, context_mask=context_mask)[3])
        noise_gen = None
        if cfg.method == "noise" and cfg.noise.sigma > 0:
            noise_gen = torch.Generator(device=model.device).manual_seed((int(seed) ^ 0x5EED) & 0x7FFFFFFF)
        for s in range(steps):
            ts_a = t_action[s].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
            pred, _ = self._action_layers(latents_action, ts_a, context, context_mask, cache, set(), noise_gen, ckv=ckv_a, token=token)
            latents_action = model.infer_action_scheduler.step(pred, d_action[s], latents_action)

        diag["modes"] = modes
        diag["computed_steps"] = sum(1 for m in modes if m != "skip")
        if prune is not None:
            pruned = sum(1 for m in modes if m == "pruned")
            diag["future_token_steps"] = (steps - pruned) * num_future + pruned * int(kept.numel()) + int(keep2.numel())
            diag["future_token_steps_dense"] = steps * num_future + num_future
        if tea is not None:
            diag["teacache_rel_l1"] = [round(r, 6) for r in tea_rel]
        if noise_gen is not None:
            diag["noise_sigma"] = cfg.noise.sigma
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if defer:  # converted after the synchronising copy, in the default path's key order
            ordered = {}
            for k_, v_ in diag.items():
                ordered[k_] = v_
                if k_ == "layer_act" and "gate" in pending:
                    h_third, h_wrist, h_early, h_late = pending["gate"].tolist()
                    ordered["gate_score"] = {"h_third": h_third, "h_wrist": h_wrist, "h_early": h_early, "h_late": h_late,
                                             "c_view": h_third - h_wrist, "c_temp": h_early - h_late,
                                             "c": (h_third - h_wrist) + (h_early - h_late)}
                    ordered["av_mass_future"] = float(pending["av"].sum())
                    ordered["av_mass_obs"] = float(pending["av_mass_obs"])
                    ordered["av_scores_f16"] = _f16_b64(pending["av"])
                if k_ == "kept_count" and "kept" in pending:
                    ordered["kept_index"] = pending["kept"].tolist()
                    av_ = pending["kept_av"]
                    if av_ is not None and float(av_.sum()) > 0:
                        ordered["kept_av_mass_fraction"] = float((av_[pending["kept"]].sum() / av_.sum()).item())
            diag = ordered
        self._calls_done += 1
        return action_out, diag


def _entropy_t(dist: torch.Tensor) -> torch.Tensor:
    """``selection.entropy`` without the host sync (same values)."""
    x = dist.float().clamp_min(0)
    total = x.sum()
    p = torch.where(total > 0, x / total, torch.full_like(x, 1.0 / x.numel()))
    return -(p * torch.log(p.clamp_min(1e-12))).sum()


def _gate_entropies(p_act: torch.Tensor, grid) -> torch.Tensor:
    third, wrist = grid.future_view_masks(p_act.device)
    start = grid.per_frame
    fut = p_act[:, start : start + grid.num_future].float()
    mass = fut.sum(dim=0)
    half = p_act.shape[0] // 2
    return torch.stack([_entropy_t(mass[third]), _entropy_t(mass[wrist]), _entropy_t(fut[:half].sum(dim=0)),
                        _entropy_t(fut[half:].sum(dim=0))])


def _static(src: torch.Tensor):
    """Static buffer for ``src`` and a view with its exact strides (broadcast dims stored once)."""
    idx = tuple(slice(0, 1) if (st == 0 and sz > 1) else slice(None) for st, sz in zip(src.stride(), src.shape))
    base = src[idx].clone()
    shape = tuple(src.shape)
    return base, idx, ((lambda: base.expand(shape)) if tuple(base.shape) != shape else (lambda: base))


class _GraphCall:
    """CUDA graph of ``fn(*step_args, *consts)`` for fixed shapes; step inputs are copied in
    before each replay, per-call constants (lists of (K, V) pairs) once per call."""

    def __init__(self, fn, step_args, consts):
        self.inp = [_static(t) for t in step_args]
        self.consts = [[(k.clone(), v.clone()) for k, v in c] for c in consts]
        self.token = None

        def run():
            return fn(*[view() for _b, _i, view in self.inp], *self.consts)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = run()

    def set_consts(self, token, consts) -> None:
        if token is self.token:
            return
        for dst, src in zip(self.consts, consts):
            for (dk, dv), (sk, sv) in zip(dst, src):
                dk.copy_(sk)
                dv.copy_(sv)
        self.token = token

    def __call__(self, step_args):
        for (base, idx, _view), src in zip(self.inp, step_args):
            base.copy_(src[idx])
        self.graph.replay()
        return self.out.clone()


__all__ = ["IDMCaseStudyRunner"]
