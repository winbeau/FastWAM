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
``prune.reuse_step0_kv`` (V2-k0, keep_ratio 0): future tokens are computed on stage-1 step 0 only (latents keep
             that velocity) and the action stage reads those step-0 future K/V; there is no stage-2 re-encode.
Fewer steps is ``dense`` with a smaller ``num_inference_steps`` (both stages).
``frame0_cache=false`` (dense only) calls the native ``infer_action`` unchanged; so does
``action_infer_mode=first_frame`` (dense only), the release's Fast-WAM mode without future imagination.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate

from .config import CaseStudyConfig
from . import fovea as fv
from . import selection as sel


def _f16_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float16).tobytes()).decode()


class IDMCaseStudyRunner:
    def __init__(self, model, config: CaseStudyConfig, prompt_store: Optional[dict] = None):
        self.model = model
        self.cfg = config
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

    def _video_layers(self, x, freqs, t_mod, ctx, ctx_mask, kv0, keep_kv: bool, mask=None, recv=None):
        """Video stack over the given (future) tokens with frame-0 K/V prepended. ``mask``: additive key bias (fovea-w);
        ``recv``: [Kf] accumulator of the attention every future key receives from these rows (head mean, summed)."""
        kv = []
        f0 = kv0[0][0].shape[1]
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.video, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([kv0[layer][0], io[1]], dim=1)
            v = torch.cat([kv0[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads, ctx_mask=mask)
            if recv is not None:
                recv += self._probs(io[0], k)[:, f0:].sum(dim=0)
            if keep_kv:
                kv.append((k, v))
            x = self._post(block, io, out, ctx, ctx_mask)
        return x, kv

    def _action_layers(self, latents_action, ts_a, context, context_mask, cache, capture: set[int], noise_gen=None, mask=None):
        tokens, _t, t_mod, a_ctx, a_ctx_mask, freqs = self.action.prepare(
            action_tokens=latents_action, timestep=ts_a, context=context, context_mask=context_mask
        )
        x = tokens
        probs = {}
        for layer in range(self.num_layers):
            block = self.action.blocks[layer]
            io = self.mot._build_expert_attention_io(expert=self.action, block=block, x=x, freqs=freqs, t_mod=t_mod)
            k = torch.cat([cache[layer][0], io[1]], dim=1)
            v = torch.cat([cache[layer][1], io[2]], dim=1)
            out = flash_attention(q=io[0], k=k, v=v, num_heads=self.num_heads, ctx_mask=mask)
            if layer in capture:
                probs[layer] = self._probs(io[0], k)
            if noise_gen is not None:
                rms = out.float().pow(2).mean().sqrt()
                out = out + (self.cfg.noise.sigma * rms * torch.randn(
                    out.shape, generator=noise_gen, device=out.device, dtype=torch.float32)).to(out.dtype)
            x = self._post(block, io, out, a_ctx, a_ctx_mask)
        return self.action.post(x), probs

    @torch.no_grad()
    def infer_action(self, prompt, input_image, proprio, action_horizon, num_video_frames, seed,
                     sigma_shift=None, rand_device="cpu") -> tuple[torch.Tensor, dict[str, Any]]:
        cfg, model = self.cfg, self.model
        steps = int(cfg.num_inference_steps)
        if not cfg.frame0_cache or cfg.action_infer_mode == "first_frame":
            # Native path: the dense reference (frame0_cache=false), and the release's Fast-WAM mode
            # (first_frame: no imagined video, actions denoised against the observation alone).
            text = {"prompt": prompt}
            if self.prompt_store is not None:
                c, m, _ = self._encode_prompt(prompt)
                text = {"prompt": None, "context": c, "context_mask": m}
            out = model.infer_action(**text, input_image=input_image, action_horizon=action_horizon,
                                     num_video_frames=num_video_frames, proprio=proprio, num_inference_steps=steps,
                                     sigma_shift=sigma_shift, seed=seed, rand_device=rand_device,
                                     action_infer_mode=cfg.action_infer_mode)
            return out["action"], {"path": "native", "action_infer_mode": cfg.action_infer_mode, "seed": int(seed), "steps": steps}

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
            x = self._post(block, io, out, ctx0, ctxm0)

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
        v2k0 = prune is not None and prune.reuse_step0_kv
        kv_step0 = None
        # fovea-w revision 2 (M4 / M4w on IDM): the plan of stage 1, the step-0 attention every future token receives
        # (video rows and the probe's action rows, all layers), the step-0 Diff score, interlace bookkeeping.
        fovea_m = prune.fovea if prune is not None else None
        fplan = None
        recv_acc = torch.zeros(num_future, device=model.device) if fovea_m is not None else None
        diff_score = None
        fv_count = torch.zeros(num_future, dtype=torch.long, device=model.device) if fovea_m is not None else None
        fv_rows: list[int] = []
        pt, ph, pw = (int(x) for x in patch)
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
            idx_rel = (fplan.real(s) if fplan is not None else kept) if mode == "pruned" else all_future
            idx_abs = idx_rel + f0
            vmask = None
            if fplan is not None and mode == "pruned":
                bias = torch.zeros(int(idx_rel.numel()), device=model.device)
                for j, lg in fplan.bias_at(s).items():  # interlace_w: this step's slice log(N / n_s)
                    bias[torch.nonzero(idx_rel == j).flatten()] = lg
                if bool((bias != 0).any()):
                    vmask = torch.zeros((1, 1, 1, f0 + int(idx_rel.numel())), device=model.device)
                    vmask[..., f0:] = bias
                    vmask = vmask.to(latents_video.dtype)
                fv_rows.append(int(idx_rel.numel()))
                fv_count[idx_rel] += 1
            if idx_rel.numel():  # V2-k0: no future token after step 0, the latents keep step 0's velocity
                xv_in = xt[:, idx_abs]
                if mode == "skip":
                    xv = xv_in + tea_res
                    tea_cons += 1
                else:
                    need_kv = s == 0 and (probe or v2k0)
                    xv, kv_step = self._video_layers(xv_in, freqst[idx_abs], tmodt[:, idx_abs], ctxt, ctxmt[:, idx_abs], kv0,
                                                     keep_kv=need_kv, mask=vmask,
                                                     recv=recv_acc if (s == 0 and recv_acc is not None) else None)
                    if tea is not None:
                        tea_res, tea_acc, tea_cons = xv - xv_in, 0.0, 0
                    if need_kv and probe:
                        # Probe: one action pass at the first action timestep against step-0 video K/V.
                        ts_a0 = t_action[0].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
                        cap = set(range(self.num_layers)) if fovea_m is not None else act_layers
                        _pred, probe_probs = self._action_layers(latents_action, ts_a0, context, context_mask, kv_step, cap)
                        if recv_acc is not None:
                            for p in probe_probs.values():
                                recv_acc += p[:, f0:f0 + num_future].sum(dim=0)
                    if need_kv and v2k0:
                        kv_step0 = kv_step
                part = self.video.head(xv, tt[:, idx_abs])
                if vel is None:
                    vel = torch.zeros((1, f0 + num_future, part.shape[-1]), device=part.device, dtype=part.dtype)
                vel[:, idx_abs] = part
            if s == 0 and fovea_m is not None:
                # Diff (as on main): step 0's clean estimate x - sigma_0 v of every future token against the current frame.
                sigma_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                x0 = latents_video.float() - sigma_v * self.video.unpatchify(vel, (f, h, w)).float()
                z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
                z_cur = rearrange(latents_video[0, :, 0:pt].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)[0]
                diff_score = sel.diff_scores(z[1:].reshape(-1, z.shape[-1]), z_cur, grid)
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
                diag.update({"layer_act": self._layers["act"], "gate_score": sel.gate_score(p_act, grid),
                             "av_mass_future": float(av.sum()), "av_mass_obs": float(p_act[:, :f0].float().sum()),
                             "av_scores_f16": _f16_b64(av)})
            elif s == 0 and first_call:
                self._layers = {"act": -1, "inst": -1}
            if prune is not None and s == prune.warmup_steps - 1:
                k = 0 if v2k0 else max(1, int(round(prune.keep_ratio * num_future)))
                if v2k0:
                    kept = all_future[:0]
                elif prune.selection == "random":
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
                if fovea_m is not None:
                    # M4 / M4w on IDM (revision 2): the plan from step 0's A->V (probe), Diff and received attention.
                    recv = recv_acc / self.num_layers
                    fplan = fv.make_plan(fovea_m, k, grid, sel.av_scores(probe_probs[self._layers["act"]], grid).float().cpu(),
                                         diff_score.float().cpu(), recv.float().cpu(), off=prune.fovea_off).to(kept.device)
                    kept = fplan.fixed
                    diag["fovea"] = {
                        "method": fovea_m, "off": bool(prune.fovea_off), "k": int(k), "sinks": fplan.sinks.tolist(),
                        "sink_share_step0": float(recv[fplan.sinks].sum() / recv.sum()), "core_index": fplan.core.tolist(),
                        "periphery": int(fplan.periphery.numel()), "slices": [int(x.numel()) for x in fplan.slices],
                        "truncated": 0 if fplan.truncated is None else int(fplan.truncated.numel()),
                        "slice_bias": [round(x, 6) for x in fplan.slice_bias]}
                diag["kept_count"] = int(kept.numel())
                diag["kept_index"] = kept.tolist()
                if probe_probs:
                    av = sel.av_scores(probe_probs[self._layers["act"]], grid)
                    if float(av.sum()) > 0:
                        diag["kept_av_mass_fraction"] = float((av[kept].sum() / av.sum()).item())

        # ---- stage 2: clean video cache (frame 0 + kept future tokens), action denoising
        keep2 = kept if (prune is not None and not gated_dense) else all_future
        if v2k0:
            # V2-k0: the action stage reads stage-1 step 0's future K/V (frame 0 prepended, as the re-encode would be).
            cache = kv_step0
            diag["reuse_step0_kv"] = True
        else:
            idx2_rel = keep2
            if fplan is not None and fplan.slices:  # M4 / M4w: every token stage 1 computed after step 0
                idx2_rel = torch.sort(torch.cat([fplan.fixed] + [x.to(fplan.fixed.device) for x in fplan.slices])).values
            keep2 = idx2_rel
            idx2 = idx2_rel + f0
            xc, _tc, tmodc, ctxc, ctxmc, freqsc, *_ = self.video.prepare(
                x=latents_video, timestep=zero_t, context=context, context_mask=context_mask, action=None,
                fuse_vae_embedding_in_latents=fuse)
            _x, cache = self._video_layers(xc[:, idx2], freqsc[idx2], tmodc[:, idx2], ctxc, ctxmc[:, idx2], kv0, keep_kv=True)
        noise_gen = None
        if cfg.method == "noise" and cfg.noise.sigma > 0:
            noise_gen = torch.Generator(device=model.device).manual_seed((int(seed) ^ 0x5EED) & 0x7FFFFFFF)
        for s in range(steps):
            ts_a = t_action[s].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
            cache_s, amask = cache, None
            if fplan is not None and fplan.slices:
                # M4 / M4w: action step s sees frame 0 + sinks + core + slice (s - 1) mod 9 of the clean re-encode.
                vis = fplan.real(s)
                cols = torch.cat([torch.arange(f0, device=vis.device), f0 + torch.searchsorted(idx2_rel, vis)])
                cache_s = [(kk[:, cols], vv[:, cols]) for kk, vv in cache]
                bias_s = fplan.bias_at(s)
                if bias_s:
                    n_act = int(latents_action.shape[1])
                    amask = torch.zeros((1, 1, 1, f0 + int(vis.numel()) + n_act), device=model.device)
                    for j, lg in bias_s.items():
                        amask[..., f0 + torch.nonzero(vis == j).flatten()] = lg
                    amask = amask.to(latents_action.dtype)
            pred, _ = self._action_layers(latents_action, ts_a, context, context_mask, cache_s, set(), noise_gen, mask=amask)
            latents_action = model.infer_action_scheduler.step(pred, d_action[s], latents_action)

        diag["modes"] = modes
        diag["computed_steps"] = sum(1 for m in modes if m != "skip")
        if prune is not None:
            pruned = sum(1 for m in modes if m == "pruned")
            rows = sum(fv_rows) if fplan is not None else pruned * int(kept.numel())
            diag["future_token_steps"] = (steps - pruned) * num_future + rows + int(keep2.numel())
        if fplan is not None:
            diag["fovea"]["rows_per_step"] = fv_rows
            diag["fovea"]["stage2_reencoded"] = int(keep2.numel())
            if fplan.slices:
                cut = set(fplan.truncated.tolist())
                cnt = fv_count[fplan.periphery].tolist()
                once = all((c == 0) if int(j) in cut else (c == 1) for j, c in zip(fplan.periphery.tolist(), cnt))
                diag["fovea"]["interlace_once"] = once
                if not once and len(fv_rows) == len(fplan.slices):
                    raise RuntimeError("fovea interlace: a periphery token was not computed exactly once")
            diag["future_token_steps_dense"] = steps * num_future + num_future
        if tea is not None:
            diag["teacache_rel_l1"] = [round(r, 6) for r in tea_rel]
        if noise_gen is not None:
            diag["noise_sigma"] = cfg.noise.sigma
        return latents_action[0].detach().to(device="cpu", dtype=torch.float32), diag


__all__ = ["IDMCaseStudyRunner"]
