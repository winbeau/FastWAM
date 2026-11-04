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
action stream. ``prune.steer`` runs the action rows of pruned steps in plain fp32 attention
with a per-row bias on the kept future columns (see ``_steer_bias``); video rows keep SDPA
unless ``prune.video_eager`` (then also fp32, with the optional ``steer_video`` bias and the
``video_full_kv`` keys). ``oracle_kv`` runs a dense pass of the same call first and gives the
action rows of pruned steps its kept-token K/V; ``dense_bias`` never prunes and biases the
action rows' future columns. ``record_video_frame0`` / ``record_motion_steps`` log per step the
video rows' frame-0 share and the frame-1 motion of the clean estimate; ``diag["cost"]`` holds
the token and key counts for a relative-compute estimate. ``prune.recon`` (r0 / r1 / r2) never
computes or caches the dropped positions: each pruned step infers their K/V from the frame-0 token
at the same (h, w) and appends them to the keys of both row groups; ``keep_ratio: 0`` drops the
video rows of pruned steps altogether (V2-k0, R0-k0). ``agc`` / ``record_pilot`` log the pilot, the
warm-up step's clean action estimate ``x - sigma_0 * v_0`` (``diag["pilot_f32"]``, normalized); the gain
itself is applied by the adapter in simulator units (``agc.py``). ``prune.action_no_future`` (S0-noF) drops
every future key of the action rows on pruned steps; ``prune.extrapolate_w`` (EX) adds a second action-expert
stream with those keys on pruned steps and steps with ``v_s + w * (v_s - v_noF)``.
N1-N4 (``prune.fovea``, ``fovea.py``): ``interlace_cal`` biases the slice keys per layer / head / row by the step-0 log-ratio
of periphery to slice mass (``_cal_bias``); ``interlace_fill`` gives every uncomputed position the K / V of the nearest token
computed on the same step; ``merge`` keeps a residual stream per periphery member, merges the groups per layer and adds each
group's increment back to its members; ``interlace_ht`` follows the plan's per-token frequencies.
step0-sparse (``prune.step0``): step 0 computes the 39 probes only (every probe key + log(Kf / 39), video and action rows),
the other future tokens move along the interpolated draft (``_probe_draft``) until computed; plan b scores with the
interpolated A->V and Diff, plan d with the frame-0 attention and the pilot's end-effector tube (``_step0_tube`` from the
adapter); the leftover probes go to the last slices. ``prune.e9`` (E9) redoes step 0's update of M4w with the interpolated
draft on slices 1-6 or 8-9.
``frame0_cache=false`` (dense only) calls the model's native
``infer_action`` unchanged, as the reference for equivalence checks. ``cfg.dump`` writes
one call's attention, action hidden states and decoded video to an ``.npz``.
"""

from __future__ import annotations

import base64
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate, rope_apply

from .config import CaseStudyConfig
from . import fovea as fv
from . import selection as sel


def _f16_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float16).tobytes()).decode()


def _f32_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float32).tobytes()).decode()


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
        self._cost: Optional[dict] = None
        #: step0-sparse-r2 (prune.mem): per memory key (probe seed offset / the executed arm) the last call's imagination.
        self._memory: dict[Any, dict] = {}

    # ------------------------------------------------------------------ episode
    def reset_episode(self) -> None:
        """Per-episode state: prompt cache and the CF layer choices (frozen per episode)."""
        self._prompt_cache.clear()
        self._layers = None
        self._memory.clear()

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

    def _masked_probs(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``_attention_probs`` with the additive key mask [1, 1, 1, Sk] of the forward pass in the logits (none: the same)."""
        if mask is None:
            return self._attention_probs(q, k)
        return torch.softmax(self._head_logits(q, k) + mask[0].float(), dim=-1).mean(dim=0)

    def _probe_draft(self, latents: torch.Tensor, vel_tokens: torch.Tensor, grid_size, grid: sel.Grid, probes: torch.Tensor,
                     sigma: torch.Tensor, patch: tuple[int, int, int], gain: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """step0-sparse: the clean estimate x - sigma v of the ``probes`` interpolated over every future block
        (``fv.interp_tokens``) -> (draft velocity (x - x0) / sigma in the token layout of ``vel_tokens`` [1, f0 + Kf, D]
        (fp32), interpolated clean estimate of the future tokens [Kf, D'] in the Diff layout (fp32)). step0-explore: ``gain``
        scales the probes' change from the current frame at the same (h, w) before the interpolation (1: untouched)."""
        pt, ph, pw = patch
        x0 = latents.float() - sigma * self.video.unpatchify(vel_tokens, grid_size).float()
        z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
        zfut = z[1:].reshape(-1, z.shape[-1])
        if gain != 1.0:
            zc = z[0][grid.future_obs_index(z.device)][probes]
            zfut = zfut.clone()
            zfut[probes] = zc + gain * (zfut[probes] - zc)
        zf = fv.interp_tokens(zfut, grid, probes)
        z_int = torch.cat([z[:1], zf.reshape(z.shape[0] - 1, z.shape[1], -1)], dim=0)
        x0_int = rearrange(z_int, "f (h w) (c a b e) -> c (f a) (h b) (w e)", h=grid.height, a=pt, b=ph, e=pw)[None]
        v = (latents.float() - x0_int) / sigma
        return rearrange(v, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw), zf

    def _spread(self, values: torch.Tensor, rows: torch.Tensor, grid: sel.Grid, mode: str, fallback: torch.Tensor,
                lam: float = 0.0) -> torch.Tensor:
        """step0-explore: values [Kf, D] known on ``rows`` spread to every future token (float32): ``idw`` block-local
        inverse-distance mean (a block without a row takes ``fallback``); ``near`` the same inside the near frame, the far frame
        = near + lam (near - fallback) at the same (h, w) unless computed itself."""
        rows = rows.to(values.device)
        kf, pf = grid.num_future, grid.per_frame
        src = rows if mode == "idw" else rows[rows < pf]
        filled, upd = fv.idw_fill(values, src, grid)
        out = fallback.float().clone()
        out[src] = values[src].float()
        out[upd] = filled[upd]
        if mode == "near":
            near = out[:pf]
            far = near + lam * (near - fallback[:pf].float())
            own = rows[rows >= pf]
            out[pf:] = far
            out[own] = values[own].float()
        return out

    def _set_draft(self, latents: torch.Tensor, vel_tokens: torch.Tensor, grid_size, grid: sel.Grid, rows: torch.Tensor,
                   sigma: torch.Tensor, patch: tuple[int, int, int], gain: float, mode: str, lam: float):
        """step0-explore drafts from any computed set (``_spread``): the rows' clean estimate (their change from the current frame
        x ``gain``) spread over every future token -> (draft velocity, token layout [1, f0 + Kf, D] fp32; spread estimate [Kf, D'])."""
        pt, ph, pw = patch
        x0 = latents.float() - sigma * self.video.unpatchify(vel_tokens, grid_size).float()
        z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
        zfut = z[1:].reshape(-1, z.shape[-1]).clone()
        zc = z[0][grid.future_obs_index(z.device)]
        rows = rows.to(z.device)
        zfut[rows] = zc[rows] + gain * (zfut[rows] - zc[rows])
        zf = self._spread(zfut, rows, grid, mode, zc, lam)
        return self._map_draft(latents, zf, grid, sigma, patch), zf

    def _map_draft(self, latents: torch.Tensor, zf: torch.Tensor, grid: sel.Grid, sigma: torch.Tensor,
                   patch: tuple[int, int, int]) -> torch.Tensor:
        """step0-sparse-r2: the velocity (x - x0) / sigma [1, f0 + Kf, D] (fp32, the token layout of the video head) that moves
        every future token towards the clean estimate ``zf`` [Kf, D'] (the Diff layout)."""
        pt, ph, pw = patch
        z = rearrange(latents[0].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
        z_int = torch.cat([z[:1], zf.float().reshape(z.shape[0] - 1, z.shape[1], -1)], dim=0)
        x0 = rearrange(z_int, "f (h w) (c a b e) -> c (f a) (h b) (w e)", h=grid.height, a=pt, b=ph, e=pw)[None]
        v = (latents.float() - x0) / sigma
        return rearrange(v, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw)

    def _clean_future(self, latents: torch.Tensor, vel_tokens: torch.Tensor, grid_size, sigma: torch.Tensor,
                      patch: tuple[int, int, int]) -> torch.Tensor:
        """x - sigma v of every future token [Kf, D'] (fp32, the Diff layout)."""
        pt, ph, pw = patch
        x0 = latents.float() - sigma * self.video.unpatchify(vel_tokens, grid_size).float()
        z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
        return z[1:].reshape(-1, z.shape[-1])

    def _side_forward(self, pre_v: dict, pre_a: dict, kv0: list, probes: torch.Tensor, bias: float, m: int, f0: int,
                      num_future: int, score_layer: int) -> dict:
        """step0-sparse-r2 side forward beside the main pass: the future rows of every token on blocks < m, of ``probes`` from
        block m on (their keys + ``bias`` from there, video and action rows), and the action rows; the same kernels as the
        main pass. -> {rows, vel (video head of the rows), pred_action, probs (score layer, [A, f0 + Kf + A], 0 on absent
        keys), recv [Kf] (layer mean, absent 0), layers ([rows, video keys, action rows, action keys] per block)}."""
        dev = pre_v["tokens"].device
        allf = torch.arange(num_future, device=dev)
        probes = probes.to(dev)
        n_act = int(pre_a["tokens"].shape[1])
        rows = allf if m > 0 else probes
        xv = pre_v["tokens"][:, rows + f0]
        xa = pre_a["tokens"]
        recv = torch.zeros(num_future, device=dev)
        probs, layers = None, []
        mask_v = mask_a = None
        for layer in range(self.num_layers):
            if layer == m and m > 0:
                xv = xv[:, probes] if rows.numel() == num_future else xv
                rows = probes
            if layer == m or (layer == 0 and m == 0):
                if bias != 0.0 and rows.numel() != num_future:
                    n = int(rows.numel())
                    mask_v = torch.zeros((1, 1, 1, f0 + n), device=dev)
                    mask_v[..., f0:] = bias
                    mask_a = torch.zeros((1, 1, 1, f0 + n + n_act), device=dev)
                    mask_a[..., f0:f0 + n] = bias
                    mask_v, mask_a = mask_v.to(xv.dtype), mask_a.to(xv.dtype)
            idx_abs = rows + f0
            n_q = int(rows.numel())
            bv, ba = self.video.blocks[layer], self.action.blocks[layer]
            io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=pre_v["freqs"][idx_abs],
                                                        t_mod=pre_v["t_mod"][:, idx_abs])
            io_a = self.mot._build_expert_attention_io(expert=self.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"])
            k_vid = torch.cat([kv0[layer]["k"], io_v[1]], dim=1)
            v_vid = torch.cat([kv0[layer]["v"], io_v[2]], dim=1)
            out_v = flash_attention(q=io_v[0], k=k_vid, v=v_vid, num_heads=self.num_heads, ctx_mask=mask_v)
            k_all = torch.cat([k_vid, io_a[1]], dim=1)
            v_all = torch.cat([v_vid, io_a[2]], dim=1)
            out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads, ctx_mask=mask_a)
            recv[rows] += self._masked_probs(io_v[0], k_vid, mask_v)[:, f0:].sum(dim=0)
            recv[rows] += self._masked_probs(io_a[0], k_all, mask_a)[:, f0:f0 + n_q].sum(dim=0)
            if layer == score_layer:
                pm = self._masked_probs(io_a[0], k_all, mask_a)
                probs = torch.zeros((n_act, f0 + num_future + n_act), dtype=pm.dtype, device=dev)
                probs[:, :f0] = pm[:, :f0]
                probs[:, f0 + rows] = pm[:, f0:f0 + n_q]
                probs[:, f0 + num_future:] = pm[:, f0 + n_q:]
            layers.append([n_q, int(k_vid.shape[1]), int(io_a[0].shape[1]), int(k_all.shape[1])])
            xv = self._post(bv, io_v, out_v, {"context": pre_v["context"], "mask": pre_v["context_mask"][:, idx_abs]})
            xa = self._post(ba, io_a, out_a, {"context": pre_a["context"], "mask": pre_a["context_mask"]})
        vel = self.video.head(xv, pre_v["t"][:, rows + f0])
        return {"rows": rows, "vel": vel, "pred_action": self.action.post_dit(xa, pre_a), "probs": probs,
                "recv": recv / self.num_layers, "layers": layers}

    def _explore_forward(self, pre_v: dict, pre_a: dict, kv0: list, rows: torch.Tensor, f0: int, num_future: int,
                         score_layer: int, grid: sel.Grid, row_bias: float = 0.0, vscale: float = 1.0,
                         layers: Optional[set] = None, kv_rows: Optional[torch.Tensor] = None, groups: Optional[list] = None,
                         track: Optional[list] = None) -> dict:
        """step0-explore forward beside the main pass (nothing of it is used by the arm).
        ``rows``: future tokens computed in full on the blocks in ``layers`` (every block if None; a skipped block leaves the
        video and action hidden states as they are); their keys + ``row_bias`` (video rows: x ``vscale``). ``kv_rows``: tokens
        that only give keys / values, projected on every run block from their own input embedding of this step (never
        updated). ``groups``: proxy tokens, fully computed: input = the members' mean embedding, RoPE at the members' centroid,
        key + log(members). -> {rows, vel [1, rows, D], proxy_vel [1, G, D], pred_action, probs / probs_layer (the run block
        nearest the score layer; full layout, rows only), recv [Kf] (rows), layers (per block [rows + proxies, video keys,
        action rows, action keys, kv-only tokens] or None if skipped), track (per run block |h - h_track| / |h_track| on rows)}."""
        dev = pre_v["tokens"].device
        rows = rows.to(dev)
        n_act = int(pre_a["tokens"].shape[1])
        n = int(rows.numel())
        idx_abs = rows + f0
        xv = pre_v["tokens"][:, idx_abs]
        freqs, tmod, ctxm, t_rows = pre_v["freqs"][idx_abs], pre_v["t_mod"][:, idx_abs], pre_v["context_mask"][:, idx_abs], pre_v["t"][:, idx_abs]
        sizes: list[int] = []
        if groups:
            sizes = [int(g.numel()) for g in groups]
            xg = torch.stack([pre_v["tokens"][:, g.to(dev) + f0].float().mean(dim=1) for g in groups], dim=1).to(xv.dtype)
            pos = torch.stack([torch.stack([(g // grid.per_frame + 1).double().mean(), ((g % grid.per_frame) // grid.width).double().mean(),
                                            ((g % grid.per_frame) % grid.width).double().mean()]) for g in [x.cpu() for x in groups]])
            gfreq = fv.rope_freqs_at(pos, 2 * int(pre_v["freqs"].shape[-1])).to(device=freqs.device, dtype=freqs.dtype)
            first = torch.stack([g[0] for g in groups]).to(dev) + f0
            xv = torch.cat([xv, xg], dim=1)
            freqs, tmod = torch.cat([freqs, gfreq], dim=0), torch.cat([tmod, pre_v["t_mod"][:, first]], dim=1)
            ctxm, t_rows = torch.cat([ctxm, pre_v["context_mask"][:, first]], dim=1), torch.cat([t_rows, pre_v["t"][:, first]], dim=1)
        nq = n + len(sizes)
        kvn = 0 if kv_rows is None else int(kv_rows.numel())
        if kvn:
            kv_abs = kv_rows.to(dev) + f0
            x_kv, f_kv, t_kv = pre_v["tokens"][:, kv_abs], pre_v["freqs"][kv_abs], pre_v["t_mod"][:, kv_abs]
        kb = torch.zeros(nq + kvn, device=dev)
        kb[:n] = row_bias
        if sizes:
            kb[n:nq] = torch.log(torch.tensor(sizes, dtype=torch.float32, device=dev))
        mask_v = mask_a = None
        if bool((kb != 0).any()):
            kbv = kb.clone()
            kbv[:n] = row_bias * vscale
            mask_v = torch.zeros((1, 1, 1, f0 + nq + kvn), device=dev)
            mask_v[..., f0:] = kbv
            mask_a = torch.zeros((1, 1, 1, f0 + nq + kvn + n_act), device=dev)
            mask_a[..., f0:f0 + nq + kvn] = kb
            mask_v, mask_a = mask_v.to(xv.dtype), mask_a.to(xv.dtype)
        xa = pre_a["tokens"]
        recv = torch.zeros(num_future, device=dev)
        run = set(range(self.num_layers)) if layers is None else set(layers)
        p_layer = max((l for l in run if l <= score_layer), default=min(run))
        probs, counts, trk = None, [], []
        for layer in range(self.num_layers):
            if layer not in run:
                counts.append(None)
                continue
            bv, ba = self.video.blocks[layer], self.action.blocks[layer]
            io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=freqs, t_mod=tmod)
            io_a = self.mot._build_expert_attention_io(expert=self.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"])
            k_f, v_f = io_v[1], io_v[2]
            if kvn:
                io_kv = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=x_kv, freqs=f_kv, t_mod=t_kv)
                k_f, v_f = torch.cat([k_f, io_kv[1]], dim=1), torch.cat([v_f, io_kv[2]], dim=1)
            k_vid = torch.cat([kv0[layer]["k"], k_f], dim=1)
            v_vid = torch.cat([kv0[layer]["v"], v_f], dim=1)
            out_v = flash_attention(q=io_v[0], k=k_vid, v=v_vid, num_heads=self.num_heads, ctx_mask=mask_v)
            k_all, v_all = torch.cat([k_vid, io_a[1]], dim=1), torch.cat([v_vid, io_a[2]], dim=1)
            out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads, ctx_mask=mask_a)
            recv[rows] += self._masked_probs(io_v[0][:, :n], k_vid, mask_v)[:, f0:f0 + n].sum(dim=0)
            recv[rows] += self._masked_probs(io_a[0], k_all, mask_a)[:, f0:f0 + n].sum(dim=0)
            if layer == p_layer:
                pm = self._masked_probs(io_a[0], k_all, mask_a)
                probs = torch.zeros((n_act, f0 + num_future + n_act), dtype=pm.dtype, device=dev)
                probs[:, :f0] = pm[:, :f0]
                probs[:, f0 + rows] = pm[:, f0:f0 + n]
                probs[:, f0 + num_future:] = pm[:, f0 + nq + kvn:]
            counts.append([nq, int(k_vid.shape[1]), int(io_a[0].shape[1]), int(k_all.shape[1]), kvn])
            xv = self._post(bv, io_v, out_v, {"context": pre_v["context"], "mask": ctxm})
            xa = self._post(ba, io_a, out_a, {"context": pre_a["context"], "mask": pre_a["context_mask"]})
            if track is not None:
                ref = track[layer][:, rows].float()
                trk.append(float((xv[:, :n].float() - ref).norm() / ref.norm()))
        vel = self.video.head(xv, t_rows)
        return {"rows": rows, "vel": vel[:, :n], "proxy_vel": vel[:, n:], "pred_action": self.action.post_dit(xa, pre_a),
                "probs": probs, "probs_layer": p_layer, "recv": recv / max(1, len(run)), "layers": counts, "track": trk,
                "sizes": sizes}

    def _explore_rows(self, spec: str, grid: sel.Grid, plan) -> torch.Tensor:
        """step0-explore visible sets (sorted future indices): grid39 (B's probes), rand<n> / strat<n> (random, per block
        stratified; fixed generator), patch39 (each block's central 3 x 3 + 182, 7, 105), agf1 (agentview, frame 1), frame1,
        cam_ag, m4w39 / core23 (the dense plan's sinks + core + step-9 slice / sinks + core), all."""
        kf, pf, half = grid.num_future, grid.per_frame, grid.width // 2
        gen = torch.Generator(device="cpu").manual_seed(1234)
        blocks = [[f * pf + r * grid.width + cam * half + c for r in range(grid.height) for c in range(half)]
                  for f in range(grid.frames - 1) for cam in (0, 1)]
        if spec == "grid39":
            out = fv.probe_index(grid)
        elif spec.startswith("rand"):
            out = torch.randperm(kf, generator=gen)[:int(spec[4:])]
        elif spec.startswith("strat"):
            n = int(spec[5:])
            per = [n // len(blocks) + (1 if i < n % len(blocks) else 0) for i in range(len(blocks))]
            out = torch.cat([torch.tensor(b)[torch.randperm(len(b), generator=gen)[:k]] for b, k in zip(blocks, per)])
        elif spec == "patch39":
            out = torch.tensor([f * pf + r * grid.width + cam * half + c for f in range(grid.frames - 1) for cam in (0, 1)
                                for r in (2, 3, 4) for c in (2, 3, 4)] + [fv.PROBE_SINK, *fv.PROBE_CORNERS])
        elif spec.startswith("f1s"):  # step0-explore: the near frame only, stratified random over its two camera blocks
            n = int(spec[3:])
            out = torch.cat([torch.tensor(b)[torch.randperm(len(b), generator=gen)[:k]] for b, k in zip(blocks[:2], (n - n // 2, n // 2))])
        elif spec == "f1g32":  # the near frame on rows / columns {0, 2, 4, 6} of both cameras
            out = torch.tensor([r * grid.width + cam * half + c for cam in (0, 1) for r in (0, 2, 4, 6) for c in (0, 2, 4, 6)])
        elif spec == "agf1":
            out = torch.tensor(blocks[0])
        elif spec == "frame1":
            out = torch.arange(pf)
        elif spec == "cam_ag":
            out = torch.tensor(blocks[0] + blocks[2])
        elif spec == "m4w39":
            out = plan.real(len(plan.slices)).cpu()
        elif spec == "core23":
            out = plan.fixed.cpu()
        elif spec == "all":
            out = torch.arange(kf)
        else:
            raise ValueError(f"explore rows {spec!r}")
        return torch.sort(torch.unique(out.long())).values

    def _explore_layers(self, spec: str) -> Optional[set]:
        """step0-explore blocks to run: all, first<k>, last<k>, f<a>l<b> (first a + last b), every<k> (0, k, 2k, ...)."""
        L = self.num_layers
        if spec == "all":
            return None
        if spec.startswith("first"):
            return set(range(int(spec[5:])))
        if spec.startswith("last"):
            return set(range(L - int(spec[4:]), L))
        if spec.startswith("every"):
            return set(range(0, L, int(spec[5:])))
        if spec.startswith("f") and "l" in spec:
            a, b = (int(x) for x in spec[1:].split("l"))
            return set(range(a)) | set(range(L - b, L))
        raise ValueError(f"explore layers {spec!r}")

    def _tok_to_diff(self, v: torch.Tensor) -> torch.Tensor:
        """Token layout (x y z c) of the video head -> the Diff layout (c a b e), on the last axis."""
        pt, ph, pw = (int(x) for x in self.video.patch_size)
        c = v.shape[-1] // (pt * ph * pw)
        return v.reshape(*v.shape[:-1], pt, ph, pw, c).movedim(-1, -4).reshape(*v.shape[:-1], -1)

    def _head_logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Per-head q k^T / sqrt(d) for batch 1 -> [H, Sq, Sk] (fp32)."""
        qh = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)[0].float()
        kh = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)[0].float()
        return (qh @ kh.transpose(-1, -2)) * (qh.shape[-1] ** -0.5)

    @staticmethod
    def _steered_probs(logits: torch.Tensor, cols, bias: Optional[torch.Tensor], lam: float) -> torch.Tensor:
        """softmax over keys after adding ``lam * bias`` [H, Sq] to the columns ``cols`` (a slice
        or an index tensor)."""
        if lam != 0.0:
            logits = logits.clone()
            logits[..., cols] += lam * bias.unsqueeze(-1)
        return torch.softmax(logits, dim=-1)

    def _steered_attention(self, q, k, v, cols, bias: Optional[torch.Tensor], lam: float):
        """Plain fp32 attention with ``lam * bias`` on the columns ``cols`` -> (output [1, Sq, H*d]
        in q's dtype, probabilities [H, Sq, Sk]). Action rows of pruned steps (``steer``) and, with
        ``video_eager``, video rows of pruned steps."""
        probs = self._steered_probs(self._head_logits(q, k), cols, bias, lam)
        vh = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)[0].float()
        out = rearrange(probs @ vh, "n s d -> 1 s (n d)").to(dtype=q.dtype)
        return out, probs

    def _steer_bias(self, steer, fut_probs: dict, logits: dict, kept: torch.Tensor, f0: int, num_future: int,
                    dense_bias: bool = False):
        """Per-layer bias b = log(mF / mFk) [H, A], clamped at ``steer.clamp``, from the action rows'
        probabilities at the last warm-up (full) step: mF on all future tokens, mFk on the kept ones.

        Also the formula check: that step's logits restricted to ``[frame-0 | kept | action]``, with
        b added to the kept columns (lam = 1), must give the future share mF wherever b is not
        clamped (adding log(mF/mFk) scales the kept weights so they sum to the dense future weight).
        With ``dense_bias`` and lam = -1, also: the full logits with -b on every future column must
        give the share of the restricted (sparse) row, mFk / (1 - mF + mFk).
        """
        nk = int(kept.numel())
        bias: dict[int, torch.Tensor] = {}
        rows = {"b": [], "err": [], "clamped": [], "m_f": [], "m_fk": [], "err_dense": []}
        for layer in sorted(fut_probs):
            p = fut_probs[layer]
            m_f = p.sum(dim=-1)
            m_fk = p[..., kept].sum(dim=-1)
            raw = torch.log(m_f.clamp_min(1e-30) / m_fk.clamp_min(1e-30))
            b = torch.ones_like(raw) if steer.const else raw.clamp(max=steer.clamp)
            bias[layer] = b
            lg = logits[layer]
            cols = torch.cat([
                torch.arange(f0, device=lg.device), kept + f0,
                torch.arange(f0 + num_future, lg.shape[-1], device=lg.device),
            ])
            share = self._steered_probs(lg[..., cols], slice(f0, f0 + nk), b, 1.0)[..., f0:f0 + nk].sum(dim=-1)
            for key, value in (("b", b), ("err", (share - m_f).abs()), ("clamped", (raw > steer.clamp) & (not steer.const)),
                               ("m_f", m_f), ("m_fk", m_fk)):
                rows[key].append(value)
            if dense_bias and steer.lam == -1.0 and not steer.const:
                fut = slice(f0, f0 + num_future)
                d_share = self._steered_probs(lg, fut, b, -1.0)[..., fut].sum(dim=-1)
                rows["err_dense"].append((d_share - m_fk / (1.0 - m_f + m_fk)).abs())
        t = {key: torch.stack(value) for key, value in rows.items() if value}  # [L, H, A]
        free = ~t["clamped"]
        out = {
            "lam": float(steer.lam), "clamp": float(steer.clamp), "layers": len(bias),
            "b_mean": float(t["b"].mean()), "b_max": float(t["b"].max()), "b_min": float(t["b"].min()),
            "b_layer_mean": [round(float(x), 5) for x in t["b"].mean(dim=(1, 2))],
            "clamped_fraction": float(t["clamped"].float().mean()),
            "check_max_abs": float(t["err"][free].max()) if bool(free.any()) else None,
            "check_max_abs_clamped": float(t["err"][t["clamped"]].max()) if bool(t["clamped"].any()) else None,
            "dense_future_share": float(t["m_f"].mean()),
            "kept_future_share": float(t["m_fk"].mean()),
        }
        if "err_dense" in t:
            out["dense_bias_check_max_abs"] = float(t["err_dense"][free].max()) if bool(free.any()) else None
        if steer.const:  # b = 1: no restoring formula to check
            out.update(const=True, check_max_abs=None, check_max_abs_clamped=None)
        return bias, out

    def _steer_video_bias(self, steer, fut_probs: dict, logits: dict, kept: torch.Tensor, f0: int):
        """Per-layer bias b_v = log(mF / mFk) [H, kept queries], clamped at ``steer.clamp``, from the
        video rows' probabilities at step 0 ([H, Kf queries, Kf keys]): for each kept future query,
        mF on all future keys and mFk on the kept ones. Formula check as in ``_steer_bias``: that
        step's logits of the kept queries restricted to ``[frame-0 | kept]``, plus b_v on the kept
        columns, give the future share mF wherever b_v is not clamped."""
        nk = int(kept.numel())
        bias: dict[int, torch.Tensor] = {}
        rows = {"b": [], "err": [], "clamped": [], "m_f": [], "m_fk": []}
        for layer in sorted(fut_probs):
            p = fut_probs[layer][:, kept]  # [H, nk, Kf], a contiguous copy
            m_f = p.sum(dim=-1)
            m_fk = p[..., kept].sum(dim=-1)  # same layout as p when every token is kept: b_v == 0 exactly
            raw = torch.log(m_f.clamp_min(1e-30) / m_fk.clamp_min(1e-30))
            b = raw.clamp(max=steer.clamp)
            bias[layer] = b
            lg = logits[layer][:, kept]
            cols = torch.cat([torch.arange(f0, device=lg.device), kept + f0])
            share = self._steered_probs(lg[..., cols], slice(f0, f0 + nk), b, 1.0)[..., f0:].sum(dim=-1)
            for key, value in (("b", b), ("err", (share - m_f).abs()), ("clamped", raw > steer.clamp),
                               ("m_f", m_f), ("m_fk", m_fk)):
                rows[key].append(value)
        t = {key: torch.stack(value) for key, value in rows.items()}  # [L, H, nk]
        free = ~t["clamped"]
        return bias, {
            "lam": float(steer.lam), "clamp": float(steer.clamp), "layers": len(bias),
            "b_mean": float(t["b"].mean()), "b_max": float(t["b"].max()), "b_min": float(t["b"].min()),
            "clamped_fraction": float(t["clamped"].float().mean()),
            "check_max_abs": float(t["err"][free].max()) if bool(free.any()) else None,
            "check_max_abs_clamped": float(t["err"][t["clamped"]].max()) if bool(t["clamped"].any()) else None,
            "dense_future_share": float(t["m_f"].mean()),
            "kept_future_share": float(t["m_fk"].mean()),
        }

    @staticmethod
    def _summary_groups(grid, n: int, views: int, device) -> torch.Tensor:
        """Region id of every future token [Kf]: each (future frame, camera view) block of the
        latent grid split ``n`` x ``n`` (rows and columns as ``np.array_split``)."""
        h, w = int(grid.height), int(grid.width)
        wv = w // views
        rows = np.concatenate([np.full(len(c), i) for i, c in enumerate(np.array_split(np.arange(h), n))])
        cols = np.concatenate([np.full(len(c), i) for i, c in enumerate(np.array_split(np.arange(wv), n))])
        f, r, c = np.meshgrid(np.arange(int(grid.frames) - 1), np.arange(h), np.arange(w), indexing="ij")
        gid = ((f * views + c // wv) * n + rows[r]) * n + cols[c % wv]
        return torch.as_tensor(gid.reshape(-1), dtype=torch.long, device=device)

    @staticmethod
    def _summary_build(k_full: torch.Tensor, v_full: torch.Tensor, dropped: torch.Tensor, gid: torch.Tensor):
        """Summary tokens of the dropped tokens: one per non-empty region, K / V = the region's mean
        -> (k [1, G, D], v [1, G, D], counts [G], membership [G, Kf] as 0 / 1)."""
        groups, inverse = torch.unique(gid[dropped], return_inverse=True)
        member = torch.zeros((groups.numel(), gid.numel()), dtype=torch.float32, device=gid.device)
        member[inverse, dropped] = 1.0
        counts = member.sum(dim=1)
        avg = member / counts.unsqueeze(1)
        k = (avg @ k_full[0].float()).unsqueeze(0).to(k_full.dtype)
        v = (avg @ v_full[0].float()).unsqueeze(0).to(v_full.dtype)
        return k, v, counts, member

    def _summary_lse_bias(self, q: torch.Tensor, k_full: torch.Tensor, k_sum: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        """Per head and query [H, Sq, G]: the region's true log-sum-exp of q k / sqrt(d) over its
        members minus the summary key's logit, so that the summary carries the region's exact
        attention weight for that query (at the step these tensors come from)."""
        full = self._head_logits(q, k_full)  # [H, Sq, Kf]
        mask = torch.where(member > 0, 0.0, float("-inf"))  # [G, Kf]
        lse = torch.logsumexp(full.unsqueeze(-2) + mask, dim=-1)  # [H, Sq, G]
        return lse - self._head_logits(q, k_sum)

    def _cal_bias(self, plan, qk: dict, kv0: list, f0: int, kf: int, clamp: float = 20.0):
        """N1 (interlace_cal): per layer, the biases (video [slices, H, Kf], action [slices, H, A]) of slice S_k's keys,
        b = LSE over the periphery - LSE over S_k of each row's step-0 logits (clamped at ``clamp``), and the formula check:
        on the step-0 rows, [frame-0 | fixed | S_k + b (| action)] gives S_k the periphery's share of the full row."""
        per, fixed, n_fix = plan.periphery, plan.fixed, int(plan.fixed.numel())
        out, err, n_clamped, n_el, means = {}, 0.0, 0, 0, {"video": [], "action": []}
        for layer, (qv, qa, k_fut, k_act) in qk.items():
            k0 = kv0[layer]["k"]
            per_rows = []
            for name, q, extra in (("video", qv, None), ("action", qa, k_act)):
                parts = [self._head_logits(q, k0), self._head_logits(q, k_fut)]
                if extra is not None:
                    parts.append(self._head_logits(q, extra))
                full = torch.cat(parts, dim=-1)  # [H, rows, f0 + Kf (+ A)]
                fut = full[..., f0:f0 + kf]
                lse_p = torch.logsumexp(fut[..., per], dim=-1)
                share_p = torch.softmax(full, dim=-1)[..., f0 + per].sum(dim=-1)
                tail = [torch.arange(f0 + kf, full.shape[-1], device=full.device)] if extra is not None else []
                bs = []
                for sl in plan.slices:
                    raw = lse_p - torch.logsumexp(fut[..., sl], dim=-1)  # [H, rows]
                    b = raw.clamp(max=clamp)
                    n_clamped += int((raw > clamp).sum())
                    n_el += int(raw.numel())
                    cols = torch.cat([torch.arange(f0, device=full.device), f0 + fixed, f0 + sl] + tail)
                    lg = full[..., cols].clone()
                    lg[..., f0 + n_fix:f0 + n_fix + sl.numel()] += b.unsqueeze(-1)
                    share = torch.softmax(lg, dim=-1)[..., f0 + n_fix:f0 + n_fix + sl.numel()].sum(dim=-1)
                    free = raw <= clamp
                    if bool(free.any()):
                        err = max(err, float((share - share_p).abs()[free].max()))
                    bs.append(b)
                    means[name].append(float(b.mean()))
                per_rows.append(torch.stack(bs))
            out[layer] = (per_rows[0], per_rows[1])
        n_per = int(per.numel())
        return out, {"check_max_abs": err, "clamped_fraction": n_clamped / max(n_el, 1),
                     "b_video_mean": float(np.mean(means["video"])), "b_action_mean": float(np.mean(means["action"])),
                     "b_count": [round(math.log(n_per / int(x.numel())), 6) for x in plan.slices]}

    @staticmethod
    def _merge_rows(h: torch.Tensor, mg: dict) -> torch.Tensor:
        """merge: every group's input, the fp32 mean of its members' streams (contiguous spans of ``h``), in h's dtype."""
        return torch.stack([h[:, a:b].float().mean(dim=1) for a, b in mg["spans"]], dim=1).to(h.dtype)

    def _io_pre_rope(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor):
        """``mot._build_expert_attention_io`` with the same operations in the same order (so the same
        numbers), also returning the pre-RoPE key k~ (after norm_k) -> (io tuple, k~)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mot._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k_t = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k_t, freqs, block.num_heads)
        ckpt = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, ckpt), k_t

    def _r2_project(self, block, x: torch.Tensor, t_mod: torch.Tensor):
        """R2: hidden states ``x`` with the time modulation ``t_mod`` through modulate(norm1(.)), W_k /
        norm_k and W_v -> (pre-RoPE key k~, value)."""
        shift_msa, scale_msa, *_ = self.mot._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        return block.self_attn.norm_k(block.self_attn.k(attn_input)), block.self_attn.v(attn_input)

    @staticmethod
    def _future_freqs(pre_v: dict, idx: torch.Tensor, f0: int) -> torch.Tensor:
        """RoPE frequencies of future positions ``idx`` (0-based among the future tokens): the rows of
        the full sequence's 3D table, i.e. the rotation of a real token at that (frame, h, w)."""
        return pre_v["freqs"][idx + f0]

    @staticmethod
    def _median0(x: torch.Tensor) -> torch.Tensor:
        """Channel-wise median over rows [n, D] in fp32, numpy's convention (mean of the two middle
        values for even n)."""
        s = torch.sort(x.float(), dim=0).values
        n = int(s.shape[0])
        return 0.5 * (s[(n - 1) // 2] + s[n // 2])

    @staticmethod
    def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine per token over the whole vector (all heads): [1, n, D] x 2 -> [n] (fp32)."""
        a, b = a[0].float(), b[0].float()
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)

    def _cost_model(self) -> dict:
        """Parameter counts for the relative-compute estimate (per block; context K/V projections
        apart, since they run on the context tokens, not on the queries)."""
        if self._cost is None:
            def split(block) -> tuple[int, int]:
                total = sum(p.numel() for p in block.parameters())
                ctx = sum(p.numel() for n, p in block.named_parameters() if n.startswith(("cross_attn.k.", "cross_attn.v.")))
                return int(total), int(ctx)
            pv, pv_ctx = split(self.video.blocks[0])
            pa, pa_ctx = split(self.action.blocks[0])
            attn = self.video.blocks[0].self_attn
            proj = sum(p.numel() for m in (attn.k, attn.v) for p in m.parameters())
            self._cost = {"video_block_params": pv, "video_ctx_kv_params": pv_ctx,
                          "action_block_params": pa, "action_ctx_kv_params": pa_ctx, "layers": self.num_layers,
                          "video_kv_proj_params": int(proj)}
        return dict(self._cost)

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

    def _prefill_frame0(self, pre0: dict, num_text: int, inst_layers: set[int],
                        keep_pre: bool = False) -> tuple[list, dict[int, torch.Tensor]]:
        """Run frame-0 tokens alone through the video expert; return per-layer K/V and,
        for ``inst_layers``, the cross-attention mass on real instruction tokens. With ``keep_pre``
        (prune.recon) also each layer's pre-RoPE key ``kt`` and input hidden states ``x``."""
        x = pre0["tokens"]
        payload = {"context": pre0["context"], "mask": pre0["context_mask"]}
        kv: list[dict[str, torch.Tensor]] = []
        inst_mass: dict[int, torch.Tensor] = {}
        for layer in range(self.num_layers):
            block = self.video.blocks[layer]
            k_t = None
            if keep_pre:
                io, k_t = self._io_pre_rope(self.video, block, x, pre0["freqs"], pre0["t_mod"])
            else:
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
            entry = {"k": k, "v": v}
            if keep_pre:
                entry.update(kt=k_t, x=x)
            x = self._post(block, io, mixed, payload)
            kv.append(entry)
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
        _capture_kv: Optional[dict] = None,
        _capture_video: Optional[dict] = None,
        _step0_tube: Optional[Callable] = None,
        _memory_key: Any = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.cfg
        model = self.model
        steps = int(cfg.num_inference_steps)
        if cfg.prune is not None and (cfg.prune.side0_use is not None or cfg.prune.mem_route):
            # step0-sparse-r2: R1 / R3b / R7 score on casestudy.score_layer (B's layer; R7 has no dense step 0 from call 1 on).
            self._layers = {"act": int(cfg.score_layer), "inst": -1}
        if cfg.prune is not None and cfg.prune.step0 is not None:
            # step0-sparse: no dense step 0 to choose the layer on; A->V always on casestudy.score_layer.
            self._layers = {"act": int(cfg.score_layer), "inst": -1}
            if cfg.prune.step0 == "d" and _step0_tube is None:
                raise ValueError("prune.step0 d needs the adapter's end-effector tube (_step0_tube)")
        # X2 (prune.oracle_kv): a dense pass of the same call and seed first, keeping every
        # step's future K/V per layer for the action rows of the pruned pass below.
        # record_recon_quality: the same dense pass, its K/V only compared with the reconstruction.
        oracle: Optional[dict] = None
        dense_kv: Optional[dict] = None
        if cfg.prune is not None and (cfg.prune.oracle_kv or cfg.record_recon_quality):
            dense_kv = {}
            dense_cfg = replace(cfg, method="dense", prune=None, record_attention=False, record_future_share=False,
                                record_motion=False, record_video_frame0=False, record_motion_steps=False, dump=None,
                                record_recon_quality=False, record_pilot=False, agc=None)
            JointCaseStudyRunner(model, dense_cfg, prompt_store=self.prompt_store).infer_action(
                prompt, input_image, proprio, action_horizon, num_video_frames, seed,
                sigma_shift=sigma_shift, rand_device=rand_device, _capture_kv=dense_kv,
            )
            if cfg.prune.oracle_kv:
                oracle = dense_kv
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
            num_inference_steps=steps, device=model.device, dtype=latents_video.dtype,
            shift_override=sigma_shift if cfg.video_sigma_shift is None else cfg.video_sigma_shift,
        )
        t_action, d_action = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_action.dtype, shift_override=sigma_shift
        )

        prune = cfg.prune
        is_cf = prune is not None and prune.selection == "cf"
        first_call = self._layers is None
        want_scores = prune is not None and prune.selection in ("av", "vv", "cf", "diff", "diff_global")
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
        recon = prune.recon if prune is not None else None
        kv0, inst_mass = self._prefill_frame0(pre0, num_text, inst_layers, keep_pre=recon is not None)

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
        # A->V attention that scores the kept set: step 0 unless P4 moves it (score_step).
        score_step = prune.score_step if prune is not None and prune.score_step is not None else 0
        act_probs_sel: dict[int, torch.Tensor] = {}
        vid_step_probs: dict[int, torch.Tensor] = {}
        k_future_store: dict[int, torch.Tensor] = {}
        # P5 / V2: on pruned steps the action rows (action_full_kv) and/or the video rows
        # (video_full_kv) also see the dropped tokens' K/V of the last full step. The SDPA path
        # keeps those per layer as one block once the kept set is known (keys [frame-0 | dropped |
        # kept]); a later full step (prune.full_steps) refreshes it. The fp32 diagnostic path
        # (video_eager) keeps every future position in order and overwrites the kept ones.
        full_kv = prune is not None and prune.action_full_kv
        video_full_kv = prune is not None and prune.video_full_kv
        cache_kv = full_kv or video_full_kv
        video_eager = prune is not None and prune.video_eager
        kv_future_full: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        kv_drop: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        dropped: Optional[torch.Tensor] = None
        # E: summary tokens of the dropped tokens (step-0 K/V averaged per region) and the additive
        # logit masks of the video / action rows (log count, or the per-query log-sum-exp gap).
        summary = prune is not None and prune.summary_grid > 0
        summary_lse = summary and prune.summary_bias == "lse"
        kv_sum: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        sum_mask_v: dict[int, torch.Tensor] = {}
        sum_mask_a: dict[int, torch.Tensor] = {}
        q0_v: dict[int, torch.Tensor] = {}
        q0_a: dict[int, torch.Tensor] = {}
        n_sum = 0
        # P6: future tokens outside the listed latent frames cannot be kept.
        frame_exclude = None
        if prune is not None and prune.score_frames is not None:
            bad = [f for f in prune.score_frames if not 1 <= f < grid.frames]
            if bad:
                raise ValueError(f"prune.score_frames {bad} outside the future frames 1..{grid.frames - 1}")
            allowed = torch.tensor(sorted(set(prune.score_frames)), device=model.device)
            frame_exclude = torch.nonzero(~torch.isin(grid.future_frame_index(model.device), allowed)).flatten()
        steer = prune.steer if prune is not None else None
        want_share = cfg.record_future_share
        # Steer: per layer, the action rows' future probabilities [H, A, Kf] and logits at the last
        # warm-up step (until the kept set is known), then the bias [H, A] for the pruned steps.
        steer_fut: dict[int, torch.Tensor] = {}
        steer_logits: dict[int, torch.Tensor] = {}
        steer_bias: dict[int, torch.Tensor] = {}
        share_rows: list[torch.Tensor] = []
        # Context probes: video rows of pruned steps in fp32 (video_eager), their re-budget
        # (steer_video: step-0 future probabilities [H, Kf, Kf] and logits, then the bias
        # [H, kept]), dropped K/V for the video rows too (video_full_kv), and dense_bias.
        steer_v = prune.steer_video if prune is not None else None
        dense_bias = prune is not None and prune.dense_bias
        steer_vfut: dict[int, torch.Tensor] = {}
        steer_vlogits: dict[int, torch.Tensor] = {}
        steer_vbias: dict[int, torch.Tensor] = {}
        want_vf0 = cfg.record_video_frame0
        vf0_rows: list[torch.Tensor] = []
        want_motion_steps = cfg.record_motion_steps
        motion_rows: list[torch.Tensor] = []
        want_pilot = cfg.record_pilot or cfg.agc is not None
        # S0-noF: action rows of pruned steps without future keys; EX: a second, no-future action stream on
        # pruned steps and the extrapolated velocity v_s + w * (v_s - v_noF).
        want_fattn = cfg.record_future_attn
        fattn_rows: list[torch.Tensor] = []
        # diff selections: the warm-up step's clean-estimate distance of every future token to the current frame
        diff_score: Optional[torch.Tensor] = None
        # Fovea round: the plan of the pruned steps (fovea.py), the step-0 attention received by every future token, the
        # step-0 clean estimate of the future tokens (k-means), the synthetic region tokens' members / RoPE rows and the
        # interlace bookkeeping (how often each future token was computed on a pruned step).
        fovea_m = prune.fovea if prune is not None else None
        fplan: Optional[fv.Plan] = None
        want_recv = cfg.record_sinks or fovea_m is not None
        recv_acc = torch.zeros(num_future, device=model.device) if want_recv else None
        x0_fut: Optional[torch.Tensor] = None
        fv_members: list[torch.Tensor] = []
        fv_first: Optional[torch.Tensor] = None
        fv_freqs = None
        fv_count = torch.zeros(num_future, dtype=torch.long, device=model.device)
        fv_rows: list[int] = []
        # N1-N4: interlace_cal's step-0 queries / keys per layer, then its per-layer biases (video [slices, H, Kf], action
        # [slices, H, A]) and each slice's positions among its step's rows; interlace_fill's nearest-computed map per slice;
        # merge's grouping of the periphery (absolute indices, group of each member, group sizes, member spans).
        fv_cal = fovea_m == "interlace_cal"
        fv_fill = fovea_m == "interlace_fill"
        fv_merge = fovea_m == "merge"
        cal_qk: dict[int, tuple] = {}
        cal_bias: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        cal_pos: list[torch.Tensor] = []
        fill_src: list[torch.Tensor] = []
        fill_checked = False
        mg: Optional[dict] = None
        # step0-sparse: step 0's probes (every future token for step0_all) and their key bias; plan b's interpolated clean
        # estimate (its Diff), plan d's pilot, the step-0 facts; E9's step-0 latents (to redo step 0's update).
        s0_mode = prune.step0 if prune is not None else None
        s0_probes: Optional[torch.Tensor] = None
        s0_bias = 0.0
        if s0_mode is not None:
            s0_probes = all_future if prune.step0_all else (fv.probe_index(grid) if prune.step0_rows is None else
                                                             self._explore_rows(prune.step0_rows, grid, None)).to(model.device)
            if prune.late is not None:  # step0-explore late anchor: step 0 computes no future token
                s0_probes = all_future[:0]
            if not (prune.step0_all or prune.step0_nobias or prune.late is not None):
                s0_bias = math.log(num_future / int(s0_probes.numel()))
        s0_zf: Optional[torch.Tensor] = None
        s0_pilot: Optional[torch.Tensor] = None
        s0_info: dict[str, Any] = {}
        e9 = prune.e9 if prune is not None else None
        e9_prev: Optional[torch.Tensor] = None
        # step0-explore: the dense step-0 hidden states (explore_track), D1's draft modification of M4w (step 0's update redone).
        explore_h: Optional[list] = [] if cfg.explore_track else None
        dmod = prune.draft_mod if prune is not None else None
        # step0-explore C2F: the raw step-0 x^0 of the probes, their A->V / Diff, then step 1's measurements (the plan of steps 2-9).
        c2f: Optional[dict] = {} if prune is not None and prune.c2f is not None else None
        # step0-explore late anchor: step 0 computes no future token, step 1 computes the set ``prune.late`` and forms the
        # imagination (drafts spread from it), the plan of steps 2-9 comes from step 1.
        late = prune.late if prune is not None else None
        late_st: dict = {"recv": torch.zeros(num_future, device=model.device)} if late is not None else {}
        z_late = rearrange(latents_video[0, :, 0:int(patch[0])].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=int(patch[0]),
                           b=int(patch[1]), e=int(patch[2]))[0] if late is not None else None
        z_c2f = rearrange(latents_video[0, :, 0:int(patch[0])].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=int(patch[0]),
                          b=int(patch[1]), e=int(patch[2]))[0] if c2f is not None else None
        # step0-sparse-r2: R1 / R3b side forward on step 0 (its plan inputs / outputs), R5 / R7 step-by-step drafts, the call's
        # imagination (prune.mem: A->V of every token's last computed step, step-9 recv), R7's plan / drafts from the previous
        # call (plan known before step 0, step 0 pruned), R7b's latest clean estimates; R2 / R3 side records (dense).
        side0 = prune.side0_use if prune is not None else None
        side0_inputs: Optional[dict] = None
        draft_from = prune.draft_from if prune is not None else None
        mem_on = prune is not None and prune.mem
        mem_key = "exec" if _memory_key is None else _memory_key
        mem_av = torch.zeros(num_future, device=model.device) if mem_on else None
        mem_recv = torch.zeros(num_future, device=model.device) if mem_on else None
        route_plan = False
        route_map: Optional[torch.Tensor] = None
        x0_est: Optional[torch.Tensor] = None
        reselected = False
        r2_rows: dict = {}
        r3_rows: dict = {}
        z_cur0 = rearrange(latents_video[0, :, 0:int(patch[0])].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)",
                           a=int(patch[0]), b=int(patch[1]), e=int(patch[2]))[0] if (mem_on or side0 is not None) else None
        if mem_on and mem_key in self._memory:
            # R6: the plan the previous call's imagination gives this call, both maps (interpolated / every terminal aligned).
            prev = self._memory[mem_key]
            k_mem = max(1, int(round(prune.keep_ratio * num_future)))
            sinks_mem = torch.cat([torch.tensor([fv.PROBE_SINK], device=model.device), prev["sinks2"]])
            probes = fv.probe_index(grid).to(model.device)
            mlog: dict[str, Any] = {"sinks": sorted(sinks_mem.tolist())}
            for name, direct in (("interp", False), ("direct", True)):
                mmap = fv.memory_map(prev["terminal"], prev["computed"], grid, direct=direct)
                diff_m = sel.diff_scores(mmap, z_cur0, grid)
                plan_m = fv.make_plan("interlace_w", k_mem, grid, prev["av"].float().cpu(), diff_m.float().cpu(),
                                      torch.zeros(num_future), sinks=sinks_mem.cpu(), n_slices=10, from_step0=True)
                mlog[name] = {"core": plan_m.core.tolist(), "diff_top10": sel.topk_indices(diff_m, 10).tolist(),
                              "probe_f32": _f32_b64(mmap[probes])}
                if prune.mem_route and direct == prune.mem_direct:
                    fplan, route_map, route_plan = plan_m.to(model.device), mmap, True
                    kept = fplan.fixed
                    mlog["used"] = name
                    mlog["slices"] = [int(x.numel()) for x in fplan.slices]
                    mlog["truncated"] = int(fplan.truncated.numel())
                    diag["fovea"] = {"method": "interlace_w", "route": True, "k": int(k_mem), "sinks": fplan.sinks.tolist(),
                                     "core": int(fplan.core.numel()), "periphery": int(fplan.periphery.numel()),
                                     "slices": mlog["slices"], "truncated": mlog["truncated"],
                                     "core_index": fplan.core.tolist(), "slice_bias": [round(x, 6) for x in fplan.slice_bias]}
                    diag["kept_index"] = kept.tolist()
            route_av = prev["av"]
            diag["mem"] = mlog
            if route_plan and prune.reselect_step is not None:
                x0_est = route_map.clone()
        no_future = prune is not None and prune.action_no_future
        ex_w = prune.extrapolate_w if prune is not None else None
        cost_extra: list[list[int]] = []
        # Reconstruction (prune.recon): the per-call layout of the dropped positions (set with the kept
        # set), the quality against dense_kv per [k, v, k step-0, v step-0] x step x layer x view, and
        # the self-checks (worst |diff| and the matching |true| scale per check).
        rc: Optional[dict] = None
        want_rq = cfg.record_recon_quality
        rq = torch.full((4, steps, self.num_layers, 2), float("nan"), device=model.device) if want_rq else None
        checks: dict[str, torch.Tensor] = {}

        def _check(name: str, got: torch.Tensor, true: torch.Tensor) -> None:
            if not got.numel():
                return
            new = torch.stack([(got.float() - true.float()).abs().max(), true.float().abs().max()])
            checks[name] = torch.maximum(checks[name], new) if name in checks else new

        if recon is not None and want_rq:
            for layer in range(self.num_layers):  # RoPE of frame 0's own pre-RoPE key = its true key
                _check("frame0_rope", rope_apply(kv0[layer]["kt"], pre0["freqs"], self.num_heads), kv0[layer]["k"])
        pt, ph, pw = (int(x) for x in patch)
        cost_steps: list[list[int]] = []
        cost_recon: list[int] = []
        cost_dim = 0
        dump = cfg.dump
        d_future = d_obs = d_act = d_hidden = d_cur = None
        d_latents: list[torch.Tensor] = []
        pre_v_freeze: Optional[dict] = None

        for s in range(steps):
            ts_v = t_video[s].unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
            ts_a = t_action[s].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
            pre_v = self.video.pre_dit(
                x=latents_video, timestep=ts_v, context=context, context_mask=context_mask,
                action=None, fuse_vae_embedding_in_latents=fuse,
            )
            if cfg.video_freeze:
                # fovea-w G4: every step recomputes the video stream from step 0's latents and sigma, so the action
                # rows read freshly computed future K/V equal to step 0's (V2-k0 reads them from a cache).
                if s == 0:
                    pre_v_freeze = dict(pre_v)
                else:
                    pre_v = pre_v_freeze
            pre_a = self.action.pre_dit(action_tokens=latents_action, timestep=ts_a, context=context, context_mask=context_mask)

            # ---- mode for this step
            mode = "full"
            if (prune is not None and not dense_bias and s >= prune.warmup_steps and not gated_dense
                    and s not in prune.full_steps):
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

            if route_plan:
                mode = "pruned"  # R7: the plan is known before step 0, every step is a pruned step
                if prune.reselect_step is not None and s == prune.reselect_step:
                    # R7b: the core recomposed before this step from the previous call's A->V and the Diff of the latest
                    # clean estimates (computed tokens: their last x^0; the others: their drafts).
                    old_core = set(fplan.core.tolist())
                    fplan = fv.reselect(fplan, route_av.float().cpu(), sel.diff_scores(x0_est, z_cur0, grid).float().cpu())
                    kept, reselected = fplan.fixed, True
                    diag["reselect"] = {"step": int(s), "core": fplan.core.tolist(),
                                        "entered": sorted(set(fplan.core.tolist()) - old_core),
                                        "left": sorted(old_core - set(fplan.core.tolist())),
                                        "slice_sizes": [int(x.numel()) for x in fplan.slices]}
            fv_step = fplan is not None and mode == "pruned"
            p0_step = s0_probes is not None and s == 0
            idx_rel = (fplan.real(s) if fv_step else kept) if mode == "pruned" else (s0_probes if p0_step else all_future)
            idx_abs = idx_rel + f0
            xv = pre_v["tokens"][:, idx_abs]
            n_syn = len(fv_members) if fv_step else 0
            merge_step = n_syn > 0 and mg is not None
            h_mem = None
            if merge_step:
                # merge: every periphery member's own residual stream, from its own token embedding of this step.
                h_mem = pre_v["tokens"][:, mg["abs"]]
                xv = torch.cat([xv, self._merge_rows(h_mem, mg)], dim=1)
            elif n_syn:
                # Fov: each region token's input is the mean of its members' token embeddings (the patch embedding is
                # linear, so this is the embedding of the members' mean latent), recomputed from this step's latents.
                xv_syn = torch.stack([pre_v["tokens"][:, m].float().mean(dim=1) for m in fv_members], dim=1).to(xv.dtype)
                xv = torch.cat([xv, xv_syn], dim=1)
            xa = pre_a["tokens"]
            xv_in, xa_in = xv, xa
            ex_step = ex_w is not None and mode == "pruned"
            xa_nf = pre_a["tokens"] if ex_step else None

            fattn_acc = torch.zeros(f0 + num_future, device=model.device) if want_fattn and mode != "skip" else None
            share_layers: list[torch.Tensor] = []
            vf0_layers: list[torch.Tensor] = []
            if mode == "skip":
                xv = xv_in + tea_res_v
                xa = xa_in + tea_res_a
                tea_consecutive += 1
            else:
                capture_layers = act_layers if (s == 0 and capture_first_step and not route_plan) else set()
                if mode == "pruned" and prune.reselect_every_step:
                    capture_layers = {self._layers["act"]}
                if want_scores and score_step > 0 and s == score_step:
                    # P4: score on a later full step; the layer was fixed at step 0.
                    capture_layers = {self._layers["act"]}
                freqs_v = pre_v["freqs"][idx_abs]
                tmod_v = pre_v["t_mod"][:, idx_abs]
                ctx_rows = pre_v["context_mask"][:, idx_abs]
                if n_syn:
                    # Region tokens: fractional RoPE position, the members' (shared) time modulation and context mask.
                    freqs_v = torch.cat([freqs_v, fv_freqs.to(device=freqs_v.device, dtype=freqs_v.dtype)], dim=0)
                    tmod_v = torch.cat([tmod_v, pre_v["t_mod"][:, fv_first]], dim=1)
                    ctx_rows = torch.cat([ctx_rows, pre_v["context_mask"][:, fv_first]], dim=1)
                payload_v = {"context": pre_v["context"], "mask": ctx_rows}
                payload_a = {"context": pre_a["context"], "mask": pre_a["context_mask"]}
                step_probs: dict[int, torch.Tensor] = {}
                n_q = int(idx_rel.numel()) + n_syn  # 0 on the pruned steps of keep 0: no video rows at all
                fv_mask_v = fv_mask_a = None
                if fv_step or p0_step:
                    # Key logit bias log(n_g) of representatives (rep) and region tokens (fov), for every row; no mask at
                    # all without a bias, so that the kernel is S0's. step0-sparse: every probe key + log(Kf / 39) on step 0.
                    bias = torch.zeros(n_q, device=model.device)
                    if p0_step:
                        bias += s0_bias
                    else:
                        for j, lg in fplan.bias_at(s).items():  # rep: log n; interlace_w: this step's slice log(N / n_k)
                            bias[torch.nonzero(idx_rel == j).flatten()] = lg
                        for g, (_, lg) in enumerate(fplan.regions):
                            bias[int(idx_rel.numel()) + g] = lg
                    if bool((bias != 0).any()):
                        n_act = int(pre_a["tokens"].shape[1])
                        fv_mask_v = torch.zeros((1, 1, 1, f0 + n_q), device=model.device)
                        fv_mask_v[..., f0:] = bias
                        fv_mask_a = torch.zeros((1, 1, 1, f0 + n_q + n_act), device=model.device)
                        fv_mask_a[..., f0:f0 + n_q] = bias
                        fv_mask_v, fv_mask_a = fv_mask_v.to(xv.dtype), fv_mask_a.to(xv.dtype)
                    if fv_step:
                        fv_rows.append(n_q)
                    elif int(idx_rel.numel()) != int(s0_probes.numel()):
                        raise RuntimeError("step0: step 0 does not compute exactly the probes")
                rc_step = recon is not None and mode == "pruned" and rc is not None and rc["n"] > 0
                if rc_step and recon == "r2":
                    tmod_src = pre_v["t_mod"][:, f0 + rc["src_u"]]  # this step's future-frame modulation
                if recon is not None:  # R2's extra projections this step (frame-0 positions per layer)
                    cost_recon.append(int(rc["src_u"].numel()) if rc_step and recon == "r2" else 0)
                for layer in range(self.num_layers):
                    bv = self.video.blocks[layer]
                    ba = self.action.blocks[layer]
                    io_v = k_t = None
                    fill_now = fv_step and bool(fill_src)
                    xg_in = xv[:, int(idx_rel.numel()):] if merge_step else None  # merge: the groups' input to this layer
                    cal_v = cal_a = None
                    if fv_step and cal_bias:
                        # N1: this step's slice keys + b of their row (step-0 calibration), per head; video rows by token.
                        ki = (s - 1) % len(fplan.slices)
                        b_v, b_a = cal_bias[layer]
                        cols = f0 + cal_pos[ki]
                        n_act = int(pre_a["tokens"].shape[1])
                        cal_v = torch.zeros((1, self.num_heads, n_q, f0 + n_q), device=model.device)
                        cal_v[0, :, :, cols] = b_v[ki][:, idx_rel].unsqueeze(-1).expand(-1, -1, cols.numel())
                        cal_a = torch.zeros((1, self.num_heads, n_act, f0 + n_q + n_act), device=model.device)
                        cal_a[0, :, :, cols] = b_a[ki].unsqueeze(-1).expand(-1, -1, cols.numel())
                        cal_v, cal_a = cal_v.to(xv.dtype), cal_a.to(xv.dtype)
                    if n_q and (recon is not None or fill_now):
                        io_v, k_t = self._io_pre_rope(self.video, bv, xv, freqs_v, tmod_v)
                    elif n_q:
                        io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=freqs_v, t_mod=tmod_v)
                    io_a = self.mot._build_expert_attention_io(
                        expert=self.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
                    )
                    k_new, v_new = (io_v[1], io_v[2]) if io_v is not None else (kv0[layer]["k"][:, :0], kv0[layer]["v"][:, :0])
                    if fill_now:
                        # N2: every future position's key / value from the token computed on this step nearest to it (its
                        # pre-RoPE key rotated to the position's own RoPE); a computed token keeps its own.
                        src = fill_src[(s - 1) % len(fill_src)]
                        k_new = rope_apply(k_t[:, src], pre_v["freqs"][f0:f0 + num_future], self.num_heads)
                        v_new = io_v[2][:, src]
                        if not fill_checked:
                            diag["fovea"]["fill_self_key_bitwise"] = bool(torch.equal(k_new[:, idx_rel], io_v[1]))
                            fill_checked = True
                    k_vid = torch.cat([kv0[layer]["k"], k_new], dim=1)
                    v_vid = torch.cat([kv0[layer]["v"], v_new], dim=1)
                    if rc_step:
                        # Reconstructed K/V of the dropped positions from the frame-0 token at the same (h, w).
                        if recon == "r2":
                            kt_s, v_s = self._r2_project(bv, kv0[layer]["x"][:, rc["src_u"]], tmod_src)
                            kt_hat, v_hat = kt_s[:, rc["inv"]], v_s[:, rc["inv"]]
                        else:
                            kt_hat, v_hat = kv0[layer]["kt"][:, rc["src"]], kv0[layer]["v"][:, rc["src"]]
                            if recon == "r1":
                                dk = k_t[0].float() - kv0[layer]["kt"][0, rc["kept_src"]].float()
                                dv = io_v[2][0].float() - kv0[layer]["v"][0, rc["kept_src"]].float()
                                delta_k = torch.zeros(kt_hat.shape[1:], dtype=torch.float32, device=kt_hat.device)
                                delta_v = torch.zeros_like(delta_k)
                                for g_src, rows_kept, rows_drop in rc["groups"]:
                                    delta_k[rows_drop] = self._median0(dk[rows_kept])
                                    delta_v[rows_drop] = self._median0(dv[rows_kept])
                                kt_hat = (kt_hat.float() + delta_k).to(kt_hat.dtype)
                                v_hat = (v_hat.float() + delta_v).to(v_hat.dtype)
                        k_hat = rope_apply(kt_hat, rc["freqs"], self.num_heads)
                        k_vid = torch.cat([k_vid, k_hat], dim=1)
                        v_vid = torch.cat([v_vid, v_hat], dim=1)
                        if want_rq:
                            k_d, v_d = dense_kv[(s, layer)]
                            k_d, v_d = k_d[:, rc["dropped"]], v_d[:, rc["dropped"]]
                            k_0, v_0 = dense_kv[(0, layer)]
                            vals = (self._cos_rows(k_hat, k_d), self._cos_rows(v_hat, v_d),
                                    self._cos_rows(k_0[:, rc["dropped"]], k_d), self._cos_rows(v_0[:, rc["dropped"]], v_d))
                            for i, c in enumerate(vals):
                                for view, m in rc["view_masks"]:
                                    rq[i, s, layer, view] = c[m].mean()
                    if recon is not None and want_rq and mode == "pruned" and io_v is not None:
                        # Self-checks on the kept tokens: the recon path's RoPE on their own k~, and R2's
                        # projection on their own input hidden states and modulation.
                        _check("kept_rope", rope_apply(k_t, self._future_freqs(pre_v, idx_rel, f0), self.num_heads), io_v[1])
                        kt_c, v_c = self._r2_project(bv, xv, tmod_v)
                        _check("r2_key", kt_c, k_t)
                        _check("r2_value", v_c, io_v[2])
                    if _capture_kv is not None and io_v is not None:
                        _capture_kv[(s, layer)] = (io_v[1], io_v[2])
                    if summary_lse and s == 0:
                        q0_v[layer], q0_a[layer] = io_v[0], io_a[0]
                    if (cache_kv or summary) and mode == "full":
                        if dropped is None or video_eager:
                            kv_future_full[layer] = (io_v[1], io_v[2])
                        else:  # a full step after warm-up refreshes the dropped tokens' block
                            kv_drop[layer] = (io_v[1][:, dropped], io_v[2][:, dropped])
                    k_fut = v_fut = k_drop = v_drop = amask = None
                    if cache_kv and mode == "pruned":
                        if video_eager:
                            if layer in kv_future_full:
                                # Diagnostic layout: every future position, the kept ones fresh.
                                k_fut, v_fut = (x.clone() for x in kv_future_full[layer])
                                k_fut[:, idx_rel] = io_v[1]
                                v_fut[:, idx_rel] = io_v[2]
                        else:
                            k_drop, v_drop = kv_drop[layer]
                    vf0 = None
                    n_vk = 0
                    if io_v is None:
                        out_v = None  # keep 0: no video rows on this step
                    elif video_eager and mode == "pruned":
                        if video_full_kv and k_fut is not None:
                            # V2: the video rows also see the dropped tokens' K/V of the last full step.
                            kv_k = torch.cat([kv0[layer]["k"], k_fut], dim=1)
                            kv_v = torch.cat([kv0[layer]["v"], v_fut], dim=1)
                            v_cols = idx_rel + f0
                        else:
                            kv_k, kv_v, v_cols = k_vid, v_vid, slice(f0, f0 + n_q)
                        out_v, pv = self._steered_attention(
                            io_v[0], kv_k, kv_v, v_cols, steer_vbias.get(layer), steer_v.lam if steer_v is not None else 0.0
                        )
                        vf0 = pv[..., :f0].sum(dim=-1)
                        n_vk = int(kv_k.shape[1])
                    else:
                        vmask = None
                        if video_full_kv and k_drop is not None:
                            kv_k = torch.cat([kv0[layer]["k"], k_drop, io_v[1]], dim=1)
                            kv_v = torch.cat([kv0[layer]["v"], v_drop, io_v[2]], dim=1)
                        elif summary and mode == "pruned" and n_sum:
                            k_s, v_s = kv_sum[layer]
                            kv_k = torch.cat([kv0[layer]["k"], k_s, io_v[1]], dim=1)
                            kv_v = torch.cat([kv0[layer]["v"], v_s, io_v[2]], dim=1)
                            vmask = sum_mask_v[layer]
                        else:
                            kv_k, kv_v = k_vid, v_vid
                            if fv_step or p0_step:
                                vmask = cal_v if cal_v is not None else fv_mask_v
                        out_v = flash_attention(q=io_v[0], k=kv_k, v=kv_v, num_heads=self.num_heads, ctx_mask=vmask)
                        n_vk = int(kv_k.shape[1])
                        steer_v_warm = steer_v is not None and s == 0
                        if want_vf0 or steer_v_warm:
                            # Beside the forward pass, as for the action rows below.
                            logits_v = self._head_logits(io_v[0], kv_k)
                            if vmask is not None:
                                logits_v = logits_v + vmask[0].float()
                            pv = torch.softmax(logits_v, dim=-1)
                            vf0 = pv[..., :f0].sum(dim=-1)
                            if steer_v_warm:
                                steer_vfut[layer] = pv[..., f0:]
                                steer_vlogits[layer] = logits_v
                    if want_vf0 and vf0 is not None:
                        vf0_layers.append(vf0.mean(dim=0))  # [queries], head mean
                    if full_kv and k_fut is not None:
                        k_all = torch.cat([kv0[layer]["k"], k_fut, io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_fut, io_a[2]], dim=1)
                        kept_cols = idx_rel + f0
                    elif full_kv and k_drop is not None:
                        k_all = torch.cat([kv0[layer]["k"], k_drop, k_new, io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_drop, v_new, io_a[2]], dim=1)
                        n_d = int(k_drop.shape[1])
                        kept_cols = slice(f0 + n_d, f0 + n_d + n_q)
                    elif summary and mode == "pruned" and n_sum:
                        k_s, v_s = kv_sum[layer]
                        k_all = torch.cat([kv0[layer]["k"], k_s, io_v[1], io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_s, io_v[2], io_a[2]], dim=1)
                        kept_cols = slice(f0 + n_sum, f0 + n_sum + n_q)
                        amask = sum_mask_a[layer]
                    elif oracle is not None and mode == "pruned":
                        # X2: the kept tokens' K/V of the dense pass at this step and layer.
                        k_o, v_o = oracle[(s, layer)]
                        k_all = torch.cat([kv0[layer]["k"], k_o[:, idx_rel], io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_o[:, idx_rel], io_a[2]], dim=1)
                        kept_cols = slice(f0, f0 + n_q)
                    elif fv_step or p0_step:
                        # Fovea: the action rows see frame 0, this step's computed future rows (real and region
                        # tokens, nothing older; N2: every position filled from them) and themselves. step0-sparse:
                        # step 0's probes with their bias.
                        k_all = torch.cat([k_vid, io_a[1]], dim=1)
                        v_all = torch.cat([v_vid, io_a[2]], dim=1)
                        n_fk = int(k_new.shape[1])
                        kept_cols = slice(f0, f0 + n_fk)
                        amask = cal_a if cal_a is not None else fv_mask_a
                        if n_fk != (num_future if fill_now else n_q) or int(k_all.shape[1]) != f0 + n_fk + int(io_a[1].shape[1]):
                            raise RuntimeError("fovea: the action rows see keys that were not computed on this step")
                    elif no_future and mode == "pruned":
                        # S0-noF: the action rows see frame 0 and themselves, no future token.
                        k_all = torch.cat([kv0[layer]["k"], io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], io_a[2]], dim=1)
                        kept_cols = slice(f0, f0)
                    else:
                        k_all = torch.cat([k_vid, io_a[1]], dim=1)
                        v_all = torch.cat([v_vid, io_a[2]], dim=1)
                        kept_cols = slice(f0, f0 + n_q)
                    if layer == 0:
                        cost_steps.append([n_q, n_vk, int(io_a[0].shape[1]), int(k_all.shape[1])])
                        cost_dim = int(io_a[0].shape[-1])
                        if ex_w is not None:  # EX's no-future action pass: [queries, keys]
                            n_a = int(io_a[0].shape[1])
                            cost_extra.append([n_a, f0 + n_a] if ex_step else [0, 0])
                    biased = steer is not None and (mode == "pruned" or (dense_bias and s >= prune.warmup_steps))
                    if biased:
                        nf = k_all.shape[1] - f0 - io_a[1].shape[1]
                        cols = slice(f0, f0 + nf) if dense_bias else kept_cols
                        out_a, pa_ = self._steered_attention(io_a[0], k_all, v_all, cols, steer_bias[layer], steer.lam)
                        share = pa_[..., f0:f0 + nf].sum(dim=-1)
                    else:
                        out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads, ctx_mask=amask)
                        share = None
                        steer_warm = steer is not None and s == prune.warmup_steps - 1
                        if want_share or steer_warm:
                            # Beside the forward pass: the SDPA output above is what the arm uses.
                            logits = self._head_logits(io_a[0], k_all)
                            if amask is not None:
                                logits = logits + amask[0].float()
                            nf = k_all.shape[1] - f0 - io_a[1].shape[1]
                            fut = torch.softmax(logits, dim=-1)[..., f0:f0 + nf]
                            share = fut.sum(dim=-1)
                            if steer_warm:
                                # Contiguous, so that with every token kept mFk is summed exactly like mF.
                                steer_fut[layer] = fut.contiguous()
                                steer_logits[layer] = logits
                    if want_share:
                        share_layers.append(share.mean())
                    if dump is not None and dump.attention:
                        pa = self._attention_probs(io_a[0], k_all)  # [A, f0 + future keys + A]
                        nf = k_all.shape[1] - f0 - io_a[1].shape[1]
                        if d_future is None:
                            d_future = torch.zeros((steps, self.num_layers, pa.shape[0], num_future), dtype=torch.float16)
                            d_obs = torch.zeros((steps, self.num_layers, pa.shape[0]), dtype=torch.float16)
                            d_act = torch.zeros_like(d_obs)
                        fut_idx = (all_future if nf == num_future else idx_rel).cpu()
                        d_future[s, layer][:, fut_idx] = pa[:, f0:f0 + nf].half().cpu()
                        d_obs[s, layer] = pa[:, :f0].sum(dim=-1).half().cpu()
                        d_act[s, layer] = pa[:, f0 + nf:].sum(dim=-1).half().cpu()
                        if d_cur is None:
                            d_cur = torch.zeros((steps, self.num_layers, pa.shape[0], f0), dtype=torch.float16)
                        d_cur[s, layer] = pa[:, :f0].half().cpu()  # per frame-0 key
                    if recv_acc is not None and s == 0 and idx_rel.numel() != num_future:
                        # step0-sparse: the probes' received attention, softmax with step 0's key bias.
                        if io_v is not None:  # step0-explore late: step 0 may compute no future token
                            recv_acc[idx_rel] += self._masked_probs(io_v[0], k_vid, fv_mask_v)[:, f0:].sum(dim=0)
                        recv_acc[idx_rel] += self._masked_probs(io_a[0], k_all, fv_mask_a)[:, f0:f0 + n_q].sum(dim=0)
                    elif recv_acc is not None and s == 0:
                        # Sinks: the step-0 attention every future key receives from the video rows (all future
                        # tokens on this full step) and the action rows, head mean, summed over rows.
                        recv_acc += self._attention_probs(io_v[0], k_vid)[:, f0:].sum(dim=0)
                        recv_acc += self._attention_probs(io_a[0], k_all)[:, f0:f0 + num_future].sum(dim=0)
                    if late is not None and s == 1:
                        # late anchor: step 1's A->V (score layer, normalized) and the attention its future tokens receive.
                        if layer == self._layers["act"]:
                            nf_ = int(k_all.shape[1]) - f0 - int(io_a[1].shape[1])
                            av_s = self._masked_probs(io_a[0], k_all, amask)[:, f0:f0 + nf_].sum(dim=0)
                            late_st["av"] = av_s / av_s.sum()
                        late_st["recv"][idx_rel] += self._masked_probs(io_v[0], k_vid, fv_mask_v)[:, f0:].sum(dim=0)
                        late_st["recv"][idx_rel] += self._masked_probs(io_a[0], k_all, amask)[:, f0:f0 + n_q].sum(dim=0)
                    if c2f is not None and s == 1 and layer == self._layers["act"]:
                        # C2F: step 1's A->V on its computed future keys (softmax with the step's key bias), normalized.
                        nf_ = int(k_all.shape[1]) - f0 - int(io_a[1].shape[1])
                        av_s = self._masked_probs(io_a[0], k_all, amask)[:, f0:f0 + nf_].sum(dim=0)
                        c2f.update(rows1=idx_rel.clone(), av1=av_s / av_s.sum())
                    if mem_on and layer == int(cfg.score_layer):
                        # step0-sparse-r2 memory: this step's A->V on its computed future keys (softmax with the step's key
                        # bias), normalized over them; every token keeps the value of its last computed step.
                        nf_ = int(k_all.shape[1]) - f0 - int(io_a[1].shape[1])
                        av_s = self._masked_probs(io_a[0], k_all, amask)[:, f0:f0 + nf_].sum(dim=0)
                        mem_av[idx_rel] = av_s / av_s.sum()
                    if mem_on and s == steps - 1:  # memory: the last step's received attention (as the fovea sinks)
                        mem_recv[idx_rel] += self._masked_probs(io_v[0], k_vid, fv_mask_v)[:, f0:].sum(dim=0)
                        mem_recv[idx_rel] += self._masked_probs(io_a[0], k_all, amask)[:, f0:f0 + n_q].sum(dim=0)
                    if fv_cal and s == 0:  # N1: the dense step's queries and keys, for the calibration once the plan is known
                        cal_qk[layer] = (io_v[0], io_a[0], io_v[1], io_a[1])
                    if fattn_acc is not None:
                        # Beside the forward pass: head- and query-mean attention on [frame-0 | kept future] keys
                        # (the plain key layout, checked in the config), summed over layers.
                        pa_m = self._attention_probs(io_a[0], k_all).mean(dim=0)
                        fattn_acc[:f0] += pa_m[:f0]
                        fattn_acc[f0 + idx_rel] += pa_m[f0:f0 + n_q]
                    if layer in capture_layers:
                        if p0_step and idx_rel.numel() != num_future:
                            # step0-sparse: the probes' columns of the full layout [frame-0 | Kf | action] (softmax with
                            # step 0's key bias), 0 on the uncomputed future tokens.
                            pm = self._masked_probs(io_a[0], k_all, fv_mask_a)
                            n_a = int(pm.shape[0])
                            probs = torch.zeros((n_a, f0 + num_future + n_a), dtype=pm.dtype, device=pm.device)
                            probs[:, :f0] = pm[:, :f0]
                            probs[:, f0 + idx_rel] = pm[:, f0:f0 + n_q]
                            probs[:, f0 + num_future:] = pm[:, f0 + n_q:]
                        elif idx_rel.numel() != num_future:
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
                    if io_v is not None:
                        xv = self._post(bv, io_v, out_v, payload_v)
                        if explore_h is not None and s == 0:  # step0-explore: the dense step-0 hidden states per block
                            explore_h.append(xv.detach())
                    if merge_step:
                        # merge: each group's residual increment of this layer onto its members' own streams (a group of
                        # one takes the output itself), then the groups re-formed from the streams for the next layer.
                        n_fix = int(idx_rel.numel())
                        out_g = xv[:, n_fix:]
                        h_new = (h_mem.float() + (out_g.float() - xg_in.float())[:, mg["gid"]]).to(h_mem.dtype)
                        h_new[:, mg["single"]] = out_g[:, mg["gid"][mg["single"]]]
                        h_mem = h_new
                        if layer + 1 < self.num_layers:
                            xv = torch.cat([xv[:, :n_fix], self._merge_rows(h_mem, mg)], dim=1)
                    xa = self._post(ba, io_a, out_a, payload_a)
                    if ex_step:
                        # EX: the action expert once more on its own stream, keys [frame-0 | action].
                        io_nf = self.mot._build_expert_attention_io(
                            expert=self.action, block=ba, x=xa_nf, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
                        )
                        out_nf = flash_attention(q=io_nf[0], k=torch.cat([kv0[layer]["k"], io_nf[1]], dim=1),
                                                 v=torch.cat([kv0[layer]["v"], io_nf[2]], dim=1), num_heads=self.num_heads)
                        xa_nf = self._post(ba, io_nf, out_nf, payload_a)
                    if dump is not None and dump.hidden:
                        if d_hidden is None:
                            d_hidden = torch.zeros((steps, self.num_layers) + tuple(xa.shape[1:]), dtype=torch.float16)
                        d_hidden[s, layer] = xa[0].half().cpu()
                if tea is not None:
                    tea_res_v = xv - xv_in
                    tea_res_a = xa - xa_in
                    tea_acc = 0.0
                    tea_consecutive = 0
                if s == 0:
                    act_probs_step0 = step_probs
                if s == score_step:
                    act_probs_sel = step_probs
            if want_share:
                share_rows.append(torch.stack(share_layers) if share_layers
                                  else torch.full((self.num_layers,), float("nan"), device=model.device))
            if want_vf0:
                row = torch.full((num_future,), float("nan"), device=model.device)
                if vf0_layers:
                    row[idx_rel] = torch.stack(vf0_layers).mean(dim=0)
                vf0_rows.append(row)
            if want_fattn:
                row = torch.full((f0 + num_future,), float("nan"), device=model.device)
                if fattn_acc is not None:
                    present = torch.cat([torch.arange(f0, device=model.device), f0 + idx_rel])
                    row[present] = fattn_acc[present] / self.num_layers
                fattn_rows.append(row)

            # ---- heads and scheduler step
            if vel_tokens is None and not idx_abs.numel():  # step0-explore late: a step 0 without future rows
                vel_tokens = torch.zeros((1, f0 + num_future, pt * ph * pw * int(latents_video.shape[1])), device=latents_video.device,
                                         dtype=latents_video.dtype)
            if idx_abs.numel() or merge_step:  # keep 0: no future token is recomputed after warm-up
                n_real = int(idx_abs.numel())
                if merge_step:
                    # merge: the real rows, then every periphery member's own stream through the head (its own velocity)
                    vel_part = self.video.head(torch.cat([xv[:, :n_real], h_mem], dim=1),
                                               torch.cat([pre_v["t"][:, idx_abs], pre_v["t"][:, mg["abs"]]], dim=1))
                else:
                    t_rows = pre_v["t"][:, idx_abs]
                    if n_syn:
                        t_rows = torch.cat([t_rows, pre_v["t"][:, fv_first]], dim=1)
                    vel_part = self.video.head(xv, t_rows)
                if vel_tokens is None:
                    vel_tokens = torch.zeros((1, f0 + num_future, vel_part.shape[-1]), device=vel_part.device, dtype=vel_part.dtype)
                vel_tokens[:, idx_abs] = vel_part[:, :n_real]  # pruned tokens keep their last velocity
                if merge_step:
                    vel_tokens[:, mg["abs"]] = vel_part[:, n_real:]
                    fv_count[mg["abs"] - f0] += 1
                else:
                    for g, members in enumerate(fv_members if n_syn else ()):
                        vel_tokens[:, members] = vel_part[:, n_real + g:n_real + g + 1]  # a region token's velocity to every member
                if fv_step:
                    fv_count[idx_rel] += 1
                if p0_step and idx_rel.numel() != num_future:
                    # step0-sparse draft (DECISIONS B.4): the probes' clean estimate interpolated over every block; the
                    # uncomputed future tokens move along (x - x0) / sigma_0 until computed. Plan b's Diff reads it.
                    sigma0_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                    if c2f is not None:  # C2F: the probes' raw x^0 (before the gain), for their Diff
                        c2f["x0_raw"] = self._clean_future(latents_video, vel_tokens, pre_v["meta"]["grid_size"], sigma0_v, (pt, ph, pw))
                    if prune.step0_draft == "grid":
                        v_draft, s0_zf = self._probe_draft(latents_video, vel_tokens, pre_v["meta"]["grid_size"], grid,
                                                           idx_rel, sigma0_v, (pt, ph, pw), gain=float(prune.step0_gain))
                    else:  # step0-explore: any computed set, drafts spread by idw / near
                        v_draft, s0_zf = self._set_draft(latents_video, vel_tokens, pre_v["meta"]["grid_size"], grid, idx_rel, sigma0_v,
                                                         (pt, ph, pw), float(prune.step0_gain), prune.step0_draft,
                                                         float(prune.step0_far_lambda))
                    rest = torch.ones(num_future, dtype=torch.bool, device=idx_rel.device)
                    rest[idx_rel] = False
                    rest_abs = torch.nonzero(rest).flatten() + f0
                    vel_tokens[:, rest_abs] = v_draft[:, rest_abs].to(vel_tokens.dtype)
                    if float(prune.step0_gain) != 1.0:  # step0-explore: the probes move along their corrected change too
                        vel_tokens[:, f0 + idx_rel] = v_draft[:, f0 + idx_rel].to(vel_tokens.dtype)
            sig_s = t_video[s].float() / float(model.infer_video_scheduler.num_train_timesteps)
            gsz = pre_v["meta"]["grid_size"]
            side_pred = None
            if late is not None and s == 0:
                # step0-explore late anchor: step 0 computes no future token; every future token stays on pure noise (x^0 = 0).
                x_tok0 = rearrange(latents_video, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw)
                vel_tokens[:, f0:] = (x_tok0[:, f0:].float() / sig_s).to(vel_tokens.dtype)
            if late is not None and s == 1:
                # late anchor: step 1 forms the imagination; every other future token is drafted from it (spread, gain), and
                # the computed ones move along their corrected change.
                v_draft, late_st["zf"] = self._set_draft(latents_video, vel_tokens, gsz, grid, idx_rel, sig_s, (pt, ph, pw),
                                                         float(prune.step0_gain), prune.step0_draft, float(prune.step0_far_lambda))
                rest = torch.ones(num_future, dtype=torch.bool, device=idx_rel.device)
                rest[idx_rel] = False
                upd_abs = torch.nonzero(rest).flatten() + f0 if float(prune.step0_gain) == 1.0 else torch.arange(num_future, device=idx_rel.device) + f0
                vel_tokens[:, upd_abs] = v_draft[:, upd_abs].to(vel_tokens.dtype)
            if side0 is not None and s == 0:
                # step0-sparse-r2 R1 / R3b: B's step 0 (the probes; every token on the first side0_m blocks) beside the dense
                # main pass; it gives B's plan inputs (plan / both) and / or step 0's outputs (draft / both).
                probes = fv.probe_index(grid).to(model.device)
                side = self._side_forward(pre_v, pre_a, kv0, probes, math.log(num_future / int(probes.numel())), int(prune.side0_m),
                                          f0, num_future, self._layers["act"])
                tmp = vel_tokens.clone()
                tmp[:, f0 + probes] = side["vel"]
                v_draft, zf_side = self._probe_draft(latents_video, tmp, gsz, grid, probes, sig_s, (pt, ph, pw),
                                                     gain=float(prune.step0_gain))
                if side0 in ("draft", "both", "video"):  # step0-explore D2: video only / action only
                    rest = torch.ones(num_future, dtype=torch.bool, device=probes.device)
                    rest[probes] = False
                    rest_abs = torch.nonzero(rest).flatten() + f0
                    vel_tokens[:, f0 + probes] = side["vel"]
                    vel_tokens[:, rest_abs] = v_draft[:, rest_abs].to(vel_tokens.dtype)
                    if float(prune.step0_gain) != 1.0:
                        vel_tokens[:, f0 + probes] = v_draft[:, f0 + probes].to(vel_tokens.dtype)
                if side0 in ("draft", "both", "action"):
                    side_pred = side["pred_action"]
                if side0 in ("plan", "both"):
                    r = side["recv"].float().clone()
                    out = torch.ones(num_future, dtype=torch.bool, device=r.device)
                    out[probes] = False
                    out[fv.PROBE_SINK] = True
                    r[out] = float("-inf")
                    side0_inputs = {"av": fv.interp_tokens(sel.av_scores(side["probs"], grid).float()[:, None], grid, probes)[:, 0],
                                    "diff": sel.diff_scores(zf_side, z_cur0, grid), "recv": side["recv"], "probes": probes,
                                    "sinks": torch.cat([torch.tensor([fv.PROBE_SINK], device=r.device), torch.topk(r, fv.N_SINKS - 1).indices])}
                diag["side0"] = {"use": side0, "m": int(prune.side0_m), "layers": side["layers"]}
            if cfg.explore and s == 0:
                # step0-explore (dense): the dense step-0 clean estimate of every future token and the current frame (Diff
                # layout), then one forward per spec beside the main pass, logged for offline analysis.
                x_tok = rearrange(latents_video, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw)[0, f0:].float()
                x0d = self._tok_to_diff(x_tok - sig_s * vel_tokens[0, f0:].float())
                zc = rearrange(latents_video[0, :, 0:pt].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)[0]
                diag["x0d_f16"], diag["zcur_f16"] = _f16_b64(x0d), _f16_b64(zc)
                lay = self._layers["act"]
                av_d = sel.av_scores(step_probs[lay], grid) if lay in step_probs else torch.zeros(num_future, device=model.device)
                plan_d = fv.make_plan("interlace_w", max(1, int(round(0.2 * num_future))), grid, av_d.float().cpu(),
                                      sel.diff_scores(x0d, zc, grid).float().cpu(), (recv_acc / self.num_layers).float().cpu()
                                      if recv_acc is not None else av_d.float().cpu())
                sig_a = t_action[0].float() / float(model.infer_action_scheduler.num_train_timesteps)
                ex_rows: dict = {}
                for spec in cfg.explore:
                    name, rspec, lspec, kv, nprox, vscale, bias = spec
                    rows_e = self._explore_rows(rspec, grid, plan_d).to(model.device)
                    rest_e = torch.ones(num_future, dtype=torch.bool, device=model.device)
                    rest_e[rows_e] = False
                    rest_idx = torch.nonzero(rest_e).flatten()
                    groups = None
                    if nprox:
                        rid = fv.grid_regions(grid).to(model.device)[rest_idx]
                        groups = [rest_idx[rid == g] for g in torch.unique(rid).tolist()]
                    rb = (math.log(num_future / int(rows_e.numel())) if bias is None and not kv and not nprox
                          and rows_e.numel() < num_future else (bias or 0.0))
                    out_e = self._explore_forward(pre_v, pre_a, kv0, rows_e, f0, num_future, lay, grid, row_bias=rb,
                                                  vscale=vscale, layers=self._explore_layers(lspec),
                                                  kv_rows=rest_idx if kv and rest_idx.numel() else None, groups=groups,
                                                  track=explore_h if cfg.explore_track else None)
                    x0r = self._tok_to_diff(x_tok[rows_e] - sig_s * out_e["vel"][0].float())
                    row = {"rows": rows_e.tolist(), "x0_f16": _f16_b64(x0r), "bias": rb, "layers": out_e["layers"],
                           "probs_layer": out_e["probs_layer"], "track": out_e["track"],
                           "pilot_f32": _f32_b64(latents_action[0].float() - sig_a * out_e["pred_action"][0].float()),
                           "e_x": float((x0r - x0d[rows_e]).double().norm() / x0d[rows_e].double().norm())}
                    if out_e["probs"] is not None:
                        row["av_f16"] = _f16_b64(sel.av_scores(out_e["probs"], grid))
                    row["recv_f16"] = _f16_b64(out_e["recv"])
                    if groups:
                        xg = torch.stack([x_tok[g].mean(dim=0) for g in groups])
                        row["proxy_x0_f16"] = _f16_b64(self._tok_to_diff(xg - sig_s * out_e["proxy_vel"][0].float()))
                        row["groups"] = [g.tolist() for g in groups]
                    ex_rows[name] = row
                diag["explore"] = ex_rows
                diag["explore_plan"] = {"sinks": plan_d.sinks.tolist(), "core": plan_d.core.tolist(),
                                        "slices": [x.tolist() for x in plan_d.slices]}
            if s in cfg.record_r2_steps or (s == 0 and cfg.record_r3_ms):
                # step0-sparse-r2 R2 / R3 (dense): side forwards on this step's latents; x^0 at the probes against the main pass.
                probes = fv.probe_index(grid).to(model.device)
                x_tok = rearrange(latents_video, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw)[0, f0 + probes].float()
                x0_main = x_tok - sig_s * vel_tokens[0, f0 + probes].float()
                runs = []
                if s in cfg.record_r2_steps:
                    allp = all_future if cfg.record_r2_all else probes
                    runs.append(("r2", s, allp, 0.0 if cfg.record_r2_all else math.log(num_future / int(probes.numel())), 0))
                if s == 0:
                    runs += [("r3", m_, probes, math.log(num_future / int(probes.numel())), m_) for m_ in cfg.record_r3_ms]
                for kind, key, rows_, b_, m_ in runs:
                    side = self._side_forward(pre_v, pre_a, kv0, rows_, b_, m_, f0, num_future, self._layers["act"])
                    pos = torch.searchsorted(side["rows"], probes)
                    v_side = side["vel"][0, pos].float()
                    x0_side = x_tok - sig_s * v_side
                    row = {"e_x": float((x0_side - x0_main).double().norm() / x0_main.double().norm()),
                           "equal": bool(torch.equal(side["vel"][0, pos], vel_tokens[0, f0 + probes]))}
                    if kind == "r2":
                        row["x0_main_f16"] = _f16_b64(x0_main)
                        r2_rows[str(key)] = row
                    else:
                        sig_a = t_action[0].float() / float(model.infer_action_scheduler.num_train_timesteps)
                        row.update(pilot_f32=_f32_b64(latents_action[0].float() - sig_a * side["pred_action"][0].float()),
                                   layers=side["layers"])
                        r3_rows[str(key)] = row
            if route_plan and s == 0:
                # R7: the uncomputed future tokens move towards the previous call's imagination, v = (x - d) / sigma_0.
                rest = torch.ones(num_future, dtype=torch.bool, device=idx_rel.device)
                rest[idx_rel] = False
                rest_abs = torch.nonzero(rest).flatten() + f0
                vel_tokens[:, rest_abs] = self._map_draft(latents_video, route_map, grid, sig_s, (pt, ph, pw))[:, rest_abs].to(vel_tokens.dtype)
            if c2f is not None and s == 1 and "av1" in c2f:  # C2F: step 1's own x^0 Diff
                zf1 = self._clean_future(latents_video, vel_tokens, gsz, sig_s, (pt, ph, pw))[idx_rel]
                c2f["df1"] = (zf1 - z_c2f[grid.future_obs_index(z_c2f.device)][idx_rel]).norm(dim=-1)
            draft_now = (draft_from is not None and fv_step and s >= draft_from and fplan.slices
                         and (route_plan or not prune.mem_route))
            if draft_now or x0_est is not None:
                zf = self._clean_future(latents_video, vel_tokens, gsz, sig_s, (pt, ph, pw))
                upd = idx_rel[:0]
                if draft_now:
                    # R5 / R7: every periphery token still to be computed after this step takes the inverse-distance mean of
                    # the x^0 of this step's computed tokens in its block as its draft, v = (x - d) / sigma_s.
                    n_sl = len(fplan.slices)
                    later = [fplan.slices[(k_ - 1) % n_sl] for k_ in range(s + 1, steps)]
                    pending = torch.unique(torch.cat(later)) if later else idx_rel[:0]
                    if pending.numel():
                        zf_new, upd = fv.idw_fill(zf, idx_rel, grid, targets=pending)
                        if upd.numel():
                            v_new = self._map_draft(latents_video, zf_new, grid, sig_s, (pt, ph, pw))
                            vel_tokens[:, f0 + upd] = v_new[:, f0 + upd].to(vel_tokens.dtype)
                            zf = zf_new
                    diag.setdefault("draft", []).append([int(s), int(pending.numel()), int(upd.numel())])
                if x0_est is not None:  # R7b: the latest clean estimate of every token
                    x0_est[idx_rel] = zf[idx_rel]
                    if upd.numel():
                        x0_est[upd] = zf[upd]
            pred_video = self.video.unpatchify(vel_tokens, pre_v["meta"]["grid_size"])
            if s == 0 and (cfg.record_step0_x0 or e9 is not None or dmod is not None):
                # step0-sparse E8: the step-0 clean estimate at the 39 probes (exact there on every arm); E9: step 0's
                # latents, to redo its update once the plan is known.
                sigma0_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                x0 = latents_video.float() - sigma0_v * pred_video.float()
                z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
                if cfg.record_step0_x0:
                    diag["step0_x0_f32"] = _f32_b64(z[1:].reshape(-1, z.shape[-1])[fv.probe_index(grid).to(z.device)])
                if e9 is not None or dmod is not None:
                    e9_prev = latents_video.clone()
            if want_motion_steps:
                # This step's clean estimate x - sigma * v (sigma_s = -sum of the remaining deltas; the
                # schedule ends at 0, so at the last step it is the final latent), frame 1 vs frame 0.
                sigma = -d_video[s:].float().sum()
                x0 = latents_video.float() - sigma * pred_video.float()
                z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
                motion_rows.append((z[1] - z[0]).norm(dim=-1))
            if s == 0 and prune is not None and not route_plan and (prune.selection in ("diff", "diff_global") or fovea_m is not None):
                # diff: x0 = x - sigma_v v of the warm-up (full) step, every future token against the pinned current
                # frame's latent at the same (h, w), over the token's full latent patch.
                sigma_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                x0 = latents_video.float() - sigma_v * pred_video.float()
                z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
                z_cur = rearrange(latents_video[0, :, 0:pt].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)[0]
                # step0-sparse plan b: the interpolated clean estimate itself (not through the draft's rounded velocity).
                diff_score = sel.diff_scores(z[1:].reshape(-1, z.shape[-1]) if s0_zf is None else s0_zf, z_cur, grid)
                diag["diff_scores_f16"] = _f16_b64(diff_score)
                diag["diff_sigma_v"] = float(sigma_v)
                if fovea_m == "fov_cluster":
                    x0_fut = z[1:].reshape(-1, z.shape[-1])  # k-means features of the periphery (step-0 clean estimate)
            pred_action = self.action.post_dit(xa, pre_a)
            if side_pred is not None:  # R1-draft / R3b: step 0's action from the side forward
                pred_action = side_pred
            if ex_step:
                # EX: v = v_s + w * (v_s - v_noF) in fp32, back to the model dtype (w = 0 gives v_s bit for bit).
                pred_nf = self.action.post_dit(xa_nf, pre_a)
                pred_action = (pred_action.float() + ex_w * (pred_action.float() - pred_nf.float())).to(pred_action.dtype)
            if want_pilot and s == 0:
                # Pilot (agc / record_pilot): the dense warm-up step's clean action estimate x - sigma_0 * v_0
                # in fp32, beside the forward pass.
                sigma0 = t_action[0].float() / float(model.infer_action_scheduler.num_train_timesteps)
                diag["pilot_f32"] = _f32_b64(latents_action[0].float() - sigma0 * pred_action[0].float())
                diag["pilot_sigma0"] = float(sigma0)
            if s0_mode == "d" and s == 0:
                # step0-sparse plan d: the pilot of the probe step (normalized), for the adapter's end-effector tube.
                sigma0_a = t_action[0].float() / float(model.infer_action_scheduler.num_train_timesteps)
                s0_pilot = (latents_action[0].float() - sigma0_a * pred_action[0].float()).cpu()
            latents_video = model.infer_video_scheduler.step(pred_video, d_video[s], latents_video)
            latents_action = model.infer_action_scheduler.step(pred_action, d_action[s], latents_action)
            latents_video[:, :, 0:1] = first.clone()
            modes.append("probe" if p0_step and idx_rel.numel() != num_future else mode)
            if dump is not None:
                d_latents.append(latents_action[0].float().cpu())

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
            if late is not None and s == 1:
                # late anchor: the plan of steps 2-9 from step 1 (sinks 182 + the 2 computed tokens receiving the most; core
                # from the spread A->V and the Diff of the spread drafts; computed tokens in the last slices).
                rows1 = idx_rel
                av_full = self._spread(torch.zeros(num_future, 1, device=rows1.device).index_copy(0, rows1, late_st["av"][:, None].float()),
                                       rows1, grid, prune.step0_draft, torch.zeros(num_future, 1, device=rows1.device), 0.0)[:, 0]
                zo = z_late[grid.future_obs_index(z_late.device)]
                df_full = (late_st["zf"] - zo).norm(dim=-1)
                r_ = (late_st["recv"] / self.num_layers).float().clone()
                out_ = torch.ones(num_future, dtype=torch.bool, device=r_.device)
                out_[rows1] = False
                out_[fv.PROBE_SINK] = True
                r_[out_] = float("-inf")
                sinks_l = torch.cat([torch.tensor([fv.PROBE_SINK], device=r_.device), torch.topk(r_, fv.N_SINKS - 1).indices])
                fplan = fv.c2f_plan(rows1.cpu(), av_full.cpu(), df_full.cpu(), sinks_l.cpu(), grid,
                                    max(1, int(round(prune.keep_ratio * num_future))), core_all=True).to(kept.device)
                kept = fplan.fixed
                diag["late"].update(core=fplan.core.tolist(), sinks=fplan.sinks.tolist(), slices=[int(x.numel()) for x in fplan.slices],
                                    truncated=int(fplan.truncated.numel()))
            if c2f is not None and s == 1 and "df1" in c2f:
                # C2F: the plan of steps 2-9 from the scores measured at steps 0 and 1 (the latest for every token).
                av_m, df_m = c2f["av"].clone(), c2f["diff"].clone()
                av_m[c2f["rows1"]], df_m[c2f["rows1"]] = c2f["av1"], c2f["df1"]
                measured = torch.unique(torch.cat([c2f["measured"], c2f["rows1"]]))
                fplan = fv.c2f_plan(measured.cpu(), av_m.cpu(), df_m.cpu(), c2f["sinks"].cpu(), grid,
                                    max(1, int(round(prune.keep_ratio * num_future)))).to(kept.device)
                kept = fplan.fixed
                diag["c2f"].update(measured=int(measured.numel()), core=fplan.core.tolist(), sinks=fplan.sinks.tolist(),
                                   slices=[int(x.numel()) for x in fplan.slices], truncated=int(fplan.truncated.numel()))
            if prune is not None and not route_plan and (s == prune.warmup_steps - 1 or (mode == "pruned" and prune.reselect_every_step)):
                k = 0 if prune.keep_ratio == 0.0 else max(1, int(round(prune.keep_ratio * num_future)))
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
                    p_src = step_probs if (mode == "pruned" and prune.reselect_every_step) else act_probs_sel
                    p_act = p_src[self._layers["act"]]
                    if prune.selection == "vv":
                        p_vid = vid_step_probs.get(self._layers["act"])
                        kept = (sel.topk_indices(sel.vv_scores(p_vid, grid), k)
                                if p_vid is not None else all_future[:k])
                    elif prune.selection == "av":
                        kept = sel.topk_indices(sel.av_scores(p_act, grid), k, exclude=frame_exclude)
                    elif prune.selection in ("diff", "diff_global"):
                        score = sel.av_scores(p_act, grid) if prune.diff_debug_av else diff_score
                        kept = (sel.topk_indices(score, k) if prune.selection == "diff_global"
                                else sel.per_view_topk(score, k, grid))
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
                            av_all = sel.av_scores(act_probs_sel[self._layers["act"]], grid)
                            extra = dropped[torch.topk(av_all[dropped], m).indices]
                        kept = torch.cat([kept, extra])
                        diag["extra_kept"] = int(m)
                kept = torch.sort(kept).values
                if fovea_m is not None and s == prune.warmup_steps - 1:
                    # Fovea: sinks / core / periphery from this step's A->V (S0's scoring layer), Diff and received
                    # attention, planned on the CPU (k-means draws from a per-call generator).
                    av_all = sel.av_scores(act_probs_sel[self._layers["act"]], grid)
                    recv = recv_acc / self.num_layers
                    gen = torch.Generator(device="cpu").manual_seed((int(seed) + 3) & 0x7FFFFFFF)
                    av_plan, diff_plan, s0_sinks = av_all, diff_score, None
                    if s0_mode is not None:
                        # step0-sparse (DECISIONS B): sinks = 182 + the 2 probes receiving the most; plan b: A->V
                        # interpolated from the probes (Diff already from the interpolated clean estimate); plan d: the
                        # frame-0 attention at the same (h, w) and the end-effector tube x (1 + it).
                        r = recv.float().clone()
                        out = torch.ones(num_future, dtype=torch.bool, device=r.device)
                        out[s0_probes] = False
                        out[fv.PROBE_SINK] = True
                        r[out] = float("-inf")
                        s0_sinks = torch.cat([torch.tensor([fv.PROBE_SINK], device=r.device), torch.topk(r, fv.N_SINKS - 1).indices])
                        s0_info.update(sinks_eq_top3=sorted(s0_sinks.tolist()) == sorted(torch.topk(recv.float(), fv.N_SINKS).indices.tolist()))
                        if s0_mode == "b" and prune.step0_draft != "grid":  # step0-explore: A->V spread like the drafts
                            av_plan = self._spread(av_all.float()[:, None], s0_probes, grid, prune.step0_draft,
                                                   torch.zeros_like(av_all.float())[:, None], 0.0)[:, 0]
                        elif s0_mode == "b":
                            av_plan = fv.interp_tokens(av_all.float()[:, None], grid, s0_probes)[:, 0]
                        else:
                            obs = act_probs_sel[self._layers["act"]][:, :f0].float().sum(dim=0)
                            av_plan = obs[grid.future_obs_index(obs.device)]
                            cells, switch, s0_info["tube"] = _step0_tube(s0_pilot)
                            tube = fv.tube_mask(cells, grid, wrist_rows=switch)
                            diff_plan = tube.to(av_plan.device).float() * (1.0 + av_plan)
                            s0_info["tube_index"] = torch.nonzero(tube).flatten().tolist()
                    plan_probes = None if s0_mode is None or prune.step0_all else s0_probes.cpu()
                    if side0_inputs is not None:  # step0-sparse-r2 R1-plan / R3b: B's plan from the side forward
                        av_plan, diff_plan, recv = side0_inputs["av"], side0_inputs["diff"], side0_inputs["recv"]
                        s0_sinks, plan_probes = side0_inputs["sinks"], side0_inputs["probes"].cpu()
                    if late is not None:
                        survey = self._explore_rows(late, grid, None)
                        lg = math.log(num_future / int(survey.numel()))
                        empty_ = survey[:0]
                        fplan = fv.Plan(fovea_m, empty_, empty_, empty_, survey, bias_real={int(j): lg for j in survey.tolist()}).to(kept.device)
                        diag["late"] = {"set": late, "n": int(survey.numel())}
                    elif c2f is not None:
                        # step0-explore C2F: step 1 computes the sinks + the neighbourhoods of the hottest probes (A->V, Diff
                        # measured at the probes themselves), every key + log(Kf / n); the plan of steps 2-9 comes after it.
                        zfo = z_c2f[grid.future_obs_index(z_c2f.device)]
                        df0 = (c2f["x0_raw"] - zfo).norm(dim=-1)
                        av0 = torch.zeros(num_future, device=av_all.device)
                        av0[s0_probes] = av_all[s0_probes] / av_all[s0_probes].sum()
                        survey = fv.c2f_survey(s0_probes.cpu(), av0.cpu(), df0.cpu(), s0_sinks.cpu(), int(prune.c2f), grid, k)
                        lg = math.log(num_future / int(survey.numel()))
                        empty_ = survey[:0]
                        fplan = fv.Plan(fovea_m, s0_sinks.cpu(), empty_, empty_, survey,
                                        bias_real={int(j): lg for j in survey.tolist()}).to(kept.device)
                        c2f.update(av=av0, diff=df0, measured=s0_probes.clone(), sinks=s0_sinks.clone())
                        diag["c2f"] = {"survey": survey.tolist()}
                    else:
                        fplan = fv.make_plan(fovea_m, k, grid, av_plan.float().cpu(), diff_plan.float().cpu(), recv.float().cpu(),
                                             None if x0_fut is None else x0_fut.cpu(), gen, off=prune.fovea_off,
                                             wzero=prune.fovea_wzero, nofar=prune.fovea_nofar,
                                             singleton=prune.fovea_singleton,
                                             sinks=None if s0_sinks is None else s0_sinks.cpu(), probes=plan_probes).to(kept.device)
                    if s0_mode is not None and fplan.slices:
                        # Runner check (DECISIONS C.3): the slices and the truncated tokens partition the periphery.
                        parts = torch.cat(fplan.slices + [fplan.truncated])
                        if parts.numel() != fplan.periphery.numel() or not torch.equal(torch.sort(parts).values, fplan.periphery):
                            raise RuntimeError("step0: the slices and the truncated tokens do not partition the periphery")
                        pset = set(s0_probes.tolist())
                        s0_info.update(probes=int(s0_probes.numel()), bias=round(s0_bias, 6),
                                       slice_probes=[sum(int(j) in pset for j in x.tolist()) for x in fplan.slices],
                                       core_probes=sum(int(j) in pset for j in fplan.core.tolist()))
                        diag["step0"] = {"mode": s0_mode, "all": bool(prune.step0_all), "nobias": bool(prune.step0_nobias),
                                         **s0_info}
                    if e9 is not None and fplan.slices:
                        # E9: the drafts of slices 1-6 (early) or 8-9 (late) from the probe interpolation of step 0's clean
                        # estimate; step 0's update redone (bitwise the same for every other token).
                        sl = fplan.slices[:6] if e9 == "early" else fplan.slices[7:]
                        toks = torch.cat(sl).to(vel_tokens.device) + f0
                        sigma0_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                        v_draft, _ = self._probe_draft(e9_prev, vel_tokens, pre_v["meta"]["grid_size"], grid,
                                                       fv.probe_index(grid).to(vel_tokens.device), sigma0_v, (pt, ph, pw))
                        vel_tokens[:, toks] = v_draft[:, toks].to(vel_tokens.dtype)
                        latents_video = model.infer_video_scheduler.step(
                            self.video.unpatchify(vel_tokens, pre_v["meta"]["grid_size"]), d_video[0], e9_prev)
                        latents_video[:, :, 0:1] = first.clone()
                        diag["e9"] = {"mode": e9, "tokens": int(toks.numel())}
                    if dmod is not None:
                        # step0-explore D1 (exploratory): every future token's step-0 draft y (the dense x^0) replaced by
                        # y + e (white / low / high frequency, or white on one region) with |e| = scale |y|, or by the current
                        # frame (prior); step 0's update redone.
                        sigma0_v = t_video[0].float() / float(model.infer_video_scheduler.num_train_timesteps)
                        gsz = pre_v["meta"]["grid_size"]
                        y = self._clean_future(e9_prev, vel_tokens, gsz, sigma0_v, (pt, ph, pw))
                        zc = rearrange(e9_prev[0, :, 0:pt].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)[0]
                        g_ = torch.Generator(device="cpu").manual_seed((int(seed) + 7) & 0x7FFFFFFF)
                        if dmod == "prior":
                            d = zc[grid.future_obs_index(zc.device)]
                        elif dmod == "atten":  # the change from the current frame scaled by draft_mod_scale
                            zo = zc[grid.future_obs_index(zc.device)]
                            d = zo + float(prune.draft_mod_scale) * (y - zo)
                        else:
                            e = torch.randn(y.shape, generator=g_).to(y.device)
                            if dmod == "low":
                                e = fv.interp_tokens(e, grid, fv.probe_index(grid, extras=False))
                            elif dmod == "high":
                                e = e - fv.block_smooth(e, grid)
                            elif dmod in ("wrist", "agent", "far", "near"):
                                j_ = torch.arange(num_future, device=y.device)
                                col = (j_ % grid.per_frame) % grid.width
                                keep = {"wrist": col >= grid.width // 2, "agent": col < grid.width // 2,
                                        "far": j_ >= grid.per_frame, "near": j_ < grid.per_frame}[dmod]
                                e = e * keep[:, None]
                            d = y + e * (float(prune.draft_mod_scale) * y.norm() / e.norm())
                        v_new = self._map_draft(e9_prev, d, grid, sigma0_v, (pt, ph, pw))
                        vel_tokens[:, f0:] = v_new[:, f0:].to(vel_tokens.dtype)
                        latents_video = model.infer_video_scheduler.step(self.video.unpatchify(vel_tokens, gsz), d_video[0], e9_prev)
                        latents_video[:, :, 0:1] = first.clone()
                        diag["draft_mod"] = {"mode": dmod, "scale": float(prune.draft_mod_scale),
                                             "rel_err": float((d - y).norm() / y.norm())}
                    kept = fplan.fixed
                    fv_members = [g + f0 for g, _ in fplan.regions]
                    if fv_members:
                        fv_first = torch.stack([m[0] for m in fv_members])
                        fv_freqs = fv.rope_freqs_at(fplan.region_pos, 2 * int(pre_v["freqs"].shape[-1]))
                    tot = float(recv.sum())
                    diag["fovea"] = {
                        "method": fovea_m, "off": bool(prune.fovea_off), "k": int(k),
                        "sinks": fplan.sinks.tolist(), "sink_share_step0": float(recv[fplan.sinks].sum()) / tot if tot > 0 else None,
                        "core": int(fplan.core.numel()), "periphery": int(fplan.periphery.numel()),
                        "regions": [int(g.numel()) for g, _ in fplan.regions], "reps": len(fplan.bias_real),
                        "slices": [int(x.numel()) for x in fplan.slices],
                        "truncated": 0 if fplan.truncated is None else int(fplan.truncated.numel()),
                        "core_index": fplan.core.tolist(), "slice_bias": [round(x, 6) for x in fplan.slice_bias],
                    }
                    if fplan.freq is not None:
                        diag["fovea"]["freq_hist"] = {int(f): int((fplan.freq == f).sum()) for f in torch.unique(fplan.freq).tolist()}
                        diag["fovea"]["freq_sum"] = int(fplan.freq.sum())
                    if fv_cal and fplan.slices:
                        cal_bias, diag["fovea"]["cal"] = self._cal_bias(fplan, cal_qk, kv0, f0, num_future)
                        cal_pos = [torch.searchsorted(fplan.real(i + 1), x) for i, x in enumerate(fplan.slices)]
                    cal_qk.clear()
                    if fv_fill and fplan.slices:
                        fills = [fv.fill_map(fplan.real(i + 1), grid) for i in range(len(fplan.slices))]
                        fill_src = [m for m, _ in fills]
                        diag["fovea"]["fill_out_of_block"] = [n for _, n in fills]
                    if fv_merge and fv_members:
                        sizes = [int(m.numel()) for m in fv_members]
                        mg = {"abs": torch.cat(fv_members), "cnt": sizes,
                              "gid": torch.cat([torch.full((n,), g, dtype=torch.long, device=kept.device) for g, n in enumerate(sizes)]),
                              "spans": [(sum(sizes[:g]), sum(sizes[:g + 1])) for g in range(len(sizes))]}
                        mg["single"] = torch.tensor([n == 1 for n in sizes], device=kept.device)[mg["gid"]]
                if cache_kv and not video_eager and s == prune.warmup_steps - 1:
                    keep_mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    keep_mask[kept] = False
                    dropped = torch.nonzero(keep_mask).flatten()
                    kv_drop = {l: (k[:, dropped], v[:, dropped]) for l, (k, v) in kv_future_full.items()}
                    kv_future_full.clear()
                    diag["cache_kv_bytes"] = int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                                     for k, v in kv_drop.values()))
                if recon is not None and s == prune.warmup_steps - 1:
                    # Layout of the dropped positions p = (frame, h, w): frame-0 source (h, w), RoPE rows,
                    # camera view (left half of the latent columns = agentview); R1's (frame, view) groups
                    # with the fallback to the other frame of the same view (or none: delta 0).
                    keep_mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    keep_mask[kept] = False
                    drop_idx = torch.nonzero(keep_mask).flatten()
                    half = int(grid.width) // 2
                    view_of = lambda idx: ((idx % f0) % int(grid.width) >= half).long()  # noqa: E731
                    group_of = lambda idx: (idx // f0) * 2 + view_of(idx)  # noqa: E731
                    src = drop_idx % f0
                    src_u, inv = torch.unique(src, return_inverse=True)
                    g_drop, g_kept = group_of(drop_idx), group_of(kept)
                    present = set(int(g) for g in g_kept.tolist())
                    groups, fb_other, fb_zero = [], 0, 0
                    for g in sorted(set(int(x) for x in g_drop.tolist())):
                        g_src = g if g in present else (g ^ 2 if (g ^ 2) in present else None)
                        if g_src is None:
                            fb_zero += 1
                            continue
                        fb_other += int(g_src != g)
                        groups.append((g_src, g_kept == g_src, g_drop == g))
                    rc = {"n": int(drop_idx.numel()), "dropped": drop_idx, "src": src, "src_u": src_u, "inv": inv,
                          "kept_src": kept % f0, "view": view_of(drop_idx), "groups": groups,
                          "freqs": self._future_freqs(pre_v, drop_idx, f0)}
                    rc["view_masks"] = [(v, rc["view"] == v) for v in (0, 1) if bool((rc["view"] == v).any())]
                    diag["recon"] = {"mode": recon, "dropped": rc["n"], "dropped_per_view": [int((rc["view"] == v).sum()) for v in (0, 1)],
                                     "groups_with_dropped": len(set(int(x) for x in g_drop.tolist())),
                                     "fallback_other_frame": fb_other if recon == "r1" else 0,
                                     "fallback_zero": fb_zero if recon == "r1" else 0,
                                     "r2_proj_tokens": int(src_u.numel()) if recon == "r2" else 0}
                if summary and s == prune.warmup_steps - 1:
                    keep_mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    keep_mask[kept] = False
                    dropped = torch.nonzero(keep_mask).flatten()
                    nk, n_act = int(kept.numel()), int(pre_a["tokens"].shape[1])
                    info: dict[str, Any] = {"grid": int(prune.summary_grid), "bias": prune.summary_bias,
                                            "dropped": int(dropped.numel()), "groups": 0}
                    if dropped.numel():
                        gid = self._summary_groups(grid, prune.summary_grid, prune.summary_views, kept.device)
                        lse_means = []
                        for layer, (k_f, v_f) in kv_future_full.items():
                            k_s, v_s, counts, member = self._summary_build(k_f, v_f, dropped, gid)
                            kv_sum[layer] = (k_s, v_s)
                            n_sum = int(k_s.shape[1])
                            heads = self.num_heads if summary_lse else 1
                            mv = torch.zeros((1, heads, nk if summary_lse else 1, f0 + n_sum + nk), device=kept.device)
                            ma = torch.zeros((1, heads, n_act if summary_lse else 1, f0 + n_sum + nk + n_act), device=kept.device)
                            if summary_lse:
                                bv = self._summary_lse_bias(q0_v[layer][:, kept], k_f, k_s, member)
                                ba = self._summary_lse_bias(q0_a[layer], k_f, k_s, member)
                                mv[0, :, :, f0:f0 + n_sum], ma[0, :, :, f0:f0 + n_sum] = bv, ba
                                lse_means.append(float(torch.cat([bv.flatten(), ba.flatten()]).mean()))
                            else:
                                mv[..., f0:f0 + n_sum] = torch.log(counts)
                                ma[..., f0:f0 + n_sum] = torch.log(counts)
                            sum_mask_v[layer] = mv.to(model.torch_dtype)
                            sum_mask_a[layer] = ma.to(model.torch_dtype)
                        info.update({"groups": n_sum, "counts": [int(c) for c in counts.tolist()],
                                     "keys_video": f0 + n_sum + nk, "keys_action": f0 + n_sum + nk + n_act,
                                     "kv_bytes": int(sum(2 * k.numel() * k.element_size() for k, _ in kv_sum.values()))})
                        if lse_means:
                            info["lse_bias_mean"] = float(np.mean(lse_means))
                    diag["summary"] = info
                    kv_future_full.clear()
                    q0_v.clear()
                    q0_a.clear()
                if steer is not None and s == prune.warmup_steps - 1:
                    steer_bias, diag["steer"] = self._steer_bias(
                        steer, steer_fut, steer_logits, kept, f0, num_future, dense_bias=dense_bias
                    )
                    steer_fut.clear()
                    steer_logits.clear()
                if steer_v is not None and s == 0:
                    steer_vbias, diag["steer_video"] = self._steer_video_bias(steer_v, steer_vfut, steer_vlogits, kept, f0)
                    steer_vfut.clear()
                    steer_vlogits.clear()
                if s == prune.warmup_steps - 1:
                    av = sel.av_scores(act_probs_sel[self._layers["act"]], grid) if act_probs_sel else None
                    diag["score_step"] = int(score_step)
                    if score_step > 0 and av is not None:
                        diag["av_scores_sel_f16"] = _f16_b64(av)
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
            if s0_probes is not None:  # step0-sparse: step 0 computed the probes only
                diag["future_token_steps"] -= num_future - int(s0_probes.numel())
            diag["future_token_steps_dense"] = steps * num_future
        if tea is not None:
            diag["teacache_rel_l1"] = [round(r, 6) for r in tea_rel]
        if noise_gen is not None:
            diag["noise_sigma"] = cfg.noise.sigma
        if full_kv:
            diag["action_full_kv"] = True
        if frame_exclude is not None:
            diag["score_frames"] = list(prune.score_frames)
        if want_share:
            shares = torch.stack(share_rows)  # [steps, layers]
            diag["future_share_f32"] = _f32_b64(shares)
            diag["future_share_step"] = [round(float(x), 6) for x in shares.mean(dim=1)]
        if cfg.record_motion:
            # Final latents are the clean prediction (the schedule ends at sigma 0). Token order
            # follows the patch embedding: (frame, row, column) of patches.
            z = rearrange(latents_video[0].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
            diag["motion_f32"] = _f32_b64((z[1:] - z[:1]).norm(dim=-1).flatten())
        if want_vf0:
            diag["video_frame0_f32"] = _f32_b64(torch.stack(vf0_rows))  # [steps, Kf]
        if want_motion_steps:
            diag["motion_steps_f32"] = _f32_b64(torch.stack(motion_rows))  # [steps, tokens of frame 1]
        if want_fattn:
            diag["future_attn_f32"] = _f32_b64(torch.stack(fattn_rows))  # [steps, frame-0 + future keys]
            diag["future_attn_shape"] = [len(fattn_rows), int(f0 + num_future)]
        diag["paths"] = {"action_eager": steer is not None, "video_eager": video_eager, "oracle_kv": oracle is not None,
                         "action_full_kv": full_kv, "video_full_kv": video_full_kv, "dense_bias": dense_bias,
                         "steer_video": steer_v is not None, "recon": recon, "action_no_future": no_future,
                         "extrapolate_w": ex_w}
        if want_rq:
            # [k, v, k of dense step 0 (V2's reuse), v of dense step 0] x step x layer x view (agentview, wrist):
            # mean cosine to the dense K / V of the same step over the dropped tokens of the view.
            diag["recon_quality_f32"] = _f32_b64(rq)
            diag["recon_quality_shape"] = list(rq.shape)
            diag["recon_checks"] = {k: {"max_abs_diff": float(v[0]), "max_abs_true": float(v[1])} for k, v in checks.items()}
        if want_recv:
            recv = recv_acc / self.num_layers
            top = torch.topk(recv, fv.N_SINKS).indices
            diag["sink_index"] = sorted(int(x) for x in top.tolist())
            diag["sink_share_step0"] = float(recv[top].sum() / recv.sum())
            diag["sink_recv_f16"] = _f16_b64(recv)
        if fplan is not None:
            diag["fovea"]["rows_per_step"] = fv_rows
            if fplan.freq is not None and fplan.slices:
                # N4: every future token computed exactly f_j times, every pruned step the same number of rows.
                ok = bool(torch.equal(fv_count, fplan.freq.to(fv_count.device))) and len(set(fv_rows)) == 1
                diag["fovea"]["ht_counts_ok"] = ok
                if not ok and len(fv_rows) == len(fplan.slices):
                    raise RuntimeError("fovea interlace_ht: a token was not computed f_j times, or the rows differ by step")
            elif fplan.slices and not reselected and c2f is None and late is None:
                # Interlace: every periphery token not cut by the quota computed exactly once on the pruned steps.
                cut = set(fplan.truncated.tolist())
                cnt = fv_count[fplan.periphery].tolist()
                once = all((c == 0) if int(j) in cut else (c == 1) for j, c in zip(fplan.periphery.tolist(), cnt))
                diag["fovea"]["interlace_once"] = once
                if not once and len(fv_rows) == len(fplan.slices):
                    raise RuntimeError("fovea interlace: a periphery token was not computed exactly once")
        if r2_rows:
            diag["r2"] = r2_rows
        if r3_rows:
            diag["r3"] = r3_rows
        if cfg.record_step0_x0 or mem_on:
            zt = rearrange(latents_video[0].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
            zt = zt[1:].reshape(num_future, -1)
            if cfg.record_step0_x0:  # step0-sparse-r2 R6: the terminal latents at the 39 probes
                diag["terminal_x0_f32"] = _f32_b64(zt[fv.probe_index(grid).to(zt.device)])
            if mem_on:
                # Memory for the next call with this key: terminal latents, the last step's computed tokens, every token's
                # last A->V, the 2 last-step tokens receiving the most besides 182.
                computed = fplan.real(steps - 1) if fplan is not None else all_future
                r = (mem_recv / self.num_layers).float().clone()
                out = torch.ones(num_future, dtype=torch.bool, device=r.device)
                out[computed] = False
                out[fv.PROBE_SINK] = True
                r[out] = float("-inf")
                sinks2 = torch.topk(r, fv.N_SINKS - 1).indices
                self._memory[mem_key] = {"terminal": zt.clone(), "computed": computed.clone(), "av": mem_av.clone(),
                                         "sinks2": sinks2}
                diag["mem_rec"] = {"key": str(mem_key), "computed": int(computed.numel()), "sinks2": sinks2.tolist(),
                                   "route": bool(route_plan)}
        # Relative compute (token passes + attention): per step [video queries, video keys, action
        # queries, action keys] at layer 0, the parameter counts and the context length.
        diag["cost"] = {**self._cost_model(), "attn_dim": cost_dim, "ctx": int(context.shape[1]), "f0": int(f0),
                        "steps": cost_steps, "oracle_pass": oracle is not None}
        if side0 == "both":  # R3b: step 0 as the side forward computed it (its per-block rows), not the dense main pass
            diag["cost"]["step0_layers"] = diag["side0"]["layers"]
        if recon is not None:  # R2: frame-0 positions through W_k / W_v per layer, per step
            diag["cost"]["recon_proj_tokens"] = cost_recon
        if ex_w is not None:  # EX: the no-future action pass per computed step
            diag["cost"]["extra_action_steps"] = cost_extra
        if dump is not None:
            arrays = {"seed": np.int64(seed), "modes": np.asarray(modes), "grid": np.asarray(diag["grid"]),
                      "kept_index": kept.cpu().numpy(), "action_latents": torch.stack(d_latents).numpy()}
            if self._layers is not None:
                arrays["layer_act"] = np.int64(self._layers["act"])
            if d_future is not None:
                arrays.update(attn_future=d_future.numpy(), attn_obs=d_obs.numpy(), attn_act=d_act.numpy(), attn_cur=d_cur.numpy())
            if d_hidden is not None:
                arrays["hidden"] = d_hidden.numpy()
            if dump.video:
                frames = model._decode_latents(latents_video)
                arrays["video"] = np.stack([np.asarray(f) for f in frames])
            _write_npz(Path(dump.path), arrays)
            diag["dump_path"] = dump.path
        if _capture_video is not None:  # future-inject: the final (clean) video latents, for F_dense
            _capture_video["latents"] = latents_video.detach().clone()
        return latents_action[0].detach().to(device="cpu", dtype=torch.float32), diag

    # ------------------------------------------------------------------ future injection
    @torch.no_grad()
    def encode_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """VAE latents [1, z, 3, h, w] of a clip [9, 3, H, W] (current frame + 8 future frames, model input range), the
        way training encodes its video clips."""
        video = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
        return self.model._encode_video_latents(video)

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> np.ndarray:
        """Decoded frames [T, H, W, 3] uint8 of video latents [1, z, t, h, w]."""
        return np.stack([np.asarray(f) for f in self.model._decode_latents(latents)])

    @torch.no_grad()
    def infer_inject(
        self,
        prompt: str,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        action_horizon: int,
        num_video_frames: int,
        seed: int,
        future_latents: torch.Tensor,
        sigma_shift: Optional[float] = None,
        rand_device: str = "cpu",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """inject_future: the two future latent frames fixed to ``future_latents`` [1, z, 2, h, w], clean (timestep 0 for
        every video token); frame 0 and its prefill as on the normal path. Only the action stream is denoised, with the
        normal path's schedule, noise (the action generator of ``seed``), text and proprio. The video tokens' inputs never
        change, so their per-layer K/V are computed once (the same numbers as resetting them after every step)."""
        model = self.model
        steps = int(self.cfg.num_inference_steps)
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
        f0 = grid.per_frame
        if tuple(future_latents.shape[2:]) != (latent_t - 1, latent_h, latent_w):
            raise ValueError(f"future latents {tuple(future_latents.shape)} do not match the grid ({latent_t - 1}, {latent_h}, {latent_w})")
        # The normal path draws the video noise from its own generator: the action noise is the same draw here.
        gen_a = torch.Generator(device=rand_device).manual_seed(int(seed))
        latents_action = torch.randn(
            (1, action_horizon, self.action.action_dim), generator=gen_a, device=rand_device, dtype=torch.float32
        ).to(device=model.device, dtype=model.torch_dtype)
        input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
        first = model._encode_input_image_latents_tensor(input_image=input_image, tiled=False)
        fuse = bool(getattr(self.video, "fuse_vae_embedding_in_latents", False))
        context, context_mask, num_text = self._encode_prompt(prompt)
        if proprio is not None:
            context, context_mask = model._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)
        t_video, _ = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=first.dtype,
            shift_override=sigma_shift if self.cfg.video_sigma_shift is None else self.cfg.video_sigma_shift,
        )
        t_action, d_action = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=model.device, dtype=latents_action.dtype, shift_override=sigma_shift
        )
        pre0 = self.video.pre_dit(
            x=first, timestep=t_video[0].unsqueeze(0).to(dtype=first.dtype, device=model.device),
            context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse,
        )
        kv0, _ = self._prefill_frame0(pre0, num_text, set())
        latents_video = torch.cat([first, future_latents.to(device=model.device, dtype=first.dtype)], dim=2)
        pre_v = self.video.pre_dit(
            x=latents_video, timestep=torch.zeros(1, dtype=first.dtype, device=model.device), context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse,
        )
        xv = pre_v["tokens"][:, f0:]
        freqs_v, tmod_v = pre_v["freqs"][f0:], pre_v["t_mod"][:, f0:]
        payload_v = {"context": pre_v["context"], "mask": pre_v["context_mask"][:, f0:]}
        kv_fut = []
        for layer in range(self.num_layers):
            bv = self.video.blocks[layer]
            io_v = self.mot._build_expert_attention_io(expert=self.video, block=bv, x=xv, freqs=freqs_v, t_mod=tmod_v)
            out_v = flash_attention(q=io_v[0], k=torch.cat([kv0[layer]["k"], io_v[1]], dim=1),
                                    v=torch.cat([kv0[layer]["v"], io_v[2]], dim=1), num_heads=self.num_heads)
            kv_fut.append((io_v[1], io_v[2]))
            xv = self._post(bv, io_v, out_v, payload_v)
        n_fut = int(xv.shape[1])
        cost_steps = [[n_fut, f0 + n_fut, 0, 0]]  # the one video pass of the clean future
        for s in range(steps):
            ts_a = t_action[s].unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
            pre_a = self.action.pre_dit(action_tokens=latents_action, timestep=ts_a, context=context, context_mask=context_mask)
            xa = pre_a["tokens"]
            payload_a = {"context": pre_a["context"], "mask": pre_a["context_mask"]}
            for layer in range(self.num_layers):
                ba = self.action.blocks[layer]
                io_a = self.mot._build_expert_attention_io(
                    expert=self.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
                )
                k_all = torch.cat([kv0[layer]["k"], kv_fut[layer][0], io_a[1]], dim=1)
                v_all = torch.cat([kv0[layer]["v"], kv_fut[layer][1], io_a[2]], dim=1)
                out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=self.num_heads)
                if layer == 0:
                    cost_steps.append([0, 0, int(io_a[0].shape[1]), int(k_all.shape[1])])
                xa = self._post(ba, io_a, out_a, payload_a)
            pred_action = self.action.post_dit(xa, pre_a)
            latents_action = model.infer_action_scheduler.step(pred_action, d_action[s], latents_action)
        diag = {"path": "inject_future", "seed": int(seed), "steps": steps, "grid": [int(grid.frames), int(grid.height), int(grid.width)],
                "cost": {**self._cost_model(), "attn_dim": int(io_a[0].shape[-1]), "ctx": int(context.shape[1]), "f0": int(f0),
                         "steps": cost_steps, "oracle_pass": False}}
        return latents_action[0].detach().to(device="cpu", dtype=torch.float32), diag


def _write_npz(path: Path, arrays: dict) -> None:
    """Atomic ``.npz`` write: a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


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
