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
the token and key counts for a relative-compute estimate.
``frame0_cache=false`` (dense only) calls the model's native
``infer_action`` unchanged, as the reference for equivalence checks. ``cfg.dump`` writes
one call's attention, action hidden states and decoded video to an ``.npz``.
"""

from __future__ import annotations

import base64
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate

from .config import CaseStudyConfig
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
            b = raw.clamp(max=steer.clamp)
            bias[layer] = b
            lg = logits[layer]
            cols = torch.cat([
                torch.arange(f0, device=lg.device), kept + f0,
                torch.arange(f0 + num_future, lg.shape[-1], device=lg.device),
            ])
            share = self._steered_probs(lg[..., cols], slice(f0, f0 + nk), b, 1.0)[..., f0:f0 + nk].sum(dim=-1)
            for key, value in (("b", b), ("err", (share - m_f).abs()), ("clamped", raw > steer.clamp),
                               ("m_f", m_f), ("m_fk", m_fk)):
                rows[key].append(value)
            if dense_bias and steer.lam == -1.0:
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
            self._cost = {"video_block_params": pv, "video_ctx_kv_params": pv_ctx,
                          "action_block_params": pa, "action_ctx_kv_params": pa_ctx, "layers": self.num_layers}
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
        _capture_kv: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.cfg
        model = self.model
        steps = int(cfg.num_inference_steps)
        # X2 (prune.oracle_kv): a dense pass of the same call and seed first, keeping every
        # step's future K/V per layer for the action rows of the pruned pass below.
        oracle: Optional[dict] = None
        if cfg.prune is not None and cfg.prune.oracle_kv:
            oracle = {}
            dense_cfg = replace(cfg, method="dense", prune=None, record_attention=False, record_future_share=False,
                                record_motion=False, record_video_frame0=False, record_motion_steps=False, dump=None)
            JointCaseStudyRunner(model, dense_cfg, prompt_store=self.prompt_store).infer_action(
                prompt, input_image, proprio, action_horizon, num_video_frames, seed,
                sigma_shift=sigma_shift, rand_device=rand_device, _capture_kv=oracle,
            )
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
        pt, ph, pw = (int(x) for x in patch)
        cost_steps: list[list[int]] = []
        cost_dim = 0
        dump = cfg.dump
        d_future = d_obs = d_act = d_hidden = None
        d_latents: list[torch.Tensor] = []

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

            idx_rel = kept if mode == "pruned" else all_future
            idx_abs = idx_rel + f0
            xv = pre_v["tokens"][:, idx_abs]
            xa = pre_a["tokens"]
            xv_in, xa_in = xv, xa

            share_layers: list[torch.Tensor] = []
            vf0_layers: list[torch.Tensor] = []
            if mode == "skip":
                xv = xv_in + tea_res_v
                xa = xa_in + tea_res_a
                tea_consecutive += 1
            else:
                capture_layers = act_layers if (s == 0 and capture_first_step) else set()
                if mode == "pruned" and prune.reselect_every_step:
                    capture_layers = {self._layers["act"]}
                if want_scores and score_step > 0 and s == score_step:
                    # P4: score on a later full step; the layer was fixed at step 0.
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
                    if _capture_kv is not None and s > 0:
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
                    n_q = int(idx_rel.numel())
                    vf0 = None
                    if video_eager and mode == "pruned":
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
                    if want_vf0:
                        vf0_layers.append(vf0.mean(dim=0))  # [queries], head mean
                    if full_kv and k_fut is not None:
                        k_all = torch.cat([kv0[layer]["k"], k_fut, io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_fut, io_a[2]], dim=1)
                        kept_cols = idx_rel + f0
                    elif full_kv and k_drop is not None:
                        k_all = torch.cat([kv0[layer]["k"], k_drop, io_v[1], io_a[1]], dim=1)
                        v_all = torch.cat([kv0[layer]["v"], v_drop, io_v[2], io_a[2]], dim=1)
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
                    else:
                        k_all = torch.cat([k_vid, io_a[1]], dim=1)
                        v_all = torch.cat([v_vid, io_a[2]], dim=1)
                        kept_cols = slice(f0, f0 + n_q)
                    if layer == 0:
                        cost_steps.append([n_q, n_vk, int(io_a[0].shape[1]), int(k_all.shape[1])])
                        cost_dim = int(io_v[0].shape[-1])
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

            # ---- heads and scheduler step
            vel_part = self.video.head(xv, pre_v["t"][:, idx_abs])
            if vel_tokens is None:
                vel_tokens = torch.zeros((1, f0 + num_future, vel_part.shape[-1]), device=vel_part.device, dtype=vel_part.dtype)
            vel_tokens[:, idx_abs] = vel_part  # pruned tokens keep their last velocity
            pred_video = self.video.unpatchify(vel_tokens, pre_v["meta"]["grid_size"])
            if want_motion_steps:
                # This step's clean estimate x - sigma * v (sigma_s = -sum of the remaining deltas; the
                # schedule ends at 0, so at the last step it is the final latent), frame 1 vs frame 0.
                sigma = -d_video[s:].float().sum()
                x0 = latents_video.float() - sigma * pred_video.float()
                z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
                motion_rows.append((z[1] - z[0]).norm(dim=-1))
            pred_action = self.action.post_dit(xa, pre_a)
            latents_video = model.infer_video_scheduler.step(pred_video, d_video[s], latents_video)
            latents_action = model.infer_action_scheduler.step(pred_action, d_action[s], latents_action)
            latents_video[:, :, 0:1] = first.clone()
            modes.append(mode)
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
                    p_src = step_probs if (mode == "pruned" and prune.reselect_every_step) else act_probs_sel
                    p_act = p_src[self._layers["act"]]
                    if prune.selection == "vv":
                        p_vid = vid_step_probs.get(self._layers["act"])
                        kept = (sel.topk_indices(sel.vv_scores(p_vid, grid), k)
                                if p_vid is not None else all_future[:k])
                    elif prune.selection == "av":
                        kept = sel.topk_indices(sel.av_scores(p_act, grid), k, exclude=frame_exclude)
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
                if cache_kv and not video_eager and s == prune.warmup_steps - 1:
                    keep_mask = torch.ones(num_future, dtype=torch.bool, device=kept.device)
                    keep_mask[kept] = False
                    dropped = torch.nonzero(keep_mask).flatten()
                    kv_drop = {l: (k[:, dropped], v[:, dropped]) for l, (k, v) in kv_future_full.items()}
                    kv_future_full.clear()
                    diag["cache_kv_bytes"] = int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                                                     for k, v in kv_drop.values()))
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
        diag["paths"] = {"action_eager": steer is not None, "video_eager": video_eager, "oracle_kv": oracle is not None,
                         "action_full_kv": full_kv, "video_full_kv": video_full_kv, "dense_bias": dense_bias,
                         "steer_video": steer_v is not None}
        # Relative compute (token passes + attention): per step [video queries, video keys, action
        # queries, action keys] at layer 0, the parameter counts and the context length.
        diag["cost"] = {**self._cost_model(), "attn_dim": cost_dim, "ctx": int(context.shape[1]), "f0": int(f0),
                        "steps": cost_steps, "oracle_pass": oracle is not None}
        if dump is not None:
            arrays = {"seed": np.int64(seed), "modes": np.asarray(modes), "grid": np.asarray(diag["grid"]),
                      "kept_index": kept.cpu().numpy(), "action_latents": torch.stack(d_latents).numpy()}
            if self._layers is not None:
                arrays["layer_act"] = np.int64(self._layers["act"])
            if d_future is not None:
                arrays.update(attn_future=d_future.numpy(), attn_obs=d_obs.numpy(), attn_act=d_act.numpy())
            if d_hidden is not None:
                arrays["hidden"] = d_hidden.numpy()
            if dump.video:
                frames = model._decode_latents(latents_video)
                arrays["video"] = np.stack([np.asarray(f) for f in frames])
            _write_npz(Path(dump.path), arrays)
            diag["dump_path"] = dump.path
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
