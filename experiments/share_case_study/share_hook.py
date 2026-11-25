"""share_case_study (2026-11-25): a hook on the action rows' joint attention of ``fastwam.casestudy.joint_runner``.

It replaces the module-level name ``flash_attention`` of a module (``joint_runner``) by a wrapper, in this process only. The
wrapper recognises the action rows' calls (q has ``action_tokens`` rows, keys are ``[frame-0 (f0) | future (nf) | action]``),
counts them to know the denoising step and the layer, and

* with no modification asked (or outside the asked steps) calls the original function unchanged, so the output is the same
  bit for bit;
* otherwise adds an additive mask on the future columns ``[f0, f0 + nf)`` and passes it to the same SDPA: the logit bias
  ``c`` (every step or only the last one), the HT weight ``log(n_future / nf)`` when the action rows see fewer future keys than
  ``n_future``, and, for the injected-sparse arm, ``-inf`` on the future columns outside the kept set (chosen at step 0 on
  ``select_layer`` by the A->V score: action-row probabilities averaged over heads, summed over action rows);
* measures, beside the pass, the future share m of the action rows (fp32 softmax of q k^T / sqrt(d) + the same mask): per
  [layer, head] on the last step and head-averaged per [step, layer].

Video rows, frame-0 prefill and every other attention call never match (their q length is not ``action_tokens`` or their keys
hold no action block) and go through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Condition:
    c: float = 0.0  # logit bias on the action rows' future columns
    scope: str = "all"  # "all": every step; "last": only the last step
    force_mask: bool = False  # pass a mask even when it is all zero (the bias path at c = 0)
    ht: bool = False  # + log(n_future / nf) when nf < n_future
    sparse_k: Optional[int] = None  # injected-sparse: keep k future keys from step 1 on
    select_layer: Optional[int] = None  # layer whose step-0 A->V score picks the kept set


@dataclass
class Record:
    calls: int = 0
    m_last: Optional[torch.Tensor] = None  # [layers, heads]
    m_steps: Optional[torch.Tensor] = None  # [steps, layers]
    nf_steps: list = field(default_factory=list)  # visible future keys per step (layer 0)
    kept: Optional[torch.Tensor] = None
    masked_calls: int = 0


class ShareHook:
    def __init__(self, module, *, num_layers: int, num_heads: int, steps: int, action_tokens: int, f0: int, n_future: int):
        self.module = module
        self.orig = module.flash_attention
        self.L, self.H, self.steps = int(num_layers), int(num_heads), int(steps)
        self.A, self.f0, self.N = int(action_tokens), int(f0), int(n_future)
        self.cond: Optional[Condition] = None
        self.rec: Optional[Record] = None
        self.installed = False

    # ------------------------------------------------------------------ control
    def install(self) -> None:
        if not self.installed:
            self.module.flash_attention = self._wrapped
            self.installed = True

    def uninstall(self) -> None:
        if self.installed:
            self.module.flash_attention = self.orig
            self.installed = False

    def begin(self, cond: Condition) -> None:
        if cond.scope not in ("all", "last"):
            raise ValueError(f"scope {cond.scope!r}")
        if cond.sparse_k is not None and cond.select_layer is None:
            raise ValueError("sparse_k needs select_layer")
        self.cond = cond
        self.rec = Record(m_last=torch.zeros(self.L, self.H), m_steps=torch.zeros(self.steps, self.L))

    def end(self) -> Record:
        rec = self.rec
        if rec is None:
            raise RuntimeError("end() without begin()")
        if rec.calls != self.steps * self.L:
            raise RuntimeError(f"share hook: {rec.calls} action-row calls, expected {self.steps} x {self.L}")
        self.cond, self.rec = None, None
        return rec

    # ------------------------------------------------------------------ wrapper
    def _is_action(self, q: torch.Tensor, k: torch.Tensor) -> bool:
        return int(q.shape[1]) == self.A and int(k.shape[1]) >= self.f0 + self.A + 1

    def _logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Per-head q k^T / sqrt(d) for batch 1 -> [H, Sq, Sk] (fp32)."""
        b, sq, hd = q.shape
        d = hd // self.H
        qh = q[0].float().view(sq, self.H, d).transpose(0, 1)
        kh = k[0].float().view(k.shape[1], self.H, d).transpose(0, 1)
        return (qh @ kh.transpose(-1, -2)) * (d ** -0.5)

    def _wrapped(self, q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        cond, rec = self.cond, self.rec
        if cond is None or not self._is_action(q, k):
            return self.orig(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask, compatibility_mode=compatibility_mode)
        if int(num_heads) != self.H:
            raise RuntimeError(f"share hook: {num_heads} heads, expected {self.H}")
        idx = rec.calls
        rec.calls += 1
        step, layer = divmod(idx, self.L)
        sk = int(k.shape[1])
        nf = sk - self.f0 - self.A
        if layer == 0:
            rec.nf_steps.append(nf)
        fut = slice(self.f0, self.f0 + nf)
        bias = torch.zeros(sk, dtype=torch.float32, device=q.device)
        use = cond.force_mask
        if cond.c != 0.0 and (cond.scope == "all" or step == self.steps - 1):
            bias[fut] += float(cond.c)
            use = True
        if cond.ht and nf < self.N:
            bias[fut] += math.log(self.N / nf)
            use = True
        if cond.sparse_k is not None:
            if nf != self.N:
                raise RuntimeError("injected-sparse expects all future keys from the runner")
            if step == 0 and layer == cond.select_layer:
                p = torch.softmax(self._logits(q, k), dim=-1).mean(dim=0)  # [A, Sk]
                av = p[:, fut].sum(dim=0)
                rec.kept = torch.topk(av, int(cond.sparse_k)).indices.sort().values
            if step >= 1:
                if rec.kept is None:
                    raise RuntimeError("injected-sparse: no kept set at step 1")
                drop = torch.ones(nf, dtype=torch.bool, device=q.device)
                drop[rec.kept.to(q.device)] = False
                bias[self.f0:self.f0 + nf][drop] = float("-inf")
                if cond.ht:
                    bias[self.f0:self.f0 + nf][~drop] += math.log(self.N / int(cond.sparse_k))
                use = True
        mask = None
        if use:
            mask = bias.to(q.dtype).view(1, 1, 1, sk)
            if ctx_mask is not None:
                mask = mask + ctx_mask
            rec.masked_calls += 1
        out = self.orig(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=mask if use else ctx_mask,
                        compatibility_mode=compatibility_mode)
        # measurement, beside the pass
        lg = self._logits(q, k)
        m_add = mask if use else ctx_mask
        if m_add is not None:
            lg = lg + m_add[0].float()
        share = torch.softmax(lg, dim=-1)[..., fut].sum(dim=-1).mean(dim=-1)  # [H]
        rec.m_steps[step, layer] = float(share.mean())
        if step == self.steps - 1:
            rec.m_last[layer] = share.float().cpu()
        return out


def bf16(x: float) -> float:
    """The value a bf16 mask actually adds."""
    return float(torch.tensor(float(x), dtype=torch.bfloat16).float())
