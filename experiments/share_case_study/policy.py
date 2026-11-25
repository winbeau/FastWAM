"""share_case_study (2026-11-25): action-eval entrypoint ``experiments/share_case_study/policy.py:create_policy``.

``FastWAMJointPolicy`` (evaluation/action_eval/infer.py, loaded by path, unchanged) with one extra option ``share_case``:

    share_case: {frames_dir, calls: [0], seeds: [0, 1, 2, 3, 4], keep_k: 39,
                 sparse: <S0 casestudy config>}

The executed arm (dense) runs as usual. On each listed call it runs, through ``share_hook.ShareHook`` (a wrapper of
``joint_runner.flash_attention`` installed only for these probes), the conditions of DECISIONS.md B for every seed offset and
logs them in ``diagnostics.share_case``: experiment 1 (normal inference: dense, logit bias c on every step / the last step,
S0 keep 20 % with and without the HT weight) and experiment 2 (``infer_inject`` with the clip of the current frame and frames
round(4 i s), i = 1..8, of ``<frames_dir>/<episode>.npz``: bias c, injected-sparse with and without HT).
"""

from __future__ import annotations

import base64
import importlib.util
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("fastwam_action_eval_infer_sc", REPO / "evaluation" / "action_eval" / "infer.py")
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)

import share_hook as sh  # noqa: E402

C_GRID_1 = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
S_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
C_GRID_2 = (-1.0, 0.0, 1.0)
C_EXTRA_2 = (-3.0, -2.0, -0.5, 0.5, 2.0, 3.0)
S_EXTRA_2 = (0.0, 1.0, 2.0)


def _b64(x) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes()).decode()


class ShareCasePolicy(_infer.FastWAMJointPolicy):
    def __init__(self, config: dict, device: str):
        options = dict(config)
        sc = dict(options.pop("share_case"))
        unknown = sorted(set(sc) - {"frames_dir", "calls", "seeds", "keep_k", "sparse"})
        if unknown or not sc.get("frames_dir") or not sc.get("sparse"):
            raise ValueError(f"share_case needs frames_dir and sparse (unknown keys: {unknown})")
        super().__init__(options, device)
        from fastwam.casestudy import joint_runner as jr
        from fastwam.casestudy import make_runner, parse_config

        self._jr = jr
        self.sc = {"frames_dir": Path(str(sc["frames_dir"])), "calls": [int(c) for c in sc.get("calls") or [0]],
                   "seeds": [int(j) for j in sc.get("seeds") or [0, 1, 2, 3, 4]], "keep_k": int(sc.get("keep_k") or 39)}
        self.sc_sparse = make_runner(self.model, parse_config(sc["sparse"]), prompt_store=self.prompt_store)
        self._sc_frames = None
        self._hook = None

    def reset(self, episode: dict) -> None:
        super().reset(episode)
        self.sc_sparse.reset_episode()
        path = self.sc["frames_dir"] / f"{re.sub(r'-r[0-9]+.*$', '', str(episode.get('episode_id', '')))}.npz"
        self._sc_frames = None
        if path.is_file():
            with np.load(path) as z:
                self._sc_frames = {k: z[k] for k in ("agentview", "wrist", "eef")}
        self._sc_path = path

    def predict(self, observation: dict):
        call_index = self.predict_calls
        pred = super().predict(observation)
        if call_index in self.sc["calls"]:
            pred.diagnostics["share_case"] = self._share_case(observation, call_index, pred.diagnostics)
        return pred

    # ------------------------------------------------------------------ probes
    def _share_case(self, observation: dict, call_index: int, diag: dict) -> dict:
        torch = self._torch
        if self._sc_frames is None:
            raise FileNotFoundError(f"share_case: no rendered frames {self._sc_path}")
        started = time.perf_counter()
        images = observation["images"]
        state = np.asarray(observation.get("state"), dtype=np.float32)
        rgb = self._io.compose_model_image(images["agentview"], images["wrist"], processor=self.processor,
                                           concatenation=self.concatenation, width=self.input_w, height=self.input_h)
        image = self._io.image_to_model_tensor(rgb, device=self.device, dtype=self.model.torch_dtype)
        proprio = self._io.normalize_proprio(state, self.processor)
        prompt = self._prompt_template.format(task=observation["instruction"])
        runner = self.runner
        g = diag["grid"]
        f0 = int(g[1]) * int(g[2])
        n_future = (int(g[0]) - 1) * f0
        steps = int(runner.cfg.num_inference_steps)
        k = self.sc["keep_k"]
        c_ht = math.log(n_future / k)
        layer = int(runner._layers["act"])
        fr = self._sc_frames
        t = 10 * call_index
        out: dict = {"f0": f0, "n_future": n_future, "steps": steps, "layers": runner.num_layers, "heads": runner.num_heads,
                     "action_tokens": self.action_horizon, "select_layer": layer, "keep_k": k, "c_ht": c_ht, "c_ht_bf16": sh.bf16(c_ht),
                     "frame_equal": bool(np.array_equal(fr["agentview"][t], images["agentview"]) and np.array_equal(fr["wrist"][t], images["wrist"])),
                     "frames": int(len(fr["agentview"])), "eef": fr["eef"][t:t + 21].astype(np.float64).tolist()}

        def model_frame(i: int):
            x = self._io.compose_model_image(fr["agentview"][i], fr["wrist"][i], processor=self.processor,
                                             concatenation=self.concatenation, width=self.input_w, height=self.input_h)
            return self._io.image_to_model_tensor(x, device=self.device, dtype=self.model.torch_dtype)

        fut: dict[float, object] = {}
        out["s_valid"] = {}
        for s in S_GRID:
            idx = [t + int(round(4 * i * s)) for i in range(1, 9)]
            ok = max(idx) < len(fr["agentview"])
            out["s_valid"][f"{s:g}"] = bool(ok)
            if ok:
                clip = torch.cat([image] + [model_frame(i) for i in idx], dim=0)
                fut[s] = runner.encode_clip(clip)[:, :, 1:]
        common = dict(prompt=prompt, input_image=image, proprio=proprio, action_horizon=self.action_horizon,
                      num_video_frames=self.num_video_frames, sigma_shift=self.sigma_shift, rand_device=self.rand_device)
        hook = sh.ShareHook(self._jr, num_layers=runner.num_layers, num_heads=runner.num_heads, steps=steps,
                            action_tokens=self.action_horizon, f0=f0, n_future=n_future)
        self.sc_sparse._layers = dict(runner._layers)
        entries: dict = {}

        def log(name: str, meta: dict, cond: sh.Condition, call):
            hook.begin(cond)
            try:
                a, d = call()
                rec = hook.end()
            finally:
                hook.cond, hook.rec = None, None
            entries[name] = {**meta, **self._chunk_logs(a), "m_last_f32": _b64(rec.m_last.numpy()),
                             "m_steps": [[round(float(x), 6) for x in row] for row in rec.m_steps.tolist()],
                             "nf_steps": rec.nf_steps, "masked_calls": rec.masked_calls,
                             "kept": None if rec.kept is None else [int(i) for i in rec.kept.tolist()],
                             "kept_index": d.get("kept_index"), "c_eff": sh.bf16(cond.c) if cond.c else 0.0}

        hook.install()
        try:
            for j in self.sc["seeds"]:
                seed = self._call_seed(self._episode_key, call_index, j + self._repeat_offset)
                normal = lambda r=runner: r.infer_action(seed=seed, **common)  # noqa: E731
                log(f"dense@{j}", {"exp": 1, "path": "normal", "c": 0.0, "scope": "off"}, sh.Condition(), normal)
                for c in C_GRID_1 + (c_ht,):
                    tag = "HT" if c == c_ht else f"{c:g}"
                    log(f"e1_c{tag}_all@{j}", {"exp": 1, "path": "normal", "c": c, "scope": "all"},
                        sh.Condition(c=c, scope="all", force_mask=True), normal)
                    if c != 0.0:
                        log(f"e1_c{tag}_last@{j}", {"exp": 1, "path": "normal", "c": c, "scope": "last"},
                            sh.Condition(c=c, scope="last"), normal)
                sparse = lambda: self.sc_sparse.infer_action(seed=seed, **common)  # noqa: E731
                log(f"s0@{j}", {"exp": 1, "path": "normal", "sparse": True, "ht": False}, sh.Condition(), sparse)
                log(f"s0ht@{j}", {"exp": 1, "path": "normal", "sparse": True, "ht": True}, sh.Condition(ht=True), sparse)
                for s, z in fut.items():
                    inj = lambda z=z: runner.infer_inject(seed=seed, future_latents=z, **common)  # noqa: E731
                    cs = C_GRID_2 + (C_EXTRA_2 if s in S_EXTRA_2 else ())
                    for c in cs:
                        log(f"e2_s{s:g}_c{c:g}@{j}", {"exp": 2, "path": "inject", "s": s, "c": c, "scope": "all"},
                            sh.Condition(c=c, scope="all"), inj)
                    log(f"e2_s{s:g}_sp@{j}", {"exp": 2, "path": "inject", "s": s, "c": 0.0, "sparse": True, "ht": False},
                        sh.Condition(sparse_k=k, select_layer=layer), inj)
                    log(f"e2_s{s:g}_spht@{j}", {"exp": 2, "path": "inject", "s": s, "c": 0.0, "sparse": True, "ht": True},
                        sh.Condition(sparse_k=k, select_layer=layer, ht=True), inj)
        finally:
            hook.uninstall()
        out["entries"] = entries
        out["executed_equal_dense0"] = entries[f"dense@{self.sc['seeds'][0]}"]["raw_actions_f32"] == diag["raw_actions_f32"] \
            if self.sc["seeds"][0] == 0 else None
        out["seconds"] = round(time.perf_counter() - started, 2)
        return out


def create_policy(config: dict, device: str):
    return ShareCasePolicy(config, device)
