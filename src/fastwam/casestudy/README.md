# `fastwam.casestudy`: sparse-acceleration case study (FastWAM-Joint)

Inference arms of the CVPR'27 Sparse-rethinking case study on the third-party BadWAM LIBERO Joint checkpoint
(`LIQIIIII/badwam-libero-joint-wam@086a861`, not an official release). The adapter is `evaluation/action_eval/infer.py`; one-off
tools are in `scripts/casestudy/`. FastWAM-IDM (official Optional-IDM) has its own runner on branch `optional-idm`
(`idm_runner.py`). Module and option names stay as they are: commits, arm files and reports refer to them.

## Modules

| Module | Contents |
|---|---|
| `config.py` | `CaseStudyConfig` / `PruneConfig` and `parse_config` (strict: unknown keys and invalid combinations are errors). |
| `joint_runner.py` | `JointCaseStudyRunner`: the denoising loop with frame-0 K/V and prompt reuse, every arm, the diagnostics. |
| `selection.py` | Token grid, A->V / V->V / CF / Diff scores, top-k selection, the CF phase gate. |
| `fovea.py` | fovea / fovea-w / n1-4 plans: sinks, core, periphery; Rep, Fov-grid / Fov-cluster, Interlace (M4), the weighted M4w / M4w-far, and N1-N4 (calibrated weight, fresh fill map, merge groups, HT frequencies and schedule). |
| `hybrid.py` | component / along-cross hybrid execution (dense chunk with one component of a partner chunk). |
| `along.py` | along-cross geometry: a chunk endpoint projected on the dense path (pure numpy). |
| `agc.py` | AGC round: pilot + automatic gain of the output chunk. |

## Arms by round (`casestudy` / `prune` keys)

| Round | Keys |
|---|---|
| Keep grid, Steps 1-5 | `method: dense / prune / teacache / noise`, `prune.selection av / vv / cf / random`, `keep_ratio`, `warmup_steps`, `gate`, `full_steps`, `extra_keep_ratio`, `reselect_every_step`, `score_step`, `score_frames`, `kept_index` |
| Amplitude steer | `prune.steer`, `steer_video`, `dense_bias`, `record_future_share`, `record_motion` |
| b2-context / v2-round2 | `action_full_kv`, `video_full_kv` (V2: step-0 K/V of the dropped tokens), `oracle_kv`, `video_eager`, `summary_grid / summary_views / summary_bias`, `record_motion_steps`, `record_video_frame0` |
| r-recon | `recon r0 / r1 / r2`, `keep_ratio 0` (V2-k0), `video_sigma_shift`, `record_recon_quality` |
| agc | `agc`, `record_pilot`, `action_no_future`, `extrapolate_w` |
| future-inject | `JointCaseStudyRunner.infer_inject`, `encode_clip`, `decode_video` |
| diff-select | `selection diff / diff_global`, `diff_debug_av`, `record_future_attn`, `dump` |
| fovea | `fovea sink / rep / fov_grid / fov_cluster / interlace`, `fovea_off`, `record_sinks` |
| fovea-w | `fovea interlace_w / interlace_wfar`, `fovea_wzero`, `fovea_nofar`, `video_freeze` |
| n1-4 | `fovea interlace_cal / interlace_fill / merge / interlace_ht`, `fovea_singleton` |
| share-sparsity | `steer.const` |
| step0-sparse | `prune.step0 b / d`, `step0_nobias`, `step0_all`, `e9`, `record_step0_x0`; adapter `step0_tube` |
| step0-sparse-r2 | `prune.side0_m / side0_use`, `draft_from`, `mem / mem_route / mem_direct / reselect_step`; dense `record_r2_steps / record_r2_all / record_r3_ms`; adapter `_memory_key` |
| step0-explore | `explore` / `explore_track` (dense), `draft_mod`, `step0_gain`, `c2f`, `step0_rows`, `step0_draft` (grid / idw / near), `step0_far_lambda`, `late`, `late_extra` (the late anchor) |
| late-anchor-confirm | `late_step` (0 = X-early: the late set computed on step 0), `late_extra` on plain interlace_w (`fovea.add_last_extra`, M4w + step-9 extra reads); IDM-L on the `optional-idm` branch (`idm_runner.py`, `prune.late`) |
| late-anchor-c4 | `late_oracle` (attribution: a dense step 0 beside the late anchor), `record_drafts`, far rows `f2w` / `f2a` / `f2s` and unions `a+b`, `step0_draft: nearres`, `step0_far_lambda_wrist`, `late_plan_far_av` / `late_plan_far_diff` / `late_plan_core` (the fix L84-meas = `late_plan_core: measured`); IDM (`optional-idm`) has the `late_plan_*` switches too |
| shrinkage-mechanism | `draft_mod` shift / real (adapter `_real_future`), `record_drafts` on dense / M4w / step0 anchors, `record_kval` (o - c_bar sketch), `mech` (`_mech_forward`: explicit-attention video forward with path split, attention categories, sampling-noise injection, path patching, self-key; tuned lens fit / apply), `fovea.roll_blocks`; adapter option `shrink` (real future drafts, decoded pixel / RAFT flow / gripper metrics) |
| region-attrib | `infer_inject(_attn_capture)` (the action rows' attention on every future key, side computation), `fovea.token_latent_mask` / `token_norms`; adapter option `region_attrib` (planned calls: F_real(1), static future, token-masked conditions with static / half-speed / fade sources, per-token latent change) |
| step-marginal | `prune.step_delta` / `step_delta_seed` on the late anchor (step 1's set or step s's slice: n more tokens, stratified random, or |n| fewer, lowest plan A->V), `fovea.stratified_pick` / `lowest_av` / `apply_step_delta`, `Plan.drop` |
| role-select | `prune.role_select` (rs / notip / wdiff / aav / nofar), `role_tips`, `role_seed`: the late anchor's plan of steps 2-9 by region roles (`fovea.role_plan`, `spread_order`, `role_strata`), `Plan.step_bias` |
| cam-split (branch `cam-split`) | `prune.cs_order` / `cs_n` / `cs_k` / `cs_bias` / `cs_gain_w` / `cs_gain_a` / `cs_seed`: the late anchor over steps 1-2 by camera (per-camera drafts, context rows lend keys only, plan of steps 3-9 from per-step normalized scores); `fovea.cs_rows` / `cs_bias` / `step_normalized` / `camera_tokens`; dense records its step-2 estimate |

Tests: `tests/test_casestudy_*.py` (CPU; run on the server, not locally).
