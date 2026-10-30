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
| `fovea.py` | fovea / fovea-w plans: sinks, core, periphery; Rep, Fov-grid / Fov-cluster, Interlace (M4) and the weighted M4w / M4w-far. |
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

Tests: `tests/test_casestudy_*.py` (CPU; run on the server, not locally).
