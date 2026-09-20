# FastWAM Joint / LIBERO：fork 入口与只读接口审计

**状态：第三方 Joint 独立接入已获授权，当前按用户要求暂时搁置。** 本目录属于 [winbeau/FastWAM](https://github.com/winbeau/FastWAM)，维护分支为 `main`。用户于 2026-09-20 接受核验后的 BadWAM LIBERO Joint，作为独立接入测试；下载返回 403 `GatedRepo`，用户的访问申请仍 pending，尚未加载或验证 adapter。见 [当前准入记录](BADWAM-JOINT-ADMISSION-20260920.md)。下方是先前完成的原生接口静态审计，不是新的运行通过声明；RoboTwin、uncond、IDM、student 仍不可冒充 Joint。

- [工作约束](../../AGENTS.md)
- [证据与来源](PROVENANCE.md)
- [阻塞及解除标准](BLOCKERS.md)
- [保留的上游 README](../../README.md)

本页所有接口结论的证据：`PROVENANCE.md` 的 E03–E07（2026-09-18 UTC，基线 commit `6adf8c267e9195e17febf1b7ce72f6d2c89bf729`）。行号指该基线；下述为文本审计，未执行 Hydra composition、Python 或测试。

## 1. 原生入口与 Joint 身份

| 环节 | 准确源码入口 | 行为 |
|---|---|---|
| 批量调度 | `experiments/libero/run_libero_manager.py::main` (124–159)、`run_evaluation` (61–121) | Hydra `sim_libero.yaml`，生成任务列表、保存 `manager_config.yaml`，调用原生 shell |
| worker 启动 | `experiments/libero/run_libero_parallel_test.sh::run_libero_eval`，内嵌 `launch_task_on_pane` (322–350) | tmux worker 执行 `experiments/libero/eval_libero_single.py`；非本轮运行命令 |
| 单进程入口 | `experiments/libero/eval_libero_single.py::eval_single_process` (678–786) | instantiate model → load checkpoint → load stats → instantiate processor → suite/init states → rollout → JSON |
| 配置选择 | `configs/task/libero_joint_2cam224_1e-4.yaml` | `/data: libero_2cam`、`/model: fastwam_joint`；**必须显式选此 task**，`sim_libero.yaml` 默认是 uncond |
| 工厂 | `src/fastwam/runtime.py::create_fastwam_joint` (161–243) | 返回 `FastWAMJoint.from_wan22_pretrained(...)` |
| 模型 | `src/fastwam/models/wan22/fastwam_joint.py::FastWAMJoint` | 禁止 `action_conditioned=true`；action attention 覆盖全部 video latent token；并非 uncond/IDM/student |
| 动作调用 | `experiments/libero/eval_libero_single.py::_predict_action_chunk` (359–429) → `FastWAMJoint.infer_action` (96–236) | 默认不显示未来视频，但仍初始化未来视频噪声并逐步联合去噪 video/action，只省略视频解码返回 |
| 可视化分支 | 同一 `_predict_action_chunk` → `FastWAMJoint.infer_joint` (52–93) → 父类 `FastWAM.infer_joint` | `visualize_future_video=true` 时调用；Joint 强制 `test_action_with_infer_action=False`，不是额外拿 uncond 动作替换 |

`sim_libero.yaml` 设置 `load_text_encoder=true`、`skip_dit_load_from_pretrain=true`、`action_dit_pretrained_path=null`。这不是免权重运行许可，仍须 Wan VAE/T5/tokenizer 等资源及真正匹配的完整 Joint checkpoint。

### 加载的可靠性边界

`eval_libero_single.py::_load_model_checkpoint` (117–120) 直接调用继承的 `src/fastwam/models/wan22/fastwam.py::FastWAM.load_checkpoint` (1100–1119)。其 `mot` 使用 `strict=False`，接受 legacy `dit` 仅加载 video expert，缺 `proprio_encoder` 只警告。**加载成功不证明 Joint 完整性或来源正确**。不得利用这些宽松分支“修复”权重缺失；后续需外部核对发布来源、完整键/shape、配置和 SHA-256。

## 2. 输入 contract：沿用原生处理

入口 `_predict_action_chunk(obs, task_description, model, processor, cfg, *, action_horizon, input_w, input_h, model_device)` 返回 `(action, imgs, predicted_future_frames)`，默认分别为 NumPy `[32,7]`、两相机图字典、`None`。

1. `experiments/libero/libero_utils.py::get_libero_env` (19–39)：`OffScreenRenderEnv`，分辨率 256×256，`task.language` 为任务文本，使用 task BDDL，调用 `env.seed(cfg.seed)`。
2. `libero_utils.py::get_libero_image` (45–57)：取 `agentview_image`、`robot0_eye_in_hand_image`，**两轴反转（180°旋转）**，返回 `image`、`wrist_image`。不能只做垂直翻转。
3. `eval_libero_single.py::_obs_to_model_input` (184–241)、`_center_crop_resize` (155–164)：每相机 PIL BILINEAR 等比 resize + center crop 至224×224，主相机在左、wrist在右水平拼接；与 `data.train.video_size=[224,448]` 校验。输出 `[1,3,224,448]`，转模型 dtype/device，像素 `x*(2/255)-1`。它直接实现推理图像路径，**不是**简单调用 processor 的训练 transform。
4. `_extract_sim_state` (244–256)：位置3 + `quat2axisangle(robot0_eef_quat)` 3 + gripper qpos2，共8维 float32。`libero_utils.py::quat2axisangle` (156–180) 接受 `(x,y,z,w)`，输出弧度 axis-angle。
5. `_normalize_proprio` (167–181)：封装 `state.default` 为 `[1,8]` → processor `action_state_transform` → normalizer.forward；当前 LIBERO transforms=null，尺寸不变。Joint内部转模型 dtype/device。
6. prompt 精确格式见 `src/fastwam/datasets/lerobot/robot_video_dataset.py:24`：`A video recorded from a robot's point of view executing the following instruction: {task}`。

`FastWAMJoint.infer_action` 支持图像 `[3,H,W]` 或 `[1,3,H,W]`，仅batch1且3通道，H/W需为16倍数；`num_video_frames % 4 == 1`。proprio为 `[D]` 或 `[1,D]`。prompt 与 `(context, context_mask)` 互斥；预编码时二者都必须给出，context `[B,L,D]`、mask `[B,L]`（也允许省略batch维）；原生LIBERO路径传 prompt，不传预编码context。

## 3. stats / normalize / 动作输出

- `eval_libero_single.py::_resolve_dataset_stats_path` (89–114)：先检查 `EVALUATION.dataset_stats_path`，再检查 ckpt 的前四级父目录中的 `dataset_stats.json`。**即使指定的文件不存在，也可能回退到其他文件**；后续服务器必须预先验证显式路径、来源、hash并记录实际resolved路径。
- `src/fastwam/datasets/lerobot/utils/normalizer.py::load_dataset_stats_from_json` (175–216) 将数值列表还原为 float32 tensor。
- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py::FastWAMProcessor.set_normalizer_from_stats` (105–112) 构建 `LinearNormalizer`。
- `configs/data/libero_2cam.yaml`：`use_stepwise_action_norm=false`、`norm_default_mode=min/max`、exceptions=null。`normalizer.py::LinearNormalizer.__init__` (22–77) 从 `stats.action.default.global_*` 与 `stats.state.default.global_*` 读取；至少需各自匹配7/8维的 `global_min/global_max`，不能借用RoboTwin的统计或随便用同形状统计。
- `SingleFieldLinearNormalizer` (106–151)：通常 `scale=2/(max-min)`、`offset=-1-scale*min`；范围小于 `1e-4` 的维度有专门处理。forward 为 `x*scale+offset` 后 clamp 到 `[-5,5]`，**并非强行clip到[-1,1]**；backward 为 `(x-offset)/scale`，不clip。
- `FastWAMJoint.infer_action` 返回 `{"action": Tensor[T,7]}`，CPU float32、已detach，**仍是归一化动作**，不含视频。
- `_denormalize_action` (259–275) 转 `[1,T,7]`，只做 `normalizers['action']['default'].backward`，转 NumPy；没有额外逆 action transform。默认 transforms=null，因此后续adapter不能随意添加变换。
- `_predict_action_chunk` 对反归一化后最后维执行 `2*g-1`，再 `invert_gripper_action` 取负，默认 `binarize_gripper=true` 再取 `np.sign`；即执行值为 `sign(1-2*g)`（零保留零），不要二次翻转。前6维是配置声明的delta EEF动作，最后维不是delta。

## 4. seed、steps 与原生 rollout 参数

以下是配置文本推导值，不是运行测量：

| 参数 | 值/语义 | 来源 |
|---|---|---|
| seed | 42 | `configs/train.yaml:29` |
| 全局 RNG | seed + global_rank，用于 random/NumPy/torch/CUDA；函数要求 `0 < seed < uint32 max` | `src/fastwam/utils/pytorch_utils.py::set_global_seed` (17–34) |
| 模拟器 RNG | `env.seed(cfg.seed)`，不按trial自动加seed | `get_libero_env` |
| 推理 RNG | 每次chunk都传同一 `cfg.seed`，video/action各新建一个同seed Generator，默认rand_device=cpu | `_predict_action_chunk:402`、`FastWAMJoint.infer_action:149–162` |
| denoising steps | **10**：EVALUATION引用 `eval_num_inference_steps`；函数签名默认20不是当前配置值 | `train.yaml:24`、`sim_libero.yaml:32` |
| sigma shift | EVALUATION=null，video/action scheduler各自 infer_shift=5.0 | `fastwam_joint.yaml:47–55` |
| CFG | text_cfg_scale=1.0、negative_prompt=""；Joint `infer_action` 虽接收这两个参数，但实现没有负prompt/CFG分支，不能宣称调参有效 | `sim_libero.yaml:34–35`、`fastwam_joint.py:169–232` |
| dtype | bf16 | `train.yaml:28`、`_mixed_precision_to_model_dtype` |
| chunk horizon | 33-1=32，EVALUATION.action_horizon=null | `eval_single_process:710–714` |
| video frames | (33-1)//4+1=9 | `_get_num_video_frames`、`libero_2cam.yaml:28–30` |
| replan | 执行chunk前10步，再重新观察预测；ensembler默认false | `sim_libero.yaml:25–27`、`run_single_episode:490–522` |
| warmup | 30个dummy步骤，动作 `[0,0,0,0,0,0,-1]` | `sim_libero.yaml:24`、`libero_utils.py:41–43` |
| episode limit | spatial/object/goal=400，libero_10/libero_90=700；外加warmup30 | `_get_max_steps`、`run_single_episode` |
| trials | 每task默认50，按trial索引使用benchmark初始状态；不足时入口尝试重复扩展 | `run_single_task:610–623`、`eval_single_process:734–740` |
| suites | manager默认10/goal/spatial/object；单worker默认spatial/task0 | `sim_libero.yaml` |

`run_single_episode` 以环境 `done` 作为success，返回 `(bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr)`。`run_single_task` 统计成功/失败trial索引并保存rollout。`eval_single_process` 写 `<output_dir>/<suite>/gpu<gpu_id>_task<task_id>_results.json`。这些只是未来原生产物路径，本次未产生结果/视频。

## 5. 后续 adapter 的接口边界

1. 先满足 [当前准入记录](BADWAM-JOINT-ADMISSION-20260920.md) 的访问、权重、stats、环境和 GPU 门槛，再由获授权服务器执行原生单 worker 基准，记录完整证据。
2. 保留以上原生 `obs → image/proprio → Joint → denormalize/gripper → chunk`，不能通过换模型、截断输出、虚构stats达到“接口兼容”。
3. 比对相同观测、prompt、seed、steps、stats的原生/adapter动作；同时检查后处理后的7维动作，不只比模型tensor。未经服务器实测不创建或发布adapter。
4. GPU 按 [当前工作约束](../../AGENTS.md) 选择并显式绑定。manager shell 会为 worker 重设 `CUDA_VISIBLE_DEVICES`，`MULTIRUN.num_gpus=1` 本身不能锁定物理卡。进程内使用逻辑 cuda:0，`gpu_id` 主要用于结果标记。
5. manager默认8卡、每卡2task，且tmux执行 `source ~/.bashrc` 后调用裸 `python`；有解释器漂移风险，还会删除同名 `libero_test_v3` session。现有结果文件可能被scheduler先于退出状态当作完成。未来需独立新输出目录、核验真实解释器/进程退出码/结果一致性，不把存在文件等同通过。本轮不更改或执行这些脚本。
6. 安装、测试、模型/模拟器运行只能在授权服务器，禁止 CI。最大限度复用原始 `pyproject.toml` / `uv.lock`，二者保持原样；新依赖必须精确 pin。上游 README 的训练、浮动 pip 升级、多卡命令不是本次执行许可。
