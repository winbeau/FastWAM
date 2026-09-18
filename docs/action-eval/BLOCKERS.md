# 阻塞项与后续计划（未实施）

**当前只完成文档计划与静态审计。用户稍后自行逐步实施；不启动部署、安装、测试或adapter实现。**

证据统一见 [PROVENANCE](PROVENANCE.md)：UTC 2026-09-18、本地基线 `6adf8c267e9195e17febf1b7ce72f6d2c89bf729`。以下静态结论对应E03–E07，用户提供的服务器/权重现状对应E08；未运行的项目对应E09，退出码N/A，无运行artifact。

## P0：LIBERO 原生 Joint 权重缺失

- 状态：**BLOCKED / USER_REPORTED**。官方LIBERO Joint权重未找到；本轮未在线检索或独立验证服务器。上游README只列举uncond发布物，不能作为Joint权重证明。
- 不可替代：LIBERO uncond、IDM、student、RoboTwin Joint。禁止训练/微调/蒸馏、随机初始化缺失部分、截断维度、改head或伪造结果。
- 解除标准：取得可验证的官方LIBERO Joint发布文件/授权来源，记录URL、不可变revision、SHA-256、训练配置与完整键shape；确认对应 `FastWAMJoint`、LIBERO 2cam224、action7/proprio8。官方Joint身份不能仅靠文件名或load成功推断。
- 本轮artifact：无checkpoint；仅源码与文档。不能报告成功率、吞吐、延迟或“已可运行”。

## P0：服务器RoboTwin Joint不匹配

- 状态：服务器已有该权重是用户陈述；配置不兼容为STATIC_AUDIT。
- `configs/data/robotwin.yaml`：3相机、action14/proprio14、z-score；LIBERO为2相机、action7/proprio8、min/max。视觉拼接和任务分布也不同。
- `FastWAM.load_checkpoint` 的 `mot strict=False`、legacy dit回退和缺proprio警告不可作为转换工具。禁止用“能加载/能输出”冒充正确模型。
- 解除标准：使用真正的LIBERO Joint checkpoint，不是修改RoboTwin权重。

## P0：匹配stats与加载完整性未验证

- 状态：**BLOCKED**。未取得/校验与Joint训练相符的LIBERO stats。
- 必须给显式stats路径，并预先确认文件存在、实际resolved路径、SHA-256、`action.default.global_min/max` 7维和 `state.default.global_min/max` 8维及合法范围；同维度不等于来源匹配。
- `_resolve_dataset_stats_path` 会在显式文件缺失时继续寻找checkpoint父目录，存在静默选错风险。
- 原生checkpoint loader宽松，未来服务器须核对键完整性和shape；拒绝缺action/proprio等关键权重的“部分成功”。本轮不改loader、不写验证脚本。

## P1：服务器环境、GPU5与真实运行均未核验

- 状态：**NOT_RUN**。只允许服务器安装/运行，物理GPU5；禁止本地Python/测试，禁止CI。
- 保持 `pyproject.toml` / `uv.lock` 原样，尽量原样复用；新依赖必须精确pin，源码依赖固定完整commit。LIBERO/模拟器依赖版本、资源和Wan VAE/T5/tokenizer仍待用户后续核验。上游建议MuJoCo3.3.2不是本轮已安装证据。
- manager默认8GPU/每GPU2任务不得原样运行。GPU锁定要求显式 `CUDA_VISIBLE_DEVICES=5`；仅设置 `num_gpus=1` 不代表物理5。进程内逻辑设备通常cuda:0。
- tmux worker `source ~/.bashrc` 后使用裸python，解释器可能与启动manager不同；脚本会删除同名session。复用output_dir还可能让旧结果文件被识别为完成。后续由用户决定如何隔离验证，本轮不修改脚本或运行调度。
- 解除标准：服务器实际解释器/精确依赖、环境源码commit、GPU映射、checkpoint/stats、resolved配置、原生单worker日志/退出码/输出均有真实证据；通过smoke不等于完成完整评测。

## P1：adapter尚未实现、未验证

- 状态：**NOT_IMPLEMENTED / NOT_RUN**。本轮不创建任何adapter。
- 原生链路、准确函数路径和输入输出已在 [README](README.md) 记录，供后续使用。
- 解除顺序（仅计划）：用户确认P0解除 → 服务器GPU5验证原生链路 → 用户另行授权adapter工作 → 对固定观测/seed/steps/stats核对原生与adapter输出及后处理 → 留存真实artifact，再讨论评测。
- 不允许跳过原生基准，也不允许在阻塞状态下创建“预计能用”的adapter并宣称完成。

## 文档任务完成不等于实验完成

已完成范围仅：README fork入口、AGENTS约束、原生Joint静态接口审计、来源台账、阻塞与后续计划。未实施范围：部署、环境变更、权重下载、Python/测试、模型/模拟器运行、训练、CI、adapter、结果生成、commit/push。后续由用户逐步实施，所有数值结果继续保持未知。
