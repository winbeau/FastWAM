# FastWAM fork 工作约束

适用本仓库全部目录。Fork: `winbeau/FastWAM`；常用评测分支目标：`main`（default），不使用长期 action-eval 分支；分支切换/整合由父任务处理。本轮仅文档计划落盘，实施未开始，后续由用户逐步实施。先读 [action-eval 文档](docs/action-eval/README.md)、[来源](docs/action-eval/PROVENANCE.md) 与 [阻塞项](docs/action-eval/BLOCKERS.md)，再处理任务。

## 不可绕过的边界

- 本地仅文本阅读、静态审计、文档维护；不安装依赖，不运行 Python、测试、模型或模拟器。安装与运行只能在获授权服务器进行。本次文档任务不负责部署。
- 服务器只允许物理 **GPU 5**；显式 `CUDA_VISIBLE_DEVICES=5`。进程内通常为逻辑 `cuda:0`，不能将两者混淆。不能使用默认八卡配置，不能占用其他 GPU。
- **禁止 CI**：不新增、触发或启用 CI 工作流。
- 最大限度原样复用 `pyproject.toml` 与 `uv.lock`，**不得修改这两个文件**，不得重新生成 lock。任何确需新增的服务器依赖必须精确 pin（包 `==版本`，源码依赖固定完整 commit，记录 hash/来源）；禁止浮动升级。新增依赖与安装方案需独立审批/记录，不在本轮实施。
- FastWAM 基线必须是原生 **Joint**：`FastWAMJoint` + LIBERO Joint 对应权重与 stats。禁止用 uncond、IDM、student、RoboTwin Joint 或其他模型冒充；禁止截断维度、替换 head、训练/微调/蒸馏补齐缺失权重。
- 当前官方 LIBERO Joint 权重未找到；服务器已有 RoboTwin Joint 不匹配（用户提供事实，本地未复核服务器）。保持 `BLOCKED`，不得伪造动作、成功率、速度或可运行结论。
- 不创建未经服务器实测的 adapter；本轮只记录原生接口。既有 student/策略服务/部署代码不等于本任务许可使用的 Joint adapter。
- 经用户授权的仓库整理可以提交并 push 文档到 main；禁止强推或重置既有历史。保留上游 README 正文，仅在开头追加清楚标识的 fork 入口。

## 本地到服务器的统一流程

所有项目统一：**本地 clone / 编辑 → 本地 git add、commit、push 到 main → 算力服务器 git pull --ff-only → uv sync --locked → 经批准的训练或评测**。禁止在服务器临时改业务代码或用复制源码绕过 Git；pull 前确认工作树，不丢弃旧改动，不修改运行中环境。权重、数据、缓存不入库。若缺锁文件，仅在服务器隔离目录生成，取回本地审核并 commit/push，服务器再 pull 已提交版本后运行。本轮只落盘计划，不执行服务器部署或训练。

## 证据要求

每条结论或未来运行记录包含：**时间 / commit / 状态 / 命令 / 退出码 / artifact / 限制**。工具只读用 `read(path, offset, limit)` 表达命令，退出码写 `N/A（文件工具无 shell 退出码）`；未运行命令写 `NOT_RUN`、退出码 `N/A`。不得把静态审计标为运行 PASS；用户陈述写 `USER_REPORTED`；失败保留实际退出码和日志。

服务器未来验证应记录：实际解释器与环境、GPU物理/逻辑映射、代码及 LIBERO commit、checkpoint/stats 来源与 SHA-256、完整 resolved config、seed、steps、输入输出 contract、完整日志与退出码。权重来源及键完整性未验证时，即使加载函数返回也不得通过。
