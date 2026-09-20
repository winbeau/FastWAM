# FastWAM fork 工作约束

适用本仓库全部目录。Fork: `winbeau/FastWAM`；常用评测分支为 `main`（default），不使用长期 action-eval 分支。用户于 2026-09-20 授权：核验 BadWAM 第三方 LIBERO Joint 权重后，做独立的 FastWAM 评测接入测试；三个模型 fork 的真实评测链路打通即可完成接入目标。先读 [当前准入记录](docs/action-eval/BADWAM-JOINT-ADMISSION-20260920.md)、[原生接口](docs/action-eval/README.md) 与 [来源](docs/action-eval/PROVENANCE.md)。历史“仅计划／暂停”文字不再限制本次已授权的工作。

**最新状态：用户要求暂时搁置 FastWAM。** 用户已申请 Hugging Face 访问，但仍 pending，并说“这个先放着吧”。因此暂停下载、部署、模型加载及适配器实施，不反复探测访问或寻找替代权重。保留已有核验记录与测试授权，待用户恢复这部分工作。其他模型的现有评测不受本段暂停影响。

## 不可绕过的边界

- 本地仅阅读、编辑和静态审计；不安装依赖，不运行 Python、测试、模型或模拟器。安装与运行只能在获授权服务器进行。
- H200-target-server 上可以使用空闲或低利用率的卡；只剩一张可用卡时不占用，改为 review 或等待。GPU 7 是本轮 EGL 专用渲染卡，禁止启动 CUDA。先检查 action-eval 当前 lane 和资源记录，显式声明共享；不向任何他人进程发信号。按 UUID 绑定，进程内通常为逻辑 `cuda:0`。这些用户授权取代旧 GPU5-only 约束。
- **禁止 CI**：不新增、触发或启用 CI 工作流。
- 最大限度原样复用 `pyproject.toml` 与 `uv.lock`，**不得修改这两个文件**，不得重新生成 lock。任何确需新增的服务器依赖必须精确 pin（包 `==版本`，源码依赖固定完整 commit，记录 hash/来源）；禁止浮动升级。新增依赖与安装方案需独立审批/记录，不在本轮实施。
- FastWAM 基线必须是原生 **Joint**：`FastWAMJoint` + LIBERO Joint 对应权重与 stats。禁止用 uncond、IDM、student、RoboTwin Joint 或其他模型冒充；禁止截断维度、替换 head、训练/微调/蒸馏补齐缺失权重。
- 原官方 LIBERO Joint 基线仍未取得正确权重；服务器已有 RoboTwin Joint 不可替代。用户已另行接受 `LIQIIIII/badwam-libero-joint-wam`，必须标为第三方训练的独立接入测试，不得冒充官方基线。
- 第三方 revision 为 `086a86162cbb9e17808d01fc5c39b6302f07384b`。发布方给出的预期文件哈希不等于本地验证；必须核验实际 checkpoint SHA-256、完整 Joint/action/proprio 键和 shape、配套 stats 来源与 7/8 维语义。当前 Hugging Face 凭据返回 403 `GatedRepo`，尚未下载或加载；不得绕过访问控制。
- 按用户要求先核验资产，再验证原生与 adapter 的固定输入动作及真实闭环。未经服务器实测不得声称 adapter 或链路已打通。既有 student/策略服务/部署代码不等于本任务的 Joint adapter。写或审查 adapter 前读 action-eval 的 `skills/action-eval-adapter/SKILL.md`。
- 经用户授权的仓库整理可以提交并 push 文档到 main；禁止强推或重置既有历史。保留上游 README 正文，仅在开头追加清楚标识的 fork 入口。

## 本地到服务器的统一流程

所有项目统一：**本地 clone / 编辑 → 本地 git add、commit、push 到 main → 算力服务器 git pull --ff-only → uv sync --locked → 已授权评测**。禁止在服务器临时改业务代码或用复制源码绕过 Git；pull 前确认工作树，不丢弃旧改动，不修改运行中环境。权重、数据、缓存不入库。若缺锁文件，仅在服务器隔离目录生成，取回本地审核并 commit/push，服务器再 pull 已提交版本后运行。本轮不训练。

## 证据要求

每条结论或未来运行记录包含：**时间 / commit / 状态 / 命令 / 退出码 / artifact / 限制**。工具只读用 `read(path, offset, limit)` 表达命令，退出码写 `N/A（文件工具无 shell 退出码）`；未运行命令写 `NOT_RUN`、退出码 `N/A`。不得把静态审计标为运行 PASS；用户陈述写 `USER_REPORTED`；失败保留实际退出码和日志。

服务器未来验证应记录：实际解释器与环境、GPU物理/逻辑映射、代码及 LIBERO commit、checkpoint/stats 来源与 SHA-256、完整 resolved config、seed、steps、输入输出 contract、完整日志与退出码。权重来源及键完整性未验证时，即使加载函数返回也不得通过。
