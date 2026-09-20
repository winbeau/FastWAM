# BadWAM LIBERO Joint 独立接入核验

本记录说明 FastWAM 第三方 Joint 接入测试的授权、已查明的资产身份和当前阻塞。用户于 2026-09-20 明确接受“核验后做独立的接入测试”。完成目标是 DreamWAM、FastWAM、DreamZero 三个 fork 的真实评测链路打通；各套件 500 集全部完成不再是接入目标的前提。

**当前状态：PAUSED_BY_USER — 访问申请 pending，用户要求先搁置。** 认证探测为 403 `GatedRepo`；用户随后告知“申请了，但是还在pending，这个先放着吧”。申请状态来自用户陈述，未独立观察到批准。暂不继续下载、部署、模型加载或 adapter 实施。尚未下载 checkpoint/stats、加载模型、创建适配器或执行 FastWAM episode；没有本模型的成功率或速度结论。以后恢复的结果将标为 BadWAM 第三方训练的 FastWAM Joint，不能写成原作者官方 Joint 复现。

## 已识别的发布物

来源是 [LIQIIIII/badwam-libero-joint-wam](https://huggingface.co/LIQIIIII/badwam-libero-joint-wam)。发布元数据描述其采用 `libero_joint_2cam224_1e-4`，step 21,700、两相机 224 输入，提供 `model.pt` 与 `dataset_stats.json`。这些是发布方声明，尚未通过实际权重验证。

| 项目 | 已读取的发布值 |
|---|---|
| 不可变 revision | `086a86162cbb9e17808d01fc5c39b6302f07384b` |
| `model.pt` 大小 | 12,041,735,545 bytes |
| `model.pt` 预期 LFS SHA-256 | `7cd0af01f4a51838510265685fa533836c6c9c29e0c37e3f7235d14ac676c928` |
| `dataset_stats.json` 大小 | 40,959 bytes |
| 访问类型 | `gated: manual` |

上述权重哈希来自发布元数据，**不是已下载文件的实测 SHA-256**。stats 的内容、维度、数值和来源匹配仍未核验。

公开元数据通过 `curl -L --max-time 20 --fail --silent --show-error 'https://huggingface.co/api/models/LIQIIIII/badwam-libero-joint-wam?blobs=true'` 获取，退出 0。该请求只读取文本，不安装或执行本地模型。记录保存于评测服务器的 `outputs/fork-readiness-20260920/badwam-public-metadata.json`，SHA-256 `fc9ec36dd7189b2d036a1fa80837374d4856c2ddb0ac24bfb92bb144a5dcdf8a`。

## 访问检查：FAILED

对固定 revision 的 `dataset_stats.json` 发出认证 HEAD 请求，使用服务器既有 Hugging Face 凭据，结果为 **HTTP 403，`x-error-code: GatedRepo`**。响应明确说明该身份不在已授权列表。curl 自身退出 0 只表示 HTTP 交互完成，不能记为下载成功。

服务器直连公开 API 先前超时；认证探测使用开发机可用网络，凭据仅经 SSH 管道传给 `curl --config -`，请求目标是官方 `huggingface.co`。未打印、保存或索取 token 值。没有提交访问申请、接受条件或下载权重/stats。已请用户开通服务器账号访问，或提供已获授权的文件路径。

脱敏证据为 `outputs/fork-readiness-20260920/badwam-access-audit.json`，SHA-256 `2ada0b5b0f33b077e8d6055d7862fdf0e819b1a1fdca53f12866f4ef581615f5`。其 `recorded_at` 提供 UTC 时间，`access_probe` 记录固定 URL、HTTP 方法、响应、进程退出和限制。

## 现有服务器环境：只读核验

FastWAM 服务器工作树在 `8672e69080d6da8f8d34ee7b609441014f6a7259` 且干净；通过 `git rev-parse HEAD`、`git status --porcelain` 和既有解释器的 `importlib.metadata.version` 读取，均退出 0。只读检查没有导入模型、使用 GPU 或更改环境。

| 项目 | 实测值 |
|---|---|
| Python | 3.10.20 |
| torch | 2.7.1+cu128 |
| transformers | 4.49.0 |
| hydra-core | 1.3.2 |
| numpy | 1.26.4 |
| huggingface-hub | 0.29.2 |
| pyproject SHA-256 | `65a125112b8aaf6c2264652a4d067a9038d5b38dd3df5b2519c9d3bbb67a458f` |
| uv.lock SHA-256 | `e3f8dbfa6b256c6dba437bf77a8c0c847eccf987aec5c4198bd651d4c74a87dc` |

这些 manifests 与维护 fork 的文件不同；不能直接将现有环境同步到本地 main，也不能把版本清单称作 Joint 模型已运行。实施需使用隔离 checkout/venv，按 Git 流程部署，保留依赖来源。

## 访问解除后的核验顺序

1. 固定上述 revision 取得 checkpoint/stats，计算实际文件 SHA-256；任何不一致都停止。读取训练与模型元数据，核对 Joint、LIBERO、两相机、action 7 / proprio 8 及配套统计。
2. 检查全部 action/video/proprio 权重键及 shape。原生 `load_checkpoint` 对 `mot` 使用 `strict=False`，且允许 legacy `dit` 或缺 proprio；必须额外阻止这些不完整加载被误记为成功。
3. 显式验证 stats 路径、字段、维度和数值，拒绝原生路径函数的意外父目录回退。核对 min/max 归一化、图像朝向、prompt、denoising 与 gripper 后处理。
4. 在隔离服务器环境验证原生固定输入，再开发薄 adapter 并比较相同 observation、seed、steps 的完整动作。沿用原生预处理和后处理，避免重复翻图或翻 gripper。
5. 用独立标注的配置执行真实 LIBERO 闭环，保留输入/动作、终态、错误、来源与报告。小规模接入测试标为 pilot，不借用完整基线的成功率。

原生接口详见 [接口审计](README.md)。其他两条已运行链路、专用 EGL 资源与时间预算以 [action-eval 当前交接](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md) 为准；不抢占已有任务、不碰他人进程，GPU 7 只用于渲染。
