# 来源与证据台账

2026-09-20 的第三方 Joint 授权、访问检查与环境审计见 [独立准入记录](BADWAM-JOINT-ADMISSION-20260920.md)。以下记录保留 2026-09-18 的准备范围，不代表当前仍暂停实施。

## 范围与状态

**仅文档计划落盘与源码只读审计，实施未开始。** 用户稍后自行逐步实施；本轮未创建adapter、未部署、未安装、未执行Python/测试/模型/模拟器、未触发CI、未训练、未提交或push。文档中的函数行为为源码静态结论，不是运行验证。

- 工作目录：`/home/winbeau/Papers/ICLR2027-WAM-SA/FastWAM`
- 常用评测分支目标：`main`（default）；用户最新要求不使用长期action-eval分支，分支整合交父任务。
- 审计开始时实际观测branch：`action-eval/libero-paper-v1`（历史证据，不是推荐分支；本轮未切换）。
- 审计基线 HEAD：`6adf8c267e9195e17febf1b7ce72f6d2c89bf729`
- origin：`git@github.com:winbeau/FastWAM.git`
- upstream：`git@github.com:yuantianyuan01/FastWAM.git`
- 初始 `git status --short` 无输出。本轮仅新增4份文档并向 README 开头添加fork入口；未核验远端分支状态，未fetch/push。
- 时间来自本地 `date -u`，不是模型推断日期。会话工具输出是审计记录，未另存原始shell日志。

## 证据格式与台账

每条必须包含 **时间 / commit / 状态 / 命令 / 退出码 / artifact / 限制**。以下 `C` 统一指上面的完整基线HEAD；行号锚定此commit。时间区间为读取时段，工具无独立时间戳时不伪造精确秒。`read(...)` 是文件工具调用，不是执行源码；其退出码为N/A。shell工具成功不显示非零marker；下列成功shell记录为0。未运行项绝不记录0。

| ID | 时间 UTC | commit | 状态 | 命令/操作 | 退出码 | artifact/证据定位 | 限制 |
|---|---|---|---|---|---|---|---|
| E01 | 2026-09-18T19:00:18Z | C | OBSERVED | `pwd; git status --short; git branch --show-current; git rev-parse HEAD; git remote -v; date -u +%Y-%m-%dT%H:%M:%SZ; ls` | 0（组合命令） | 会话shell输出；上方仓库身份 | 本地身份，不代表远端同步或部署 |
| E02 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(README.md)`、`read(pyproject.toml)`、`read(uv.lock,1,55)`；`sha256sum pyproject.toml uv.lock` | read=N/A；hash shell=0 | README原文；依赖文件；下方SHA-256 | lock只读头部确认schema/Python，未解析或安装全部依赖 |
| E03 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(configs/train.yaml)`、`read(configs/sim_libero.yaml)`、`read(configs/task/libero_joint_2cam224_1e-4.yaml)`、`read(configs/model/fastwam_joint.yaml)`、`read(configs/data/libero_2cam.yaml)` | N/A | 上述原始配置；本目录README第1/4节 | 配置文本推导，未运行Hydra合成 |
| E04 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(experiments/libero/eval_libero_single.py)`、`read(experiments/libero/libero_utils.py,1,95)`、`read(.../libero_utils.py,155,40)` | N/A | 原生预处理、stats、rollout、输出；README第2–4节 | 未import LIBERO或torch，无输入样本实测 |
| E05 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(src/fastwam/runtime.py,125,135)`、`read(src/fastwam/models/wan22/fastwam_joint.py)`、`read(src/fastwam/models/wan22/fastwam.py,1100,105)` | N/A | Joint工厂/推理/加载函数；README第1节 | 未加载权重；strict=False不保证完整性 |
| E06 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(src/fastwam/datasets/lerobot/utils/normalizer.py,1,230)`、`read(.../processors/fastwam_processor.py,1,160)`及143–185、`read(src/fastwam/utils/pytorch_utils.py,1,62)`、`read(.../robot_video_dataset.py,1,28)` | N/A | normalize、seed、prompt源码；README第2–4节 | 未验证具体stats内容/数值/确定性 |
| E07 | 2026-09-18 19:00–19:02 | C | STATIC_AUDIT | `read(experiments/libero/run_libero_manager.py)`、`read(experiments/libero/run_libero_parallel_test.sh)`、`read(configs/data/robotwin.yaml,1,100)` | N/A | GPU/tmux/调度源码、RoboTwin配置 | 未运行shell脚本；RoboTwin已读训练部分足以确认维度/统计差异 |
| E08 | 本次任务接收；无独立服务器时间戳 | C（本地审计基线） | USER_REPORTED / BLOCKED | 用户说明：官方LIBERO Joint权重未找到，服务器已有RoboTwin Joint不匹配 | N/A | 本次任务文字与BLOCKERS.md | 未访问服务器/模型仓库，不声称官方权重永不存在 |
| E09 | 本次文档轮次 | C | NOT_RUN | 服务器安装、原生Joint推理、LIBERO评测、adapter验证、CI、训练 | N/A | 无运行artifact；只有本目录计划 | 用户决定稍后自行实施；不得用静态审计替代运行证明 |

只读路径发现时，`glob(configs/**/*)` 意外无匹配；随后 `git ls-files configs docs AGENTS.md uv.lock` 和 `ls configs` 确認文件存在，实际用read读取。该glob结果不用于证明文件缺失。

## 文档收尾核对

- 时间：2026-09-18T19:06:53Z；commit：C；状态：DOC_CHECK_ONLY。
- 命令：`git diff --check && git diff --stat && git status --short && sha256sum pyproject.toml uv.lock && git diff -- README.md && date -u +%Y-%m-%dT%H:%M:%SZ`。
- 退出码：0；artifact：会话shell输出，README仅开头增加4行；工作树为README修改、AGENTS.md与docs/action-eval新增；受保护文件SHA-256与审计开始一致。
- 限制：`git diff --check`不包含未跟踪文档，文档内容由文件工具审阅；未运行任何测试。这是文档范围核对，不是运行通过。

## 受保护文件指纹

本轮开始时SHA-256：

```text
968d4206505dec6afaaacbd07b0eb0a3c534e5ebe930017def9ccd2c7cce7084  pyproject.toml
0dd898d259089b5195d325b2170310961d8ba772429fe0f96312f29c401f6798  uv.lock
```

`pyproject.toml`声明Python `>=3.10,<3.11`，lock头部为`==3.10.*`。已固定torch `2.7.1+cu128`、torchvision `0.22.1+cu128` 等。保留上游已有构建依赖范围，不趁文档任务修改为其他值；新依赖精确pin规则不授权改现有两文件。本轮新增依赖：**无**。

## 外部来源（仅上游README提供的链接，不是本轮在线检索证据）

- 上游代码：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)
- 模型发布入口：[yuanty/fastwam](https://huggingface.co/yuanty/fastwam)
- LIBERO数据：[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
- LIBERO环境：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

已读README列举的是 `libero_uncond_2cam224.pt` / `robotwin_uncond_3cam_384.pt` 及对应stats，不足以证明LIBERO Joint已发布。未在本轮做在线权重检索或下载；不能将“未找到”扩大为“确定未发布”。

## 未来证据模板（未实施）

```text
时间: <UTC起止>
commit: <FastWAM完整SHA；环境源码完整SHA；dirty diff标识>
状态: PLANNED | BLOCKED | RUNNING | PASS | FAIL | NOT_RUN
命令: <完整实际命令、cwd、环境变量；敏感值脱敏>
退出码: <真实退出码；未运行写N/A>
artifact: <服务器日志/config/input/output路径与SHA-256>
限制: <未覆盖范围、配置差异、是否只有smoke而非完整评测>
```

checkpoint与stats须另记发布方URL、不可变revision、文件名、SHA-256、对应任务/架构、键shape完整性；当前这些字段均**未知/未验证**，不能填占位假值。保留未完成状态，等待用户后续授权和真实artifact。
