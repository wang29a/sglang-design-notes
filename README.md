<div align="center">

# SGLang Design Notes

**沿一个请求，读懂 LLM 推理运行时。**

从真实源码出发，解释 SGLang 的设计动机、模块边界、状态不变量与工程取舍。

![Language](https://img.shields.io/badge/Language-简体中文-2563eb)
![Format](https://img.shields.io/badge/Format-Markdown%20%2B%20SVG%20%2B%20Mermaid-0f766e)
![Evidence](https://img.shields.io/badge/Evidence-Source%20Review-7c3aed)

[开始阅读](#开始阅读) · [学习路线](ROADMAP.md) · [文章索引](index.json) · [参与贡献](#参与贡献)

</div>

![请求从入口、调度和执行视图进入模型，再将输出返回客户端的教学示意](assets/request-lifecycle.svg)

## 关于项目

SGLang 把不断到达的请求组织成动态批次，让它们共享有限的 GPU 算力与 KV 空间。理解这个过程，需要把请求入口、调度、缓存、模型执行和结果回流连起来看。

本项目是一份持续积累的中文源码学习笔记。每次围绕三个相互关联的问题，用一个具体请求串起设计动机、状态变化与实现细节，解释“为什么需要这样设计”，再讨论它的成本和适用条件。内容面向想深入 LLM Serving、Inference Runtime 与 AI Infra 的读者；理解 Python、Transformer 和自回归生成的基本概念，就可以从第一组文章开始。

每篇都固定源码提交，给出可核对的函数、行号和链接，并提供下一步调试时值得观察的状态。当前已发布的内容属于静态源码分析；文章中的简化轨迹和机制图是教学示意，实验建议也会明确标注为待执行。

## 开始阅读

第一组文章沿同一个请求走过三道边界：请求如何交给集中调度，调度状态如何变成模型输入，以及输入所引用的共享 KV 怎样保持有效。按下面的顺序阅读，能够建立后续学习的基本框架。

| 顺序 | 文章 | 阅读后应能回答 |
| :---: | --- | --- |
| 01 | [HTTP 请求为什么不能直接拥有推理循环](daily/2026-10-10/01-process-boundaries.md) | 网络等待与推理调度为什么分开？派发前后失败分别由谁清理？ |
| 02 | [为什么一个请求要经过三种不同的表示](daily/2026-10-10/02-batch-lowering.md) | `Req`、`ScheduleBatch` 和 `ForwardBatch` 各自表达什么？逻辑位置如何关联物理 KV 槽？ |
| 03 | [共享前缀 KV 为什么必须同时设计保护和回收](daily/2026-10-10/03-prefix-cache-ownership.md) | 缓存命中为什么不等于可以淘汰？引用保护如何避免共享状态被过早释放？ |

这一组发布于 **2026-10-10**，基于源码提交 [`0f6761b54fac`](https://github.com/sgl-project/sglang/tree/0f6761b54facebb47f2068f87ecccd8f14da3a0e)。后续文章按日期保存在 [`daily/`](daily/)，完整元数据见 [`index.json`](index.json)。

## 阅读归档

| 日期 | 三个相连的问题 | 源码版本 |
| --- | --- | --- |
| 2026-10-11 | [动态批次与请求归属](daily/2026-10-11/01-continuous-batching.md) → [分块进度与输出提交](daily/2026-10-11/02-chunked-prefill-progress.md) → [策略与资源准入](daily/2026-10-11/03-policy-and-admission.md) | `0f6761b54fac` |
| 2026-10-10 | [进程边界](daily/2026-10-10/01-process-boundaries.md) → [批次表示](daily/2026-10-10/02-batch-lowering.md) → [前缀生命周期](daily/2026-10-10/03-prefix-cache-ownership.md) | `0f6761b54fac` |

## 学习路线

路线从单请求闭环开始，再逐步增加请求之间的竞争、执行的并发和节点之间的数据移动。每个新主题都要说明它改变了前面哪个假设，以及已有的不变量如何继续成立。

| 阶段 | 核心问题 | 当前进度 |
| --- | --- | --- |
| 请求生命周期 | 进程边界、请求表示、前缀共享与资源归属 | 已发布第一组文章 |
| 调度与容量 | Continuous batching、chunked prefill、排队策略、KV 分配与回退 | 已发布批次形成、分块与准入；分配与回退待深入 |
| 执行与并发 | CPU/GPU overlap、延迟结果、CUDA Graph、attention backend | 规划中 |
| 功能与架构扩展 | 结构化输出、推测解码、多模态、HiCache、PD 分离与路由 | 规划中 |
| 分布式执行 | TP / DP / EP、通信、控制面与数据面 | 规划中 |

[ROADMAP.md](ROADMAP.md) 解释这些问题之间的联系。它是一条可调整的学习路线，实际选题会随源码和已有文章中暴露的疑问延伸。

## 如何使用这些笔记

先带着文章开头的问题读完贯穿例子，再沿固定版本链接核对代码。遇到新的状态字段，先问它由谁创建、在什么时刻更新、谁仍在引用它，以及什么时候可以回收；读到性能优化时，再问代价转移到了哪里、收益依赖哪些条件。

文章把源码事实、教学模型、运行观察与性能测量分别标明。断点、状态变量和预期观察帮助读者设计验证过程；只有真正执行并记录的实验才能成为运行证据。SGLang 的实现持续变化，文章结论应当结合各自的 `source_commit` 阅读。

## 本地核验

直接在 GitHub 阅读文章即可，无需启动模型服务。若要核对文章中的源码定位，准备 **Python 3.9+**、Git，以及包含对应提交的 SGLang checkout；核验脚本只使用 Python 标准库。

```bash
git clone https://github.com/wang29a/sglang-design-notes.git
cd sglang-design-notes

# 指向已有的 SGLang 源码 checkout；默认查找同级的 ../sglang
export SGLANG_SOURCE_DIR=../sglang

# 核对当天三篇正文、索引、源码符号及行号
python3 scripts/validate.py 2026-10-10

# 检查公开文件与可达 Git 历史中的敏感信息模式
python3 scripts/check_privacy.py --history
```

源码核验检查的是文章与指定提交的一致性。隐私检查通过模式识别本机路径、私人联系信息、内部标识和常见凭据格式，仍需人工审阅正文语境。

维护者可在 `main` 分支运行 `python3 scripts/publish.py YYYY-MM-DD`，提交当天文章与索引。发布器从 `origin` 识别目标仓库，使用 GitHub CLI 的已有认证，在推送后核对远端提交；这一步需要 GitHub CLI 和目标仓库的写入权限。

## 仓库结构

```text
.
├── daily/                  # 按日期归档的完整文章
├── assets/                 # README 与正文的机制图
├── index.json              # 主题、源码版本、证据等级与定位元数据
├── ROADMAP.md              # 学习路线与问题之间的联系
├── AGENTS.md               # 内容与贡献约定
└── scripts/
    ├── validate.py         # 正文、索引与固定源码定位检查
    ├── check_privacy.py    # 公开内容与历史的模式检查
    └── publish.py          # 提交、推送与远端核验
```

## 参与贡献

欢迎通过 Issue 指出解释不清楚的地方、源码定位错误，或提出值得深入的新问题。提交修正时，请附上涉及的文章、源码版本和支持判断的代码位置；补充运行证据时，还应说明配置、环境、复现步骤和实际观察。

新增文章请先阅读 [AGENTS.md](AGENTS.md)，检查已有索引，并更新正文、归档和元数据。讨论应围绕能被核对的机制与取舍展开，让后来的读者能够沿同一条证据链继续验证。

## 上游与来源

本项目独立维护，围绕 [SGLang](https://github.com/sgl-project/sglang) 的公开源码展开学习。文章中的上游代码与文档引用保留各自的出处和版本；机制图为本项目整理的教学示意。
