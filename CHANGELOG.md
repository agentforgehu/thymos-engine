# Changelog

本项目遵循语义化版本（SemVer）。研究性 API 在 1.0 前可能变化，破坏性变更会写入此处。

## [0.1.1] — 2026-11-12

面向接入的补丁版：不改变核心动力学，只降低“把它接进 Bot”的门槛。

### Added

- `AgentMemory` 门面：`context()` / `observe()` / `end_session()`，三调用接入聊天 Bot。
- `estimate_importance()`：透明、可审计的消息重要性先验。
- `docs/TUTORIAL.md`：从 0 到 1 的中文教程，解释记忆存储、遗忘、情绪、LLM 边界与改进点。
- `examples/chatbot_demo.py`：不调用真实 LLM 的可运行记忆 Bot 示例。
- 4 项 `AgentMemory` 测试，总计 71 项。

### Changed

- `examples/agent_loop.py` 改用内置 `mind.render_context()`，不再重复实现。
- `docs/INTEGRATION.md` 增加 `AgentMemory` 快速接入，修正“未内置 render_context”的过时说明。
- README 增加“接进自己的 Bot”入口。

## [0.1.0] — 2026-11-12

首个本地实验版。

### Added

- `Thymos` 引擎：评价 → 情绪 → 内分泌 → 记忆编码 → 提取 → 巩固。
- FSRS 幂律遗忘曲线、稳定度更新、间隔效应。
- ACT-R 基础激活 + 情境相似度 + 心境一致性检索。
- OCC 风格评价规则与中英文词典基线（含逐词否定）。
- 快情绪 / 慢心境 / 压力 / 疲劳 / 适应负荷。
- 八变量内分泌层：肾上腺素、去甲肾上腺素、皮质醇、多巴胺、血清素、催产素、褪黑素、内啡肽。
- CRH→ACTH→皮质醇级联，含负反馈与昼夜节律。
- 天气、光照、季节、地点、睡眠债耦合；Open-Meteo 客户端 + 离线合成天气。
- 再巩固窗口与情绪电荷改写。
- 睡眠巩固、记忆重放、语义图式抽象。
- JSON 持久化（原子写入）。
- CLI：`init`、`demo`、`perceive`、`recall`、`tick`、`consolidate`、`snapshot`、`serve`。
- 零依赖 HTTP API 与暗色可视化面板。
- 67 项 pytest 测试（0.1.0 基线；0.1.1 为 71 项）。
- 5 个可复现实验、图表与指标 JSON。
- 文档：研究综述、设计、实验、命名、路线图、集成指南。

### Known limitations

- 评价层是词典启发式，不是语义模型。
- 嵌入是哈希 n-gram，不保证语义质量。
- 内分泌是相对单位，不是生理浓度。
- 参数多为工程估计，未做全局敏感性分析/用户研究。
- 单用户、JSON 存储；没有 SQLite/加密/MCP。

## [0.1.2] — 2026-11-12

真实 LLM 接入与中文召回补强。

### Added

- `examples/llm_chat.py`：零依赖的 DeepSeek / Ollama 聊天示例，支持交互与 30 天演示。
- 示例默认只记用户事实、不保存助手回复；默认不把注入 Top-k 当成复习。
- `OllamaEmbedder` / `OpenAIEmbedder`：可选 HTTP embedding 后端，中文推荐 Ollama `bge-m3`。
- `AgentMemory(embedder=...)`；切换 embedding 后首次 `recall` 自动重算旧记忆向量。
- `experiments/retrieval_benchmark.py`：12 条中文 embedding 检索基线（hash 当前 recall@1 0.417 / recall@3 0.750 / MRR 0.572）。

### Fixed

- `estimate_importance()` 不再把“我不吃什么？”这类问句误判为偏好事实；带“提醒/答应/截止”的请求仍保留高重要性。
- 2 项问句回归测试；新增 7 项 embedding/迁移测试，当前总计 80 项。

## [Unreleased]

### Planned

- 参数恢复与消融实验。
- SQLite 后端与真实嵌入。
- MCP Server 与 Agent 框架适配。
- 英文 README 与文档站。
- PyPI 发布。
