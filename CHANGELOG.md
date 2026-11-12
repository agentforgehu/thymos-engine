# Changelog

本项目遵循语义化版本（SemVer）。研究性 API 在 1.0 前可能变化，破坏性变更会写入此处。

## [Unreleased] — 2026-11-13

### Added

- `docs/STRATEGY.md`：目标校准与同类项目调研。把“P0–P2 可靠性升级”与“科学验证”拆开，提出 E（工程）/ S（科学）/ P（产品）标签、五条可证伪主张、Thymos-30 试点与 12 个月 OKR。
- `docs/STRATEGY.md` 4.6：从“记忆引擎”迁移到“记忆控制平面”，用表示路由、情绪路由、预算路由把红海数据平面变成渠道。
- 同类项目对照：MemoryBank、emotional-memory、YourMemory、OpenFeelz、MyPersona、Memary、Mem0、MemoryOS、MemOS、Letta、Graphiti、A-MEM、HippoRAG、MemU、MemEmo。
- `docs/LOCAL_LLM_MIXED_GATE_TEST.md`：本地 35B 的 65 轮混合查询长测与四门控消融。27/27 探针、8/8 混合意图、24/24 门控对照全部命中；结论是未观察到门控误伤事实，代价主要是上下文长度与更长回复。
- `docs/LOCAL_LLM_FIXES.md`：事实预算、饮食/健康事件抽取、情绪标签与压力动力学三项修复的本地 35B 复测。65 轮 0 模型错误、整体召回 96.3%、混合意图 100%、0 旧事实污染。

### Fixed

- `AgentMemory.observe`：即使宿主用 `min_importance=0` 保留问句，也不再从问句学习结构化事实，避免“我喝咖啡加糖吗？”被写成 `咖啡糖=加糖`。
- `FactStore.render`：按查询谓词选择事实，事实块默认上限 35% 预算；超限 `break` 而不是 `continue`，不再把整份档案倒进上下文。
- 事实抽取新增 `food_like`、`food_dislike`、`food_event`、`health_event`；问句不写偏好/事件。
- `appraise`：移除 `goal_relevance >= 0.65 -> anticipation` 的兜底；中性事实保持 `neutral`，未来线索才进入 `anticipation`，未来 + 不确定进入 `anxiety`/`neutral`。
- `AffectDynamics`：压力增加有上限，正面事件可降低压力；静息目标下调，避免 30 轮温和事件把压力推到 0.8。
- `trim_to_tokens`：省略号不再把结果顶过 token 硬上限。

### Changed

- `README.md` 导航加入 `docs/STRATEGY.md`。
- `docs/ROADMAP.md` 增加标签说明：项目阶段、问题优先级、科学验证不要混用。
- 测试总数从 117 增至 126（新增事实相关性、事件抽取、标签与压力回归）。

## [0.3.0] — 2026-11-13

控制平面 MVP：不重写动力学内核，把外层改成“意图 → 策略 → 算子 → 预算 → 审计”。

### Added

- `thymos.policy`：查询意图分类（factual / technical / history / affective / social / mixed）与情绪门控；事实/技术问题关闭心境一致性，支持型问题保留。
- `thymos.control.MemoryController`：事实、情景、图邻居、图式摘要四类算子；合并去重、硬预算渲染、`AuditTrace`。
- `thymos.backends.MemoryBackend`：可插拔记忆后端协议，`MemoryStore` 与 `SQLiteMemoryStore` 均满足。
- `AgentMemory.plan()`：返回 `context` + `intent` + `affect` + 算子审计；`context()` 默认走同一控制平面。
- HTTP `/api/context`、`/api/plan`、`/api/audit`；MCP `memory_plan`；CLI `thymos plan`。
- `docs/PRODUCT.md`：保留 / 改写 / 放弃 / 冻结清单，与 Mem0、Graphiti、Letta、MemoryBank 等项目的借/不借说明。
- 12 项控制平面测试（0.3.0 当时总计 117 项；当前总计 126 项）。

### Changed

- `render_context(include_state=False)`：事实/技术问题不暴露“当前模拟状态”，避免情绪语气污染事实回答。
- `recall()` 支持查询级 `mood_weight` / `emotion_weight` 覆盖，不改变全局配置默认值。
- 引擎快照与状态文件新增 `last_audit`，重启后仍可读最近一次计划。
- 包描述与 README 从“情绪记忆引擎”更新为“Agent 的记忆控制平面”。

### Known limits

- `mood_weight` 的 0 / 0.15 / 0.35 是产品默认值，不是拟合结果；敏感性分析未做。
- 图算子只做局部邻居，不做 PPR；摘要算子读取已有语义记忆，不现场生成。
- `MemoryBackend` 只有内建 JSON/SQLite 实现；Mem0/Graphiti 参考适配器在 0.3.x。
- 外部 benchmark（LongMemEval/LoCoMo）尚未跑通，当前证据仍是自建 held-out 与本地长测。

## [0.2.0] — 2026-11-13

P0–P2 可靠性升级：把“能跑”推进到“能被检验、能被纠正、能被嵌入”。

### Added

- 事实层支持多值谓词（过敏、宠物、项目、朋友）、`observed_at`/`source`/`confidence`、事件时间（未来搬家不提前改写当前城市）、`correct()` 与 `delete()`。
- `thymos.safety`：PII 脱敏、控制字符清理、注入短语替换、`<memory trust="untrusted">` 边界；`render_context()` 默认启用。
- `thymos.tokens`：保守 token 估算与硬预算裁剪；`AgentMemory.context(max_tokens=...)`。
- 混合检索：在语义/标签/地点/天气/时间之外加入词法通道，`recall()` 传入查询文本。
- `thymos.mcp_server`：零依赖 stdio MCP，提供 `memory_context`、`memory_observe`、`memory_recall`、`memory_correct_fact`、`memory_delete_fact`、`memory_snapshot`。
- `thymos.llm_extract.LLMFactExtractor`：可选 LLM 结构化抽取，带 allow-list、JSON 校验与规则兜底。
- HTTP API：Bearer token 认证、`/api/facts`、`/api/facts/correct`、`/api/facts/delete`、`/api/memories/delete`；面板加入当前事实查看/删除。
- `experiments/heldout_eval.py`：120 个模板生成的 held-out 多会话案例，thymos/recent/full/summary/rag 五条件，报告准确率、95% CI、上下文召回、陈旧值泄漏、假记忆率、字符/token/延迟。
- `docs/LOCAL_LLM_50_TEST.md`：本地 35B 模型 50 轮连续测试（同一状态文件、5 段 × 10 轮），16/16 探针通过，0 旧值误用。
- 15 项新测试（安全、token、MCP、API、LLM 抽取、SQLite、多值事实），总计 104 项。

### Changed

- `AgentMemory.context()` 支持 `max_tokens`，并把事实块与情景记忆一起放进信任边界。
- `AgentMemory.observe()` 支持 `learn_facts=False`；新增 `observe_llm()`、`facts()`、`correct_fact()`、`delete_fact()`。
- `engine.save()` 尽力将状态文件权限设为 `0600`。
- `serve` 增加 `--token` / `THYMOS_API_TOKEN`；默认仍为无 token 的 localhost 面板。

### Known limits

- 事实抽取的默认路径仍是规则，不是通用语义理解；LLM 抽取是可选适配器。
- 仍没有倒排/向量索引、跨进程写锁、静态加密和多用户命名空间；SQLite 只是可选存储后端。
- held-out 是合成案例，不是人类研究或公开 benchmark；真实结论需要更多模型、种子和人工盲评。

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

### Added

- `docs/EVALUATION.md`：客服/陪伴型 Bot 的任务级验证协议、样本量与 go/no-go 标准。
- `experiments/replay_harness.py`：五种记忆条件的多会话回放骨架；示例数据在 `experiments/data/replay_demo.jsonl`。
- `FactStore` / `Fact`：带有效期的高价值事实槽位，新值关闭旧值，支持历史查询与被取代值的过滤。
- `AgentMemory.context(..., max_chars=...)` 与 `Thymos.render_context(..., query, max_chars)`：硬字符预算，结构化事实优先，剩余预算才给情景记忆。
- `docs/BUDGET_CONTEXT.md`：900/600 字符预算下 50 轮 A/B 实测（Thymos 均为 16/16，最近对话基线 7/16）。
- `docs/GAPS.md`：现有不足与上线前审计，按 P0/P1/P2 给出事实可靠性、评测效度、隐私、检索与工程化优先级。
- `experiments/budget_compare.py`：同预算的 `thymos` vs `recent` 对照脚本。
- 9 项事实/预算回归测试；当前总计 89 项。

### Fixed

- 搬家意向（“考虑搬去/如果搬去”）不再提前覆盖当前城市；只有决定/完成搬家或地址陈述才更新。

### Planned

- 参数恢复与消融实验。
- SQLite 后端与真实嵌入。
- MCP Server 与 Agent 框架适配。
- 英文 README 与文档站。
- PyPI 发布。
