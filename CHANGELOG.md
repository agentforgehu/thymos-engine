# Changelog

本项目遵循语义化版本（SemVer）。研究性 API 在 1.0 前可能变化，破坏性变更会写入此处。

## [0.4.0] — 2026-11-14

行为出口、窗口事件流与真实中文评估：情绪不再直接绑死一种回复方式，一次对话不再只能吃一条消息，评估不再只靠作者构造案例。

### Added

- `behavior.py`：`BehaviorMapper` 协议、`BehaviorDirective` 与保守的 `DefaultBehaviorMapper`；`AgentMemory.behavior()` 和 `plan()["behavior"]` 可审计。
- `stream.py`：`Event`、`WindowSummary`、`normalize_events()`、`build_window_summary()`；时间戳单调补齐、用户/助手角色区分。
- `AgentMemory.observe_stream()`：逐条走写路径，窗口结束生成可检索的 semantic 摘要记忆，保留原始事件关联。
- `evals/companion_zh/`：CPED test split 156 条真实中文语句 + 8 个真实多轮窗口；固定来源、许可、采样规则与人工标注协议。
- `experiments/eval_companion_zh.py`：真实数据上的 emotion/sentiment 基线与窗口摘要检索检查。
- `tests/test_behavior.py`、`tests/test_stream.py`、`tests/test_companion_eval.py`。

### Changed

- 版本号 0.3.5 → 0.4.0；测试总数 263 → 278。
- README、集成指南、实验文档与路线图加入行为/事件流/评估集入口。

### Evidence

- CPED 156 条：emotion accuracy **0.0962**、emotion macro-F1 **0.0482**、sentiment accuracy **0.1731**、sentiment macro-F1 **0.1999**。
- 8 个真实窗口：摘要 `context()` 可检索 **8/8**，多数情绪方向 accuracy **0.50**（样本很小）。

### Not solved

- CPED 是电视对白，不是陪伴用户数据，也没有陪伴质量人工评分。
- 规则评价在真实对白上仍很弱；不能在 CPED 测试集上调词表再报告提升。
- `observe_stream` 是窗口批处理，不是异步队列；行为映射不是安全策略。

## [0.3.5] — 2026-11-13

写路径与长程稳定性：从“每轮都写”改成可审计的准入/合并，并修复检索排序被显著度翻转的问题。

### Added

- `write_path.py`：`store` / `merge` / `reinforce` / `skip` 四种写决策；精确去重跨会话生效，语义合并受 72 小时窗口、0.90 相似度和数字/日期冲突保护。
- `AgentMemory.observe()` 复用一次评价；合并时仍更新情绪/心境，但不新建第二条记忆。
- `mind.snapshot()["last_write"]` 与 `events` 中的写路径审计。
- 中文 BM25 单字通道；检索排序改为未压缩 activation，`probability` 仅保留给复习质量/遥测。
- `experiments/longrun_100.py`：100 轮事实、更新、重复、辱骂、夸奖、损失、讽刺、混合查询与 SQLite 持久化长测。
- `tests/test_write_path.py`、事实更新/查询裁剪测试与检索排序回归。

### Fixed

- 多个候选的 sigmoid 概率都饱和到 0.98 时，显著度乘数会把真正相关的记忆挤下去；现在直接按 activation 排序。
- `项目改名「潮汐 2」`、`截止 5 月 1 日`、`我换了工作，现在在做 AI 产品`、`把猫送回恩施`、`十二点睡` 等事实现在能抽取；旧值复述不会覆盖新值。
- `我养了一只叫豆豆的猫`、`我的猫豆豆最近不爱吃饭`、`把猫豆豆送回恩施` 现在能抽 `pet_name`；首次 100 轮因此从 99/100 修到 100/100。
- `现在住在徐汇区` 不再把“区”当成城市；`我搬到上海了` 会抽到上海。
- “你还记得我住在哪、做什么工作”不再被 `记得` 误判为全量档案请求。

### Changed

- 版本号 0.3.4 → 0.3.5；测试总数 247 → 263。
- 默认仍不依赖数据库/向量库；SQLite 仍是可选状态后端。

### Evidence

- 真实 100 轮（本地 35B + bge-small）：100/100 上下文与回答探针、0 模型错误、66 条记忆、平均 5.27 s/轮、心境跨度 0.3712、压力峰值 0.2671；见 [LONGRUN_100.md](docs/LONGRUN_100.md)。

### Not solved

- 语义合并阈值未做真人标注；哈希嵌入下改写通常不会合并。
- 24 条混合记忆上 bge-small-zh R@1 只有 0.583，中文召回仍不是生产级。
- 100 轮是作者构造的场景，不是真实用户数据；没有人工盲评。

## [0.3.4] — 2026-11-13

SQLite 状态后端：不再只把记忆条目放进 SQLite、同时把全部记忆重复写进 JSON。

### Added

- `.sqlite` / `.sqlite3` / `.db` 状态路径自动启用 `SQLiteMemoryStore` 状态后端。
- `memories`、`edges`、`meta.engine_state` 分表；`PRAGMA user_version` 记录 schema v1。
- `python -m thymos migrate old.json new.sqlite`：源 JSON 不改，目标已有状态时拒绝覆盖。
- `SQLiteMemoryStore.flush()`：持久化 review/遗忘/prune 后的对象变更。
- 并发写入、schema 版本、状态版本、迁移、round-trip 测试。

### Changed

- `Thymos.save()` 在 SQLite 模式下先 flush 记忆，再写非记忆状态；`engine_state.memory` 固定为空。
- 状态版本高于当前 `Thymos.VERSION` 时明确报错，不再静默读取。
- 版本号 0.3.3 → 0.3.4；测试总数 236 → 247。

### Not solved

- 检索仍是内存线性扫描，SQLite 不是向量索引；
- 没有多进程租约、加密、保留期和在线备份；
- 当前只有 schema v1，尚未提供真实的 v1→v2 迁移样例。

## [0.3.3] — 2026-11-13

评价落点修复版：解决“LLM 给了 `desirability=-0.3`，最终标签仍是 neutral”的可复现路径，并增加只读调试端点。

### Added

- `explain_appraisal()`：返回实际 `Appraisal`、`Emotion` 和阈值决策（`strong_negative`、`novelty_override`、`neutral_reason` 等）。
- HTTP `POST /api/appraise`：维度/文本进，标签与决策出；不写记忆、不推时间。
- `LLMAppraiser.last_raw` / `last_error`：单线程调试时查看模型原始输出与解析错误。

### Fixed

- `LLMAppraiser` 现在自动解包 `appraisal` / `result` / `data` 三种常见 JSON 外层，并合并外层 `note`；此前嵌套载荷会让所有维度静默默认为中性。
- `appraise()` 的 surprise 分支从 `abs(d) < 0.35` 收紧到 `< 0.22`，高 novelty 不再盖住明确负向事件。
- 增加 `d=-0.30 / agency=other / control=0.20` 与 `d=-0.60 / agency=circumstance` 的映射及 `AgentMemory` 端到端回归。

### Changed

- 版本号 0.3.2 → 0.3.3；测试总数 228 → 236。

## [0.3.2] — 2026-11-13

评价分层版：有 LLM 的宿主可以让模型填 `Appraisal`，Thymos 只负责状态、记忆和审计；规则保留为默认与失败兜底。`observe()` 不再把同一句话判断两次。

### Added

- `thymos.llm_appraisal.LLMAppraiser`：`chat(messages) -> str` 适配器，输出结构化评价 JSON；字段校验、数值 clamp、agency 枚举、失败回退。
- `experiments/appraisal_llm_probe.py`：规则与本地 LLM 的 7 条对照探针，输出标签、desirability、agency、耗时和证据。
- `docs/LLM_APPRAISAL.md`：为什么不是只判正负、三种接法、成本、真实 35B 对照与效度边界。

### Changed

- `AgentMemory.observe()`：一次 `Appraisal` 同时用于 `estimate_importance` 和 `perceive`；删掉重复规则调用。
- `AgentMemory` 新增 `appraiser=` 与 `observe(..., appraisal=)`；默认仍是规则。
- 版本号 0.3.1 → 0.3.2；测试总数 221 → 228。

### Boundaries

- 7 条本地探针不是 benchmark；没有人工 ground truth，不能宣称 LLM 评价更准。
- 本地 35B 额外评价耗时 2.9–7.4 秒/条；未默认开启。
- 讽刺、引用、转述和模型注入仍未解决；`LLMAppraiser` 只把模型输出当数据。

## [0.3.1] — 2026-11-13

语言与心境审计版：修复中文口语影响识别、否定边界、短消息被重要性门槛丢弃，并校准聊天尺度的心境响应。完整证据与未关闭项见 `docs/LANGUAGE_MOOD_AUDIT.md`。

### Added

- `docs/LANGUAGE_MOOD_AUDIT.md`：否定窗口、口语攻击/感谢/损失、短消息路径、心境敏感性、30 轮本地 35B 定向长测与 50 轮回归的审计报告。
- `experiments/longrun_language.py`：30 轮口语情绪/心境本地模型长测，含辱骂、感谢、损失、讽刺、歧义、中性对象与事实混合探针。
- `LexicalCue` 口语影响线索：直接侮辱、能力贬低、医学化侮辱、命令驱逐、轻蔑、转述贬低、感谢、夸奖、关心、重大损失、自我贬低；每条带 agency/control/stakes/norm/confidence。
- 中文否定例外与双重否定处理；`特别/别人/差不多/不断/无论/未来` 等不再被当成否定。
- 短消息重要性规则：`abs(desirability) >= 0.30` 的短消息给 0.85 重要性，避免 `闭嘴/你真好/项目黄了` 被 `len<12` 丢掉。

### Fixed

- `你怎么这么笨，闭嘴` 从 `neutral` 修为 `anger`；`别烦我` 不再判 `joy`；`今天特别开心/我特别喜欢这个方案/别人夸我` 不再被「别」反转成 `sadness`。
- `谢谢你，辛苦了`、`你真好`、`太厉害了`、`牛啊`、`靠谱`、`给力`、`你帮我大忙了` 等正向口语可识别。
- `项目黄了`、`我被裁了`、`我们分手了`、`家人生病了`、`我亏钱了` 等重大损失可识别。
- `我通过 API 调用` 不再因为「通过」被判正向；`垃圾分类/滚面团/讲笑话/菜市场` 等非情绪对象不误伤。
- `没想到项目黄了`、`没料到他会被裁` 不再因为「没」把损失反转成正向。
- 事实否定守卫：`我不住在上海`、`我不在上海工作`、`我不做软件`、`我对花生不过敏` 不再反向抽取。
- `mood_tau_hours` 18 → 10，`mood_reactivity` 0.18 → 0.40；单条辱骂心境约 -0.054，单条感谢约 +0.059。

### Changed

- 版本号 0.3.0 → 0.3.1；测试总数 134 → 221。
- README 心境描述从「好几天」改为「数小时到一天」，把更慢的皮质醇/适应负荷层与 mood 分开。
- 明确参数是工程折中，不是生理常数；讽刺/泛化脏话保持 ambiguous + 低 certainty。

## [Unreleased] — 2026-11-13

### Added

- `docs/STRATEGY.md`：目标校准与同类项目调研。把“P0–P2 可靠性升级”与“科学验证”拆开，提出 E（工程）/ S（科学）/ P（产品）标签、五条可证伪主张、Thymos-30 试点与 12 个月 OKR。
- `docs/STRATEGY.md` 4.6：从“记忆引擎”迁移到“记忆控制平面”，用表示路由、情绪路由、预算路由把红海数据平面变成渠道。
- 同类项目对照：MemoryBank、emotional-memory、YourMemory、OpenFeelz、MyPersona、Memary、Mem0、MemoryOS、MemOS、Letta、Graphiti、A-MEM、HippoRAG、MemU、MemEmo。
- `docs/LOCAL_LLM_MIXED_GATE_TEST.md`：本地 35B 的 65 轮混合查询长测与四门控消融。27/27 探针、8/8 混合意图、24/24 门控对照全部命中；结论是未观察到门控误伤事实，代价主要是上下文长度与更长回复。
- `docs/LOCAL_LLM_FIXES.md`：事实预算、饮食/健康事件抽取、情绪标签与压力动力学三项修复的本地 35B 复测。65 轮 0 模型错误、整体召回 96.3%、混合意图 100%、0 旧事实污染。
- `docs/FACT_EMOTION_FIXES.md`：事实按查询裁剪、中文句式、社交情绪与纯事实意图四项修复的 30 轮本地 35B 复测。10/10 探针、0 旧事实污染、0 模型错误；事实块在 300 字符预算下占 14–22%。
- 中文事实句式覆盖：`我从恩施来上海`、`做软件`、`他的项目叫 piko`、`我妹下个月结婚`、`香菜不要`；README 新增“支持的中文事实句式”表。

### Fixed

- `AgentMemory.observe`：即使宿主用 `min_importance=0` 保留问句，也不再从问句学习结构化事实，避免“我喝咖啡加糖吗？”被写成 `咖啡糖=加糖`。
- `FactStore.render`：按查询谓词选择事实；姓名只在被明确询问时作为锚点，项目问题不再附带姓名；未命中查询只给一行“其他已知”摘要，不再全量展开。事实块默认上限 35% 预算。
- 事实抽取新增 `food_like`、`food_dislike`、`food_event`、`health_event`；问句不写偏好/事件。
- `appraise`：移除 `goal_relevance >= 0.65 -> anticipation` 的兜底；中性事实保持 `neutral`，未来线索才进入 `anticipation`，未来 + 不确定进入 `anxiety`/`neutral`。新增夸奖 / 责骂 / 关心三类社交评价线索，避免明显事件全部塌缩为 `neutral`。
- `goal_relevance`：不再随句子长度饱和；长度项封顶，疑问句打折。
- `classify_intent`：`他住哪`、`他叫什么` 等第三人称纯事实问句直接判 `factual`，不再落入 `mixed`。
- `AffectDynamics`：压力增加有上限，正面事件可降低压力；静息目标下调，避免 30 轮温和事件把压力推到 0.8。
- `trim_to_tokens`：省略号不再把结果顶过 token 硬上限。

### Changed

- `README.md` 导航加入 `docs/STRATEGY.md`。
- `docs/ROADMAP.md` 增加标签说明：项目阶段、问题优先级、科学验证不要混用。
- 测试总数从 117 增至 134（新增事实相关性、中文句式、社交情绪与意图分类回归）。

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
