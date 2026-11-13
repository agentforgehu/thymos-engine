# 路线图：从本地实验到可开源的研究工具

> 原则：**先证明它能被检验，再证明它有用，最后才证明它好看。**
>
> 本文档区分“已交付”“下一步”“暂不做”。时间估计按 1–2 人全职投入计算。
>
> 标签说明：本文的 P0–P4 是**项目阶段**；[GAPS.md](GAPS.md) 的 P0/P1/P2 是**问题优先级**；[STRATEGY.md](STRATEGY.md) 建议用 **E（工程）/ S（科学）/ P（产品）** 区分，避免把“能发布”当成“已证明”。科学验证的主线是 S1 消融 → S2 纵向基准 → S3 下游结果。

---

## 0. 愿景与非目标

### 愿景

做一个**可解释、可复现、可嵌入**的类人情绪—记忆动力学内核，让 Agent：

- 会遗忘，而不是无限堆上下文；
- 会受经历影响，而不是每轮重置；
- 能解释“为什么想起这件事、为什么现在难过”；
- 能在本地运行，不把最私密的记忆交给云端。

### 非目标

- 不做“有意识”的 AI；
- 不做心理诊断、治疗、危机干预；
- 不做情感操控、成瘾设计、拟人化欺骗；
- 不追求在记忆基准上刷 SOTA；
- 不替代真实的人际关系。

---

## 1. 阶段总览

| 阶段 | 目标 | 时间 | 退出条件 |
|---|---|---|---|
| **P0 本地实验** | 核心引擎、CLI、面板、测试、实验 | 已完成 | 247 项测试通过；6 个实验可复现 |
| **P1 科学验证** | 参数恢复、敏感性、消融、真实日志 | 4–8 周 | 关键参数可识别；消融结果公开 |
| **P2 开源发布** | 仓库、CI、文档、示例、v0.1.0 | 2–4 周 | 外部用户能在 10 分钟内跑通 |
| **P3 集成** | LLM/Agent 适配、SQLite、真实嵌入 | 6–10 周 | 一个真实 Agent 连续运行 7 天 |
| **P4 产品/研究** | 多用户、隐私、用户研究、基准 | 3–6 个月 | 有独立复现或用户数据 |

---

## 2. P0：本地实验版（已完成）

### 交付

- [x] `Thymos` 引擎：评价 → 情绪 → 内分泌 → 记忆 → 提取 → 巩固
- [x] FSRS 遗忘曲线 + ACT-R 激活 + 情境/心境一致性
- [x] 八变量内分泌层 + HPA 级联 + 昼夜节律
- [x] Open-Meteo 在线天气 + 离线合成天气 + 缓存
- [x] JSON 持久化、原子写入、状态快照
- [x] CLI：`init / demo / perceive / recall / tick / consolidate / snapshot / serve`
- [x] 零依赖 HTTP API + 可视化面板
- [x] 247 项测试
- [x] 5 个可复现实验 + 图表 + 指标 JSON

### 已知缺口

- 评价层是词典启发式；
- 嵌入是哈希 n-gram；
- SQLite 可选后端已有；多用户命名空间、静态加密、保留期未做；
- 没有真实用户研究；
- MCP 已有；LangChain/AutoGen/Dify/AstrBot 适配包未做；
- 参数未做全局敏感性分析。

### 0.2.0 可靠性升级（2026-11-13）

- [x] 多值事实、事件时间、`correct/delete`、来源/置信度
- [x] `<memory trust="untrusted">` 边界、PII 脱敏、HTTP token
- [x] 可选 `LLMFactExtractor`（schema + allow-list + 规则兜底）
- [x] 词法混合检索、保守 token 预算、`last_context` 指标
- [x] 可选 `SQLiteMemoryStore`（WAL，事务写入）
- [x] stdio MCP 六工具 + 面板事实查看/删除
- [x] 120 例 held-out 生成器；24 例真实模型 + 50 轮长测

退出条件没有完全达到：P0 的“100 条对抗句 ≥90%”和 P1 的“100+ 条中文检索基准、reranker”仍未交付；0.2.0 只完成了低风险的一半。

### 0.3.0 记忆控制平面（2026-11-13）

- [x] `policy.py`：查询意图分类 + 情绪门控（0 / 0.15 / 0.35 三档）
- [x] `control.py`：事实 / 情景 / 图邻居 / 摘要四算子 + 合并去重
- [x] `AuditTrace`：算子报告、`selected_ids`、预算、耗时、可持久化
- [x] `backends.py`：`MemoryBackend` 协议；JSON / SQLite 实现均满足
- [x] `AgentMemory.plan()`、HTTP `/api/context`、MCP `memory_plan`、CLI `plan`
- [x] `docs/PRODUCT.md`：保留 / 改写 / 放弃 / 冻结清单
- [x] 12 项控制平面测试，总计 134 项

**这一步只证明“控制平面可运行、可审计”，还没有证明“比强基线更好”。** 下一步的 Go/No-Go 仍然看外部 benchmark 与真实案例，不看单元测试数量。

### 0.3.1 语言/心境审计（2026-11-13）

- [x] 修正 `_negated` 的字符窗口：`特别/别人/差不多/不断` 不再被当成否定
- [x] 增加口语攻击、感谢/夸奖、重大损失、转述贬低等 `LexicalCue`
- [x] 修复短消息被 `estimate_importance` 的 `len<12` 规则丢弃
- [x] 心境参数 18h/0.18 → 10h/0.40，并做敏感性扫描
- [x] 事实否定守卫：`我不住在上海`、`我不在上海工作` 等不再反向抽取
- [x] 30 轮本地 35B 定向长测 30/30；50 轮原长测 17/18（一次猫探针 top-k 漏检）
- [x] 测试总数 221

**仍未关闭：** 讽刺/引用/转述的真正理解、偏好陈述与当下情绪区分、50 轮猫探针的检索排序、参数拟合与真实用户研究。详见 [LANGUAGE_MOOD_AUDIT.md](LANGUAGE_MOOD_AUDIT.md) 与 [GAPS.md](GAPS.md) 第 8 节。

### 0.3.2 LLM 评价分层（2026-11-13）

- [x] `LLMAppraiser`：结构化 `Appraisal`、JSON 校验、数值 clamp、失败回退
- [x] `AgentMemory.observe()` 一次评价、重要性与 `perceive` 共用，删掉重复规则调用
- [x] `appraiser=` 与 `appraisal=` 两种接入；默认仍是规则
- [x] 7 条本地 35B 对照探针；全部 JSON 可解析，额外延迟 2.9–7.4 秒/条
- [x] 测试总数 228

**仍未关闭：** LLM 评价与规则评价的人工盲评 A/B、低置信混合门控、重复轮一致性、模型注入对抗测试。详见 [LLM_APPRAISAL.md](LLM_APPRAISAL.md) 与 [GAPS.md](GAPS.md) 第 9 节。

### 0.3.3 评价落点修复（2026-11-13）

- [x] `LLMAppraiser` 解包 `appraisal` / `result` / `data` 外层，外层 `note` 合并
- [x] `appraise()` 的 surprise 分支限制为 `abs(d) < 0.22`，强负向不再被 novelty 盖住
- [x] 新增 `/api/appraise`：返回维度、标签与阈值轨迹，不写状态
- [x] 覆盖 `d=-0.3 / agency=other / control=0.2` 与 `d=-0.6 / circumstance` 的端到端回归
- [x] 测试总数 236

**仍未关闭：** DeepSeek/本地模型真实载荷的重复轮一致性、模型输出字段缺失的告警、人工盲评。详见 [LLM_APPRAISAL.md](LLM_APPRAISAL.md) 5.1 与 [GAPS.md](GAPS.md) 第 10 节。

### 0.3.4 SQLite 状态后端（2026-11-13）

- [x] `.sqlite` / `.db` 路径自动启用记忆 + 引擎状态后端
- [x] `memories` / `edges` / `meta.engine_state` 分表；schema v1
- [x] JSON → SQLite 迁移；源文件不改，目标已有状态时拒绝覆盖
- [x] `flush()` 回写 review/遗忘/prune 变更；WAL checkpoint
- [x] 并发写入、schema/状态版本、迁移与 round-trip 回归
- [x] 测试总数 247

**仍未关闭：** 检索仍是内存扫描；没有崩溃注入、多进程写者协议、加密和在线备份。详见 [SQLITE_STATE.md](SQLITE_STATE.md) 与 [GAPS.md](GAPS.md) 第 11 节。

---

## 3. P1：科学验证（4–8 周）

### 3.1 参数恢复

**问题**：我们能否从生成的数据中恢复已知参数？

- 对 `S0`、`spacing` 增益、`τ_m`、`κ`、`retrieval_cortisol_penalty` 做网格/贝叶斯拟合；
- 报告后验分布、置信区间、可识别性；
- 不能恢复的参数，标记为“结构性假设”，不当作可调参数。

**退出条件**：至少 4 个核心参数可在 10% 误差内恢复；不可识别的参数写入文档。

### 3.2 全局敏感性分析

- 使用 Sobol / Morris 方法；
- 输出：每个参数对“30 天心境”“提取排名”“稳定度”的方差贡献；
- 找出“看起来重要但其实没用”的参数，简化模型。

**退出条件**：公开敏感性热图；删掉贡献 <1% 的参数或合并。

### 3.3 消融实验

分别关闭：

- 内分泌层；
- 天气耦合；
- 再巩固；
- ACT-R 激活；
- 心境一致性；
- 图式抽象。

比较：检索排名、心境轨迹、长期稳定度、行为多样性。

**退出条件**：每个模块的边际贡献有数据；没有贡献的模块被删除或降级为可选。

### 3.4 真实日志拟合

- 数据源：Anki 复习日志（公开数据集）、个人日记（需伦理审查）；
- 用真实 `R` 序列拟合 `S0` 与 `spacing`；
- 与 FSRS 默认参数对照；
- 报告样本量、缺失、选择偏差。

**退出条件**：至少一个公开数据集上的拟合结果可复现；与默认参数的差异有解释。

### 3.5 预注册

在 P1 开始前，把假设、指标、分析计划写入 `docs/PREREGISTRATION.md`，避免“调参调到显著”。

---

## 4. P2：开源发布（2–4 周）

### 4.1 仓库

- 名称：`thymos-engine`（见 [NAMING.md](NAMING.md)）
- 许可证：MIT
- 分支：`main` 保护；PR 需要测试通过
- 标签：`v0.1.0`、`v0.1.1`…
- 发布物：源码、wheel、sdist、示例、实验数据

### 4.2 基础设施

- [x] GitHub Actions：Python 3.10/3.11/3.12 测试
- [ ] PyPI 发布（Trusted Publishing）
- [ ] pre-commit：ruff + black + mypy（可选）
- [ ] 代码覆盖率（codecov）
- [ ] Dependabot
- [ ] Issue/PR 模板
- [ ] `SECURITY.md`
- [ ] `CITATION.cff`
- [ ] 文档站（MkDocs Material 或 GitHub Pages）

### 4.3 示例

- `examples/basic.py`：最小记忆/情绪循环
- `examples/agent_loop.py`：把回忆注入 Agent 上下文
- `examples/chatbot_demo.py`：三调用接入 + 遗忘/复习演示
- `examples/llm_chat.py`：DeepSeek / Ollama 真实 LLM 接入
- `examples/live_weather.py`：在线天气
- `examples/ablation.py`：关闭模块做对照（P1 后）
- Notebook：`notebooks/01_quickstart.ipynb`、`02_experiments.ipynb`

### 4.4 发布文案

- 一句话：**“A local-first affective memory engine with forgetting curves, endocrine modulation, and weather coupling.”**
- 不写“有意识”“懂你”“比人更懂人”。
- 明确“research sandbox, not clinical tool”。

### 4.5 退出条件

- 全新机器上 10 分钟内跑通 quickstart；
- CI 全绿；
- 至少 3 个外部用户能复现实验；
- 收到至少 5 条实质性 Issue（功能/参数/文档）。

---

## 5. P3：集成（6–10 周）

### 5.1 存储

- SQLite 后端：`memories`、`associations`、`events`、`snapshots`；
- 迁移工具：JSON ↔ SQLite；
- 索引：`last_access`、`stability`、`tags`；
- 加密：SQLCipher 可选，密钥由用户管理。

### 5.2 嵌入

- [x] 抽象 `Embedder` 协议（`str -> list[float]`）；
- [x] 哈希（默认）、Ollama `/api/embed`（推荐 `bge-m3`）、OpenAI-compatible `/embeddings`；
- [ ] sentence-transformers、本地 GGUF；
- [x] 文本级 LRU 缓存；批量接口尚未做；
- [x] 维度变更的自动重算（首次 `recall`/`perceive`）。

### 5.3 评价

- `AppraisalProvider` 协议；
- 规则基线、LLM few-shot、结构化 JSON schema；
- 失败回退；
- 审计日志：保留原始文本、模型、prompt hash、输出。

### 5.4 Agent 适配

- **MCP Server**：`memory.perceive`、`memory.recall`、`affect.snapshot`、`sleep.consolidate`；
- **Letta**：作为 external memory / tool；
- **LangGraph**：node + checkpointer；
- **OpenAI Agents SDK**：tool wrapper；
- **自建循环**：`examples/agent_loop.py`。

### 5.5 观测

- OpenTelemetry traces：每次 recall 的 components；
- 本地日志：不包含原始内容，只有 ID/分数；
- 面板：时间线、记忆网络、参数面板。

### 5.6 退出条件

- 一个真实 Agent 连续运行 7 天，不崩溃、不内存爆炸；
- 回忆延迟 p95 < 50 ms（1 万条记忆，SQLite + 哈希嵌入）；
- 用户能导出/删除全部数据。

---

## 6. P4：产品与研究（3–6 个月）

### 6.1 多用户与隐私

- 每用户独立状态文件/SQLite；
- 静态加密；
- 数据保留策略；
- 导出/删除 API；
- 不训练、不共享、不遥测（默认）。

### 6.2 用户研究

- 2–4 周日记研究：事件、情绪、睡眠、天气、回忆；
- 预注册假设；
- 伦理审查（IRB 或等效）；
- 结果无论正负都公开。

### 6.3 基准

- 不追求 LoCoMo SOTA；
- 建立“动力学一致性”基准：
  - 间隔效应方向；
  - 应激后提取下降；
  - 天气效应有界；
  - 再巩固可复现；
  - 长期稳定性。
- 允许其他系统提交；公开评分脚本。

### 6.4 可能的商业路径（不代表承诺）

- 开源核心 + 托管面板；
- 企业私有部署；
- 教育/写作/游戏 NPC 的情绪记忆中间件；
- 心理咨询**辅助**工具（必须与临床合作，不能独立诊断）。

---

## 7. 决策门（Go / No-Go）

| 门 | 问题 | 若否 |
|---|---|---|
| G1 | 核心参数可识别吗？ | 简化模型，删除不可识别参数 |
| G2 | 消融显示模块有贡献吗？ | 删除无贡献模块 |
| G3 | 外部用户能 10 分钟跑通吗？ | 改文档/安装/示例，不增加功能 |
| G4 | 有真实 Agent 连续运行 7 天吗？ | 先做稳定性，不做新功能 |
| G5 | 用户研究通过伦理审查吗？ | 暂停用户数据收集，只做合成/公开数据 |
| G6 | 有独立复现吗？ | 不宣称“有效”，只宣称“可运行” |

---

## 8. 风险与缓解

### 8.1 伪科学风险

**风险**：把相对单位、手调参数包装成“科学证明”，误导用户。

**缓解**：

- 全文标注 L/E/P/D 来源；
- 不给出“你的皮质醇是 X”这类绝对值；
- 明确“不是诊断”；
- 实验报告写“不能证明什么”；
- 邀请外部批评，保留反例。

### 8.2 隐私风险

**风险**：记忆包含极其私密的内容；天气 API 泄露位置。

**缓解**：

- 默认离线；
- 坐标保留两位小数；
- 不发送用户 ID；
- 本地缓存可删除；
- 未来 SQLCipher；
- 不做遥测。

### 8.3 情感依赖/操控

**风险**：系统被用来制造“懂你”的假象，诱导依赖。

**缓解**：

- 禁止“只有我懂你”“不要离开我”等话术；
- 不模拟脆弱性来索取信任；
- 明确 AI 身份；
- 提供“这不是真人”的持续提示；
- 不针对孤独/抑郁人群做增长实验。

### 8.4 文化偏差

**风险**：OCC 情绪类别、词典、天气效应都以西方/温带样本为主。

**缓解**：

- 参数化文化/地区；
- 多语言词典与本地化评价；
- 用户可覆盖标签；
- 公开偏差评估。

### 8.5 安全

**风险**：状态文件被篡改；API 被本地恶意进程访问。

**缓解**：

- 默认只监听 `127.0.0.1`；
- 无认证的本地 API 明确标注“仅本机”；
- 未来加 token；
- 输入长度/频率限制；
- 不执行用户文本中的代码。

### 8.6 可持续性

**风险**：一个人维护，热情耗尽。

**缓解**：

- 小核心、少依赖；
- 自动化测试/发布；
- 明确“维护者时间”；
- 欢迎 co-maintainer；
- 不做无法维护的功能。

---

## 9. 开源清单

### 必须

- [x] README（中文 + 英文摘要）
- [x] LICENSE（MIT）
- [x] CONTRIBUTING
- [x] CODE_OF_CONDUCT
- [x] CHANGELOG
- [x] CI
- [x] Issue/PR 模板
- [x] SECURITY.md
- [x] CITATION.cff
- [x] 英文 README
- [ ] PyPI 发布
- [ ] 文档站
- [x] 至少 3 个 examples
- [ ] 至少 1 个 Notebook

### 建议

- [ ] 录屏（3 分钟 quickstart）
- [ ] 架构图（SVG）
- [ ] 参数交互式探索（Observable/Streamlit）
- [ ] 社区 Discord/Matrix（可选）
- [ ] 月度 changelog

---

## 10. 近期 14 天计划

| 天 | 任务 | 产出 |
|---|---|---|
| 1–2 | 仓库初始化、CI、README 校对 | 可 push 的 v0.1 |
| 3–4 | 示例与截图 | `examples/`、`docs/figures/dashboard.png` |
| 5–6 | 参数恢复脚本 | `experiments/parameter_recovery.py` |
| 7–8 | 敏感性分析 | `docs/SENSITIVITY.md` |
| 9–10 | 消融实验 | `docs/ABLATION.md` |
| 11–12 | 英文 README | `README.en.md` |
| 13–14 | 发布 v0.1.0、收集反馈 | PyPI/GitHub Release |

---

## 11. 一句话

> 先把可证伪性做扎实，再谈“类人”。Thymos 的野心不是让 AI 假装有情绪，而是让情绪与记忆的动力学**可以被检查、被反驳、被改进**。
