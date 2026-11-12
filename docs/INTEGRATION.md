# 集成指南：把 Thymos 接进一个 Agent

> 目标：让 Agent 在每轮对话前后做两件小事——**先回忆，再经历**——而不是把全部历史塞进上下文。
>
> 本文是接口参考。想看“记忆怎么存、遗忘怎么算、情绪怎么来、LLM 负责什么”，先读
> [TUTORIAL.md](TUTORIAL.md)。本文给出三种集成方式：Python 直接调用、HTTP API、Agent 循环模式。
> MCP Server 在路线图中（P3）。

---

## 1. 最小集成：`AgentMemory`

```python
from thymos import AgentMemory, Profile, ThymosConfig

memory = AgentMemory(
    ".thymos/agent.json",
    profile=Profile(name="用户", location_name="上海", timezone="Asia/Shanghai"),
    config=ThymosConfig(offline=True, seed=7),
)

# 对话前：按用户这句话回忆，并渲染成提示词块
# max_chars 是硬预算；本地小模型建议 600–1200
context = memory.context(user_text, top_k=3, min_score=0.30, max_chars=900)
system_prompt = f"""你是一个有记忆的助手。以下是相关的过去经历与当前状态。
不要声称自己有真实的生理感受；这些是模拟状态，用于保持叙事一致。

{context}
"""

# 对话后：只把用户说的、值得记的部分编码为经历
memory.observe(user_text)

# 夜间/退出时：巩固并保存
memory.end_session(7.5)
```

`AgentMemory` 是薄门面，底层仍然是 `Thymos`；需要完全控制时用 `memory.mind`。
只有回复里包含承诺/约定/重要自我披露时，才传 `assistant_text`；不要默认保存每条模型回复。

### 1.1 底层等价写法

`render_context()` 已经内置，不需要自己拼：

```python
from thymos import Profile, Thymos, ThymosConfig

mind = Thymos(
    profile=Profile(name="用户", location_name="上海"),
    config=ThymosConfig(offline=True),
    state_path=".thymos/agent.json",
)

hits = mind.recall(user_text, top_k=5)
context = mind.render_context(hits)   # Top-k 记忆 + 心境/压力摘要
mind.perceive(f"用户说：{user_text}", importance=0.7, tags=["对话"])
mind.consolidate(7.5)
mind.save()
```

如果你想要产品的语气，可以基于 `hits` 自己渲染；但不要把 `snapshot()` 的完整 JSON 直接塞进 prompt。

**不要**把 `endocrine` 的原始数值直接写给 LLM，除非你在做可解释性界面。LLM 很容易把“皮质醇 0.62”编造成“你的压力激素过高”。

### 1.2 什么时候编码

| 时机 | 是否编码 | 原因 |
|---|---|---|
| 用户陈述事实/偏好 | 是 | 未来可能有用 |
| 用户表达情绪 | 是，提高 importance | 情绪电荷影响遗忘 |
| 闲聊寒暄 | 否/低 importance | 避免记忆膨胀 |
| Agent 自己的推理 | 否 | 不是经历，是中间产物 |
| 重要决定/承诺 | 是，高 importance | 需要长期保留 |
| 敏感信息 | 默认不编码；需显式同意 | 隐私 |

### 1.3 什么时候回忆

- 每轮都回忆一次（top_k=3–5），成本很低；
- 长对话中，用户提到“上次”“之前”时强制回忆；
- 生成前如果 `max(hit.score) < 0.35`，告诉模型“没有相关记忆”，不要编造；
- 把 `components` 留给调试界面，不要全塞进 prompt。

### 1.4 本地小模型：用预算而不是长历史

本地模型上下文有限时，不要把最近几十轮对话直接拼进 prompt。`max_chars` 会先放结构化事实（姓名、城市、地址、宠物、项目、过敏、作息等），再用剩余空间放紧凑的情景记忆；被新事实取代的旧情景会被过滤。

```python
# 900 字符预算，适合 2K–4K token 的本地模型
context = memory.context(user_text, top_k=5, max_chars=900)
```

如果模型更小，可以从 600 或 400 开始。不要同时把 `snapshot()`、完整 Top-k 和原始对话都塞进去；预算越小，越应该依赖事实块。900 字符预算下的 50 轮对照结果见 [BUDGET_CONTEXT.md](BUDGET_CONTEXT.md)。

---

## 2. HTTP API

启动：

```bash
python -m thymos --offline serve --host 127.0.0.1 --port 8787
```

### 2.1 端点

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/state` | 完整状态快照 |
| GET | `/api/memories?limit=50` | 最近记忆 |
| GET | `/api/export` | 导出完整状态（JSON） |
| GET | `/api/health` | 健康检查 |
| POST | `/api/perceive` | 编码事件 |
| POST | `/api/recall` | 按线索回忆 |
| POST | `/api/tick` | 推进时间 |
| POST | `/api/consolidate` | 睡眠巩固 |
| POST | `/api/reappraise` | 重新评价一条记忆 |

### 2.2 示例

```bash
curl -s http://127.0.0.1:8787/api/perceive \
  -H "Content-Type: application/json" \
  -d '{"content":"项目评审通过了","importance":0.8,"tags":["工作"]}'
```

```bash
curl -s http://127.0.0.1:8787/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"评审","top_k":3}'
```

PowerShell：

```powershell
$body = @{ content = "项目评审通过了"; importance = 0.8; tags = @("工作") } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8787/api/perceive -Method Post -Body $body -ContentType "application/json"
```

### 2.3 安全

- 默认只监听 `127.0.0.1`，不接受外网连接；
- **没有认证**：任何能访问本机端口的进程都可以读写。不要把它暴露到公网；需要时放在反向代理 + token 后面。
- 请求体大小未限制；生产部署应加限制。
- 错误会以 `{"error": "..."}` 返回，HTTP 400/404。

---

## 3. Agent 循环模式

### 3.1 推荐循环

```text
用户输入
  ↓
recall(query=用户输入, top_k=5)
  ↓
构造 prompt：系统人格 + 相关记忆 + 当前模拟状态 + 本轮输入
  ↓
LLM 生成
  ↓
perceive(本轮交流, importance=估计值, tags=主题)
  ↓
（可选）reappraise 被用户纠正的旧记忆
  ↓
返回用户
```

夜间/空闲：

```text
consolidate(7.5)
save()
```

### 3.2 importance 怎么估

不要每次都让 LLM 打 0–1 分（贵且不稳定）。规则：

```python
importance = 0.35
if any(k in text for k in ["记住", "重要", "答应", "承诺", "生日", "计划"]):
    importance = 0.85
elif any(k in text for k in ["喜欢", "讨厌", "害怕", "担心"]):
    importance = 0.7
elif len(text) < 12:
    importance = 0.3
```

也可以让 LLM 在结构化输出里给 `importance` 与 `appraisal`，但要缓存、限流、保留审计。

### 3.3 避免“记忆污染”

- Agent 自己的推理不要编码；只编码用户输入与最终回应；
- 用户更正旧事实时，调用 `reappraise` 或写入一条“更正”记忆并降低旧记忆显著度；
- 定期检查 `top_memories`，删除错误/过期/敏感条目；
- 不要在 prompt 里写“你记得所有事”；相反，明确“你只记得以下片段”。

### 3.4 情绪状态的用法

**可以**：

- 让语气与模拟心境一致（“今天有点低落，回答更简短”）；
- 在回忆负面记忆时更谨慎；
- 在面板中展示状态，让用户理解行为变化。

**不可以**：

- 用情绪状态操控用户（“我很难过，你别走”）；
- 声称自己真的有激素/身体；
- 把模拟状态当作诊断；
- 在用户脆弱时伪造共情。

---

## 4. 替换组件

### 4.1 嵌入

内置的 Ollama 后端（推荐中文用 `bge-m3`）：

```python
from thymos import OllamaEmbedder, Thymos

mind = Thymos(embedder=OllamaEmbedder(model="bge-m3"))
```

也可以接任何 `str -> list[float]` 的函数：

```python
from thymos import Thymos
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

mind = Thymos(embedder=embed)
```

要求：同一文本稳定、维度一致、返回 list[float]。换嵌入后旧向量维度不兼容，引擎会在首次 `recall`/`perceive` 时自动用新 embedder 重算全部记忆（事件日志里会有 `reembed`）；状态很大时建议先备份或 `--fresh`。

### 4.2 评价

```python
from thymos.types import Appraisal

def appraise_with_llm(text: str) -> Appraisal:
    # 调用你的 LLM，要求返回 JSON schema
    data = my_llm_json(prompt=text, schema=Appraisal)
    return Appraisal(**data).clamped()

mind.perceive(text, appraisal=appraise_with_llm(text))
```

保留 `note` 字段记录来源（模型名、prompt hash、置信度），便于审计。

### 4.3 天气

```python
mind.config.weather_enabled = False      # 完全关闭
mind.config.offline = True               # 合成天气
mind.config.weather_ttl_seconds = 600    # 更频繁刷新
```

或替换 `WeatherClient`：实现 `current(profile, timestamp) -> EnvironmentState` 即可。

---

## 5. 性能与容量

| 操作 | 1,000 条 | 10,000 条 | 备注 |
|---|---:|---:|---|
| `recall`（哈希嵌入） | ~5 ms | ~40 ms | 单核，纯 Python |
| `perceive` | ~2 ms | ~2 ms | 关联只查 top-3 |
| `tick(24h)` | ~15 ms | ~15 ms | 24 个子步 |
| `consolidate(7.5h)` | ~20 ms | ~30 ms | 重放 30 条 |
| 状态 JSON | ~0.5 MB | ~5 MB | 取决于内容长度 |

生产建议：

- 1 万条以上换 SQLite + 向量索引；
- 不要每次 `save()`；批量/退出时保存；
- `recall` 的 `top_k` 控制在 3–8；
- 定期 `prune()` 归档低可提取度记忆。

---

## 6. 隐私清单

- [ ] 默认 `offline=True` 或明确告知会调用天气 API；
- [ ] 坐标保留两位小数；
- [ ] 不发送用户 ID、设备 ID、原始文本；
- [ ] 状态文件路径可配置，权限 600；
- [ ] 提供“导出全部数据”与“删除全部数据”；
- [ ] 敏感内容不编码，或加密存储；
- [ ] 不把记忆用于训练/微调，除非用户明确同意；
- [ ] 日志不记录原始内容，只记录 ID/分数。

---

## 7. 未来：MCP Server

计划中的工具：

```text
memory.perceive(content, importance?, tags?)
memory.recall(query, top_k?)
memory.reappraise(id, delta?)
affect.snapshot()
sleep.consolidate(hours?)
```

设计约束：

- 只读工具默认返回最小必要信息；
- 写工具需要显式调用，不在后台自动写入；
- 所有返回带 `components`，方便宿主调试；
- 支持 `dry_run`。

---

## 8. 反模式

| 反模式 | 为什么不好 | 替代 |
|---|---|---|
| 把全部历史塞进 prompt | 贵、慢、被无关信息淹没 | recall top-k + 摘要 |
| 每句话都编码 | 记忆膨胀、检索噪声 | importance 门槛 |
| 用情绪状态操控用户 | 伦理问题 | 只用于语气/界面 |
| 把模拟激素当医学指标 | 误导 | 标注相对单位 |
| 不设遗忘 | 上下文无限增长 | FSRS + prune |
| 不让用户查看/删除 | 隐私风险 | 面板 + export/delete |
