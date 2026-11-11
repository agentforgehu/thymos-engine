# 从 0 到 1：把 Thymos 接进你的 Agent / Bot

> 这篇教程回答五个问题：
>
> 1. 别人怎么在自己的 Agent/Bot 里用它？
> 2. 记忆到底存在哪里、长什么样？
> 3. 遗忘是怎么算出来的？
> 4. 情绪/感受是怎么决定的？
> 5. LLM 决定什么，Thymos 决定什么，它到底能让 Bot 改进什么？
>
> 如果你只想先跑起来，跳到 [第 2 节](#2-十分钟接入agentmemory)；如果你想知道原理，按顺序读。

---

## 0. 一句话结论

**Thymos 不是 LLM，也不是“情绪提示词”。它是一个本地状态层：**

```text
用户说话
  │
  ├─ 1. recall：从记忆里找 3–5 条相关的
  │      + 当前心境/压力摘要
  │
  ├─ 2. LLM：拿着“人格 + 这几条记忆 + 当前状态”生成回复
  │
  ├─ 3. perceive：把值得记的那部分写回记忆
  │
  └─ 4. tick / consolidate：时间推进、睡眠巩固、遗忘
```

LLM 仍然是“嘴和推理机”；Thymos 是“海马体 + 时钟 + 一点内分泌”。它不替代 LLM，也不让 Bot 突然有意识。

---

## 1. 它在系统里的位置

```text
┌─────────────────────────────────────────────────────┐
│ 你的 Bot / Agent                                     │
│                                                     │
│  用户输入 ──► AgentMemory.context() ──► LLM ──► 回复 │
│                    ▲                       │         │
│                    │                       ▼         │
│              AgentMemory.observe() ◄──── 值得记的事   │
│                    │                                 │
│                    ▼                                 │
│              .thymos/bot.json                        │
└─────────────────────────────────────────────────────┘
```

三个边界要分清：

| 层 | 负责 | 不负责 |
|---|---|---|
| **LLM** | 理解这句话、组织语言、决定语气、调用工具 | 不决定“哪条记忆存在” |
| **Thymos** | 记忆的存储/遗忘/检索、心境与压力状态 | 不生成回复、不理解开放世界 |
| **宿主（你的 Bot）** | 什么时候 recall、什么值得记、用户能否删除 | 不把全部历史塞进 prompt |

---

## 2. 十分钟接入：`AgentMemory`

`AgentMemory` 是给聊天 Bot 的薄门面，只有三个核心调用：

```python
from thymos import AgentMemory, Profile, ThymosConfig

memory = AgentMemory(
    ".thymos/my_bot.json",
    profile=Profile(name="用户", location_name="上海", timezone="Asia/Shanghai"),
    config=ThymosConfig(offline=True, seed=7),
)

# 1) 回复前：拿相关记忆 + 当前状态
context = memory.context(user_text, top_k=3, min_score=0.30)

# 2) 你的 LLM
reply = my_llm(system_prompt + "\n\n" + context, user_text)

# 3) 回复后：只记用户说的、值得记的部分
memory.observe(user_text)

# 4) 每天/退出时：睡眠巩固并保存
memory.end_session(7.5)
```

`observe()` 默认会跳过“嗯、哈哈、谢谢”这类小话；要全部记录就传 `min_importance=0`。

默认**不要**把助手回复一起写回记忆。只有回复里包含承诺、约定或重要的自我披露时，才传第二个参数 `assistant_text`；否则模型的幻觉会变成长期记忆。

### 2.1 最小可运行例子

仓库里有一个不调用真实 LLM 的完整示例：

```bash
python examples/chatbot_demo.py
```

它会演示：

- Day 1 记住“我不吃虾”“我喜欢冰美式”“我周四面试”；
- 30 天不复习后，每条记忆的可提取度（R）下降；
- 第 30 天提问时，LLM 实际只看到 Top 3 记忆和当前心境；
- 一次真正的回忆会让那条记忆的 R 从 0.32 回到 1.00（检索练习）。

把 `fake_llm()` 换成 DeepSeek/OpenAI/Ollama 的调用，就是最小可用的记忆 Bot。

### 2.2 如果你想完全控制

`AgentMemory` 只是门面，底层仍然是 `Thymos`：

```python
from thymos import Profile, Thymos, ThymosConfig

mind = Thymos(
    Profile(name="用户", location_name="上海"),
    ThymosConfig(offline=True, seed=7),
    state_path=".thymos/my_bot.json",
)

hits = mind.recall("我上次说的面试", top_k=3)
context = mind.render_context(hits)       # 已内置，不需要自己拼
mind.perceive("用户说周四面试", importance=0.9, tags=["承诺"])
mind.tick(6 * 3600)
mind.consolidate(7.5)
mind.save()
```

---

## 3. 记忆是怎么保存的？

### 3.1 一条记忆不是一句话，而是一个结构

每次 `perceive()` 会创建一个 `MemoryItem`，核心字段：

| 字段 | 含义 |
|---|---|
| `content` | 原始事件文本 |
| `kind` | `episodic` / `semantic` / `affective` |
| `tags` | 主题标签，用于线索匹配 |
| `embedding` | 文本向量；默认是确定性的哈希 n-gram，不依赖外部模型 |
| `context` | 编码时的地点、天气、时段、心境快照 |
| `created_at` / `last_access` | 创建时间、上次被提取时间 |
| `access_times` | 每次提取的时间列表，用于 ACT-R 基础激活 |
| `stability` | 稳定度（天）；越大越不容易忘 |
| `difficulty` | 难度；影响复习增益 |
| `reps` / `lapses` | 成功复习次数 / 遗忘次数 |
| `valence` / `arousal` | 这件事的情绪电荷与唤醒度 |
| `salience` | 显著度；由 importance、情绪强度、新异性共同决定 |
| `associations` | 与相似记忆的稀疏关联（id → 权重） |
| `state` | `buffer` / `episodic` / `semantic` / `archived` |
| `labile_until` | 回忆后的再巩固窗口截止时间 |

### 3.2 状态文件长什么样

默认保存为普通 JSON，不是数据库：

```json
{
  "version": 1,
  "now": 1772617200.0,
  "profile": {"name": "用户", "location_name": "上海", "timezone": "Asia/Shanghai"},
  "affect": {
    "valence": 0.0,
    "mood_valence": -0.02,
    "stress": 0.14,
    "fatigue": 0.16,
    "dominant_emotion": "anticipation"
  },
  "endocrine": {
    "levels": {"cortisol": 0.32, "dopamine": 0.48, "serotonin": 0.55}
  },
  "environment": {"temp_c": 18.0, "cloud_cover": 40.0},
  "memory": [
    {
      "id": "a1b2c3d4e5f6",
      "content": "我不吃虾，过敏",
      "tags": ["对话"],
      "stability": 1.46,
      "valence": 0.0,
      "arousal": 0.26,
      "salience": 0.56,
      "state": "episodic",
      "associations": {}
    }
  ]
}
```

几个关键点：

- **一个用户一个文件**。不要把多个用户写进同一个 `.thymos` 状态，否则记忆会串。
- `save()` 先写 `.tmp` 再 `os.replace`，避免写一半坏掉。
- 默认没有加密；`.thymos/` 要当作隐私数据，不要提交到 Git。
- 记忆是**明文**的，包含原始文本；生产环境要加密磁盘/数据库，并提供导出与删除。
- 默认 `embedding` 是哈希 n-gram，只保证可复现，不保证语义质量；要更好的检索，注入真实 embedding（见 `docs/INTEGRATION.md` 第 4 节）。

### 3.3 检索时看什么

`recall()` 不是单纯向量相似度，而是几项加权：

| 分量 | 直觉 |
|---|---|
| `semantic` | 这句话和记忆内容像不像 |
| `tags` | 主题标签是否命中 |
| `context` | 地点/时段/环境是否相似 |
| `mood` | 当前心境是否与记忆的情绪电荷一致 |
| `base` | ACT-R 基础激活：常被提取、最近被提取的记忆更高 |
| `spread` | 关联扩散：通过相似记忆“顺藤摸瓜” |

每个结果都带 `components`，可以解释“为什么它排第一”。这是 Thymos 和黑盒向量库最大的差别之一。

---

## 4. 遗忘是怎么来的？

### 4.1 核心曲线

默认使用 FSRS 风格的幂律曲线：

```text
R(t, S) = (1 + factor * t / S) ^ decay
factor  = 0.9 ^ (1 / decay) - 1
```

- `R`：当前可提取度，0–1；
- `t`：距离上次提取的天数；
- `S`：稳定度（天）；
- `decay`：默认 `-0.5`。

这个 `factor` 的选取让 **`R(S, S) = 0.9`**：稳定度就是“可提取度降到 90% 所需的天数”。

以 `S = 3 天` 为例：

| 经过时间 | R |
|---:|---:|
| 1 天 | 0.963 |
| 7 天 | 0.804 |
| 30 天 | 0.547 |
| 90 天 | 0.353 |

不是“到第 3 天突然删除”，而是**连续衰减**。这就是为什么 Bot 可以“记得大概，但记不清细节”。

### 4.2 初始稳定度由什么决定

```text
salience = 0.55 * importance
         + 0.30 * 情绪强度
         + 0.15 * 新异性

S0 = base_days
     * (1 + 0.90 * |valence| * arousal + 0.30 * arousal)
     * (1 + 0.80 * salience)
     * encoding_gain
```

直觉：

- 越重要、情绪越强、越新的事，`S0` 越大；
- `encoding_gain` 来自内分泌层（肾上腺素/皮质醇/多巴胺），压力或唤醒会短暂提高编码；
- 但所有效应都有上下界，不允许单一因素“决定”一切。

### 4.3 复习为什么有用

成功提取后：

```text
spacing = 0.05 + 0.95 * (1 - R)
S_new = S * (1 + 1.7 * spacing * gain * quality)
```

- 在 `R` 很低时想起来，增益最大（间隔效应）；
- 在 `R` 接近 1 时连续重复，增益很小（集中复习不如间隔复习）；
- 提取失败（`quality < 0.35`）不会删除记忆，而是**收缩稳定度**；
- 回忆后有一个短暂的“再巩固窗口”，此时 `reappraise()` 改写情绪电荷更有效；窗口外效果减半。

### 4.4 睡眠巩固

`consolidate(7.5)` 会：

- 重放重要记忆，提升稳定度；
- 对反复出现的主题做图式抽象（episodic → semantic）；
- 让部分记忆进入更稳定的状态；
- 不会把所有记忆都变强，只重放高显著度的。

### 4.5 遗忘不是删除

- `R` 低只代表“不容易想起”，不代表条目消失；
- `prune()` 只在超过容量（默认 5000 条）时把最弱的标记为 `archived`；
- 真正删除要用 `store.remove(id)` 或宿主提供的“删除记忆”接口；
- 这意味着“遗忘”是可逆的：一条线索足够强，旧事仍可能被想起。

---

## 5. 情绪/感受是怎么决定的？

### 5.1 先评价，再产生情绪

Thymos 不把“开心/难过”当成一个分类标签，而是先算一组评价维度（OCC 风格）：

| 维度 | 问题 |
|---|---|
| `goal_relevance` | 这件事和我目标有关吗？ |
| `desirability` | 结果是好的还是坏的？ |
| `novelty` | 意外吗？ |
| `certainty` | 确定吗？ |
| `control` | 我能控制吗？ |
| `agency` | 是我、别人，还是环境造成的？ |
| `norm_compliance` | 符合规范/道德吗？ |
| `stakes` | 赌注大吗？ |

然后 `appraise()` 把评价映射成：

- 离散标签（13 类：joy / sadness / anxiety / anger / fear / trust / hope / surprise / ...）；
- PAD 连续量（valence 愉悦度、arousal 唤醒度、dominance 掌控感）；
- 强度与半衰期。

一个简化规则表：

| 评价模式 | 情绪 |
|---|---|
| 高新异 + 中性结果 | surprise |
| 好结果 + 低确定 | hope |
| 好结果 + 别人造成 + 符合规范 | trust |
| 坏结果 + 别人造成 + 高控制 + 高赌注 | anger |
| 坏结果 + 低确定 + 低控制 | anxiety / fear |
| 坏结果 + 其他 | sadness |

### 5.2 一个真实例子

输入：

```text
明天面试，不知道能不能过，有点害怕
```

词典启发式会得到近似：

```text
goal_relevance = 0.98
desirability   = -0.56
certainty      = 0.35
control        = 0.25
stakes         = 0.80
agency         = circumstance
```

映射结果：

```text
label = anxiety
valence ≈ -0.55
arousal ≈ 0.49
half_life ≈ 110 分钟
```

这不是“心理学诊断”，而是一组可审计的规则输出。你可以用 LLM 或训练好的分类器替换 `estimate_appraisal()`，引擎只依赖 `Appraisal` 结构。

### 5.3 快情绪、慢心境、内分泌

- **情绪**是快变量：一次事件产生一个脉冲，按半衰期衰减（5–480 分钟）；
- **心境**是慢变量：一阶积分，受情绪、内分泌、天气、昼夜节律影响，时间常数以小时/天计；
- **压力/疲劳/适应负荷**是更慢的背景；
- **内分泌**是八变量池（肾上腺素、去甲肾上腺素、皮质醇、多巴胺、血清素、催产素、褪黑素、内啡肽），调制编码、提取和心境；
- **天气/光照/季节**只给很小的心境偏置（有界），不允许盖过事件本身。

所以“感受”不是一个 prompt 里写死的“你现在很开心”，而是：

```text
事件 → 评价 → 情绪脉冲 → 心境积分 → 影响记忆编码/提取 → 影响下一轮语气
```

LLM 看到的只是这个状态的**摘要**，不是全部原始数值。

---

## 6. LLM 决定什么，Thymos 决定什么？

| 决策 | 谁来决定 | 说明 |
|---|---|---|
| 这句话什么意思 | LLM | 自然语言理解 |
| 怎么回复、什么语气 | LLM | 受 `context` 中的心境摘要影响，但不是被强制 |
| 要不要调用工具 | LLM / Agent 框架 | Thymos 不碰工具 |
| 这条信息值不值得记 | 宿主规则 / 可选 LLM | `AgentMemory.observe()` 用透明规则；也可传 `importance` |
| 记多深、忘多快 | Thymos | 由 importance、情绪、内分泌、复习决定 |
| 这次该想起哪几条 | Thymos | 语义 + 标签 + 情境 + 心境 + 激活 + 关联 |
| 当前“感受”是什么 | Thymos | 评价 → 情绪 → 心境；LLM 只读摘要 |
| 事实是否真实 | 外部工具 / 用户纠正 | Thymos 不保证记忆为真 |
| 能不能记、能不能删 | 用户 / 产品策略 | 必须有同意、导出、删除 |

一句话：**LLM 决定“怎么说”，Thymos 决定“此刻想起什么、忘掉什么、带着什么心情”。**

不要让 LLM 直接决定“记忆是否存在”。LLM 可以建议“这条重要”，但写入应该由宿主控制，否则它会把自己的幻觉写进长期记忆。

---

## 7. 它能让 Bot / Agent 改进什么？

### 7.1 具体改进

| 问题 | 没有 Thymos | 有 Thymos |
|---|---|---|
| 跨会话失忆 | 每次重新自我介绍 | 记住偏好、承诺、重要事件 |
| 上下文膨胀 | 把全部历史塞进 prompt | 只取 Top 3–5 条相关记忆 |
| 什么都记得 | 向量库越堆越大 | 稳定度衰减 + 巩固 + 归档 |
| 语气一成不变 | 同一套人格提示词 | 心境影响语气与回忆偏向 |
| 无法解释 | “模型觉得相关” | `components` 说明为什么想起 |
| 用户纠正后仍记错 | 旧记录一直存在 | `reappraise` / 更正记忆 / 降低显著度 |
| 情绪是假的 | prompt 写“你现在难过” | 事件评价 → 心境积分，状态连续 |

### 7.2 它不能改进什么

- **推理能力**：数学、代码、规划仍然靠 LLM；
- **世界知识**：Thymos 不提供百科知识；
- **工具使用**：搜索、订票、发邮件仍然靠 Agent 框架；
- **真实性**：记忆可能是错的；必须有纠正与删除；
- **意识/真实感受**：这是模拟状态，不是生理或临床指标。

### 7.3 最适合的场景

- 长期陪伴型 Bot：记得住偏好、承诺、重要日子；
- 角色扮演：语气随心境变化，但不失控；
- 个人助理：记住项目上下文，忘掉闲聊；
- 研究/教学：观察遗忘、情绪、环境如何耦合；
- Agent 调试：看“为什么它想起了这件事”。

不适合：医疗诊断、心理评估、用情绪操控用户、给真实的人打情绪分。

---

## 8. 三种接入方式

### 8.1 Python 进程内（推荐给 Python Bot）

```python
from thymos import AgentMemory

memory = AgentMemory(".thymos/bot.json")
context = memory.context(user_text)
# ... call LLM ...
memory.observe(user_text)  # 只记用户事实，不保存助手回复
memory.end_session()
```

优点：零网络、零依赖、状态直接可查。
缺点：Thymos 和 Bot 同进程；Thymos 崩溃会影响 Bot。

### 8.2 HTTP 侧车（推荐给 Node/Go/Java Bot）

```bash
python -m thymos --offline --state .thymos/bot.json serve --host 127.0.0.1 --port 8788
```

```bash
curl -s http://127.0.0.1:8788/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"我上次说的面试","top_k":3}'
```

优点：语言无关、进程隔离、可单独重启。
缺点：多一个进程；本地端口需要安全边界（默认只监听 127.0.0.1，无认证）。

### 8.3 MCP Server（路线图）

计划中的工具：

```text
memory.perceive(content, importance?, tags?)
memory.recall(query, top_k?)
memory.reappraise(id, delta?)
affect.snapshot()
sleep.consolidate(hours?)
```

MCP 适合 Claude Desktop、IDE Agent 等宿主；目前还没有实现，别把它当成现成功能。

---

## 9. 一个完整例子

`examples/chatbot_demo.py` 的核心就是下面这段（省略打印）：

```python
memory = AgentMemory(".thymos/chatbot_demo.json")

# Day 1
memory.observe("我不吃虾，过敏", importance=0.9)
memory.observe("我喜欢冰美式，不加糖", importance=0.85)
memory.observe("我周四面试，有点紧张", importance=0.8)

# 30 天后
memory.mind.tick(30 * 86400)
memory.end_session(7.5)

# 提问
context = memory.context("我不吃什么？", top_k=3, min_score=0.30, review=False)
reply = my_llm(context, "我不吃什么？")
```

跑起来看输出：

```bash
python examples/chatbot_demo.py
```

你会看到：

- 30 天后，重要记忆的 R 也只有 0.3 左右——**不复习就是会淡**；
- 但正确的那条仍排在 Top 3；
- 一次真正的回忆（`review=True`）把它的 R 拉回 1.00；
- “今天中午吃了拉面”这种低 importance 的记忆，R 更低。

这就是“会忘，但重要的事没忘”的工程版本。

---

## 10. 常见接错

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 每条消息都 `observe()` | 记忆膨胀、检索被闲聊淹没 | 用 importance 门槛；只记事实/偏好/承诺/情绪事件 |
| 把 `snapshot()` 全量塞 prompt | token 爆炸、模型抓不住重点 | 用 `render_context()`，Top 3–5 |
| 让 LLM 决定写不写记忆 | 幻觉进入长期记忆 | 宿主控制写入；LLM 只建议 importance |
| 两套心情同时注入 | 人格分裂 | 选一个“心情所有者”；Thymos 管记忆时，宿主心情可保留 |
| 不提供删除/导出 | 隐私风险 | 提供 `/forget`、导出、清空 |
| 声称“我真的有激素” | 误导用户 | 明确“模拟状态，不是生理指标” |
| 多个用户共用一个 state | 记忆串线 | 一个用户一个文件/命名空间 |
| 把 `.thymos` 提交到 Git | 泄露聊天内容 | `.gitignore` + 加密 |

---

## 11. 下一步

- 想直接改代码：从 `src/thymos/agent.py` 的 `AgentMemory` 开始；
- 想接自己的 LLM：把 `examples/chatbot_demo.py` 的 `fake_llm()` 换掉；
- 想接 Node/其他语言：用 HTTP 侧车；
- 想接 MCP：等 v0.3，或先自己包一层 HTTP；
- 想看设计细节：读 `docs/DESIGN.md`；
- 想看实验：读 `docs/EXPERIMENTS.md`。

如果只能记住一句话：

> **Thymos 让 Bot 的“记得”有代价、有衰减、有情绪色彩；LLM 仍然负责把话说好。**
