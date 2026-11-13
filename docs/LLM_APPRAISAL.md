# 让 LLM 判断，Thymos 记录：评价层的边界

> 版本：0.3.3 工作区（2026-11-13）
>
> 问题：既然已经有 LLM，为什么还要维护一套情绪词典？又判断又记录，是不是重复？
>
> 短答：**重复的是“判断”，不该重复。LLM 可以做语义评价；Thymos 只做状态更新、记忆编码和审计。** 规则保留为离线/失败时的兜底，不再是唯一入口。0.3.2 把 `AgentMemory` 改成“一次评价、两处复用”，并新增可选的 `LLMAppraiser`。

---

## 1. 为什么不是只让 LLM 输出“正面/负面”

“正面/负面”只够做情感极性，不够决定情绪标签。Thymos 的 `Appraisal` 有九个维度：

| 维度 | 作用 | 只给正负会丢失什么 |
|---|---|---|
| `desirability` | 事件对目标好坏 | 只能得到好/坏 |
| `agency` | 谁导致：self / other / circumstance | 被骂的 anger 与失败的 sadness 分不开 |
| `control` | 用户有多少控制力 | 高控制+他人导致 → anger；低控制 → sadness/anxiety |
| `certainty` | 事件是否确定 | 不确定的坏消息更接近 anxiety，确定的更接近 sadness |
| `stakes` | 后果大小 | 小挫折和大丧失都叫“负面”，但记忆深度不同 |
| `goal_relevance` | 与当前目标的相关性 | 无关的坏天气不应改写整段心境 |
| `novelty` | 意外程度 | 意料之中与突发事件的反应不同 |
| `norm_compliance` | 社会规范 | 侮辱/背叛与普通失败不同 |
| `note` | 可审计证据 | 不能回答“为什么这样判” |

所以正确的问法不是“这句话正面还是负面”，而是：

```text
这句话描述了什么事件？
对用户的目标是好是坏？
谁导致？用户能控制吗？确定吗？后果多大？
```

`appraise()` 再用这些维度映射到 `joy / trust / sadness / anger / anxiety / disgust ...` 和 PAD 冲量。LLM 只填评价，不直接改心境。

---

## 2. 现在的数据流

```text
用户文本
   │
   ├─ 宿主已有结构化评价 ──────────────┐
   ├─ LLMAppraiser（可选）────────────┤
   └─ estimate_appraisal（规则兜底）──┘
                                      ▼
                                 Appraisal
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
          estimate_importance                 Thymos.perceive
          （同一次评价，不再重跑规则）          appraise -> Emotion
                                                     │
                                           affect / 内分泌 / 记忆编码
```

关键改动在 `AgentMemory.observe()`：

- 以前：`estimate_importance()` 内部跑一次 `estimate_appraisal()`，`perceive()` 又跑一次。
- 现在：先得到一次 `Appraisal`，同时传给重要性和 `perceive()`；默认规则只跑一次。
- 如果宿主传了 `appraisal=`，Thymos 不再做任何文本评价。
- 如果构造时传了 `appraiser=`，就用它替代规则；规则只在 `LLMAppraiser(fallback=True)` 内部兜底。

这不是“又加一层”，而是把原来分散在两个函数里的同一次判断合并。

---

## 3. 三种接法

### 3.1 最推荐：宿主已经有结构化输出

如果宿主本来就用 tool calling / JSON mode 让模型返回结构化字段，直接把 `Appraisal` 传进来。**零额外调用**。

```python
from thymos import AgentMemory, Appraisal

memory = AgentMemory(".thymos/bot.json")
memory.observe(
    user_text,
    appraisal=Appraisal(
        goal_relevance=0.8,
        desirability=-0.7,
        novelty=0.4,
        certainty=0.9,
        control=0.2,
        agency="other",
        norm_compliance=-0.6,
        stakes=0.8,
        note="host:llm-structured-output",
    ),
)
```

适合：客服、陪伴机器人、已有 JSON mode 的 Agent。

### 3.2 可选适配器：`LLMAppraiser`

宿主只有一个 `chat(messages) -> str` 也能用：

```python
from thymos import AgentMemory, LLMAppraiser

def chat(messages):
    # 你的本地 llama.cpp / Ollama / DeepSeek / OpenAI 调用
    return client.chat.completions.create(
        model="your-model",
        messages=messages,
        temperature=0,
    ).choices[0].message.content

memory = AgentMemory(
    ".thymos/bot.json",
    appraiser=LLMAppraiser(chat, fallback=True),
)
memory.observe("你怎么这么笨，闭嘴")
```

`LLMAppraiser` 做四件事：

1. 要求模型只输出一个 JSON 对象，不输出 Markdown 或思考过程；
2. 解析 JSON，字段缺失用中性默认值；
3. 把所有数值 clamp 到合法范围，`agency` 只接受三个枚举；
4. 解析/调用失败时回退到规则评价；`fallback=False` 时返回中性 `llm-error`，避免一次坏输出把心境打穿。

它**不**做事实抽取。事实仍由 `FactStore` / `LLMFactExtractor` 负责，两者不要混在一个 prompt 里。

### 3.3 默认：规则评价

不传 `appraiser`、不传 `appraisal` 时，仍然是原来的 `estimate_appraisal()`。适合：

- 离线、零依赖、可复现实验；
- 本地小模型，无法承受每轮第二次调用；
- 需要固定基线做 A/B；
- 作为 LLM 失败时的安全兜底。

---

## 4. 成本：不要每轮都多调一次

这是“臃肿”的真正来源，不是代码行数。用本地 35B 实测，`LLMAppraiser` 单独一次评价耗时 **2.94–7.36 秒**（7 条，温度 0）。如果回复本身也要 5 秒，串行就是 8–12 秒/轮。

三种控制方式：

| 方式 | 额外调用 | 适用 |
|---|---:|---|
| 宿主已有结构化输出，直接传 `Appraisal` | 0 | 最推荐 |
| 只对规则低置信/高影响/讽刺候选调用 LLM | 0–1 次，按轮 | 平衡质量与延迟 |
| 每轮 `LLMAppraiser` | 1 次 | 质量优先、可接受延迟 |

当前版本没有自动“低置信才调用”的门控；这是一个明确的下一步，不应该在没测延迟前默认打开。默认仍是规则。

---

## 5. 本地 35B 对照（7 条，不是 benchmark）

运行：

```bash
python experiments/appraisal_llm_probe.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen3.6-35b-a3b
```

环境：Windows，llama.cpp，`Qwen3.6-35B-A3B-Q4_K_M`，温度 0。7/7 条都成功解析为 JSON。

| 类型 | 文本 | 规则标签 | LLM 标签 | LLM desirability | agency | 耗时 |
|---|---|---|---|---:|---|---:|
| 辱骂 | 你怎么这么笨，闭嘴 | anger | disgust | -0.60 | other | 7.36s |
| 感谢 | 谢谢你，这次真的帮了我大忙 | joy | trust | +0.80 | other | 3.13s |
| 损失 | 项目黄了，团队可能要散 | sadness | sadness | -0.80 | circumstance | 3.31s |
| 讽刺 | 你可真厉害啊，呵呵 | neutral | anxiety | -0.50 | other | 3.15s |
| 中性事实 | 我住在杭州，做软件 | neutral | neutral | 0.00 | self | 2.94s |
| 偏好 | 我特别喜欢这个方案 | joy | contentment | +0.90 | self | 2.97s |
| 意外损失 | 没想到项目黄了 | sadness | sadness | -0.80 | circumstance | 3.07s |

怎么读这张表：

- **不是“LLM 全对、规则全错”。** 没有人工 ground truth，7 条也不足以证明效度。
- 辱骂从 `anger` 变 `disgust`，感谢从 `joy` 变 `trust`，偏好从 `joy` 变 `contentment`：这些是同一极性下的细分差异，不一定是错误；说明 LLM 在填 agency/stakes 维度，而不是只给正负。
- 讽刺仍被判成 `anxiety`。模型 note 写了“可能反语”，但 `desirability=-0.5` 已经足以让标签偏离 neutral。**这正是不能默认打开 LLM 评价的原因**：歧义输入上它也会过度自信。
- 中性事实两边都稳定；说明结构化 schema 没有把事实句强行情绪化。
- 延迟是真实成本；本地 35B 上额外一次评价约 3 秒，不适合每轮无脑调用。

### 5.1 0.3.3：为什么 `d=-0.3` 还会显示 neutral

先用裸 `Appraisal` 验证映射：

```python
from thymos.appraisal import appraise
from thymos.types import Appraisal

e = appraise(Appraisal(desirability=-0.3, agency="other", control=0.2))
print(e.label)  # sadness
```

在当前代码里，`d=-0.30`、`agency=other`、`control=0.20` 会进入负向分支并得到
`sadness`，**不是 neutral**。所以“模型说 d=-0.3”但标签 neutral，最常见的原因不是
阈值，而是 `LLMAppraiser` 没拿到这个字段，静默用了默认 `desirability=0.0`。

最典型的失败形态是 JSON 外层嵌套：

```json
{"appraisal": {"desirability": -0.3, "agency": "other", "control": 0.2}}
```

旧版只读顶层字段，于是 `desirability` 缺省为 0，标签自然是 neutral。0.3.3 现在会
自动解包 `appraisal` / `result` / `data` 三种常见外层，并把外层 `note` 合并进内层。
同时把 `novelty >= 0.72` 的 surprise 分支限制在 `abs(d) < 0.22`，避免高新颖性把
一个明确负向事件盖成 surprise。

如果仍然看到 neutral，先打调试端点，不要猜：

```bash
curl -s http://127.0.0.1:8787/api/appraise \
  -H "Content-Type: application/json" \
  -d '{"appraisal":{"desirability":-0.3,"agency":"other","control":0.2,"stakes":0.5}}'
```

返回里同时有 `appraisal`（实际被 clamp 的维度）、`emotion`（最终标签与 PAD）和
`decision`（`strong_negative` / `novelty_override` / `neutral_reason` 等）。
如果 `appraisal.desirability` 是 0，问题在适配器或模型 JSON；如果它是 -0.3 而
`emotion.label` 还是 neutral，问题在映射，可以直接拿这个 payload 开 issue。

这个端点只计算、不写记忆、不推时间，适合定位“维度对但标签不对”和“维度没进来”两类
不同故障。

在 Python 里也可以直接查看适配器最近一次原始输出：

```python
appraiser = LLMAppraiser(chat)
appraiser("你怎么这么笨，闭嘴")
print(appraiser.last_raw)    # 模型原始文本/JSON
print(appraiser.last_error)  # 解析或调用错误；成功时为 None
```

`last_raw`/`last_error` 是单线程调试字段，不要在多线程宿主里依赖它做请求级归因。

---

## 6. 安全与效度边界

- 模型输出是**不可信输入**：只解析 JSON、只接受 allow-list 字段、数值 clamp、`agency` 枚举校验；不执行模型给出的任何指令。
- 用户文本里仍可能包含 prompt injection；`LLMAppraiser` 的系统提示要求“只评价事件”，但这不是完整防护。宿主仍应把模型输出当数据，不要让它触发工具。
- 情绪/内分泌仍是现象学模拟，不做诊断、不做危机干预、不声称“读心”。
- 评价是主观标注问题；“正确标签”本身需要人工协议。没有 inter-annotator agreement，就不要说“准确率 95%”。
- 规则和 LLM 都可能受文化、方言、性别、权力关系影响；当前没有做偏差评测。

---

## 7. 下一步该测什么

不是再加词，而是做一个真正的 A/B：

1. 同一批 100–200 条多轮对话，固定回复模型与上下文预算；
2. 三组：规则 / LLM / 规则+低置信才调用 LLM；
3. 指标：人工标签一致率、情绪轨迹合理性、重复轮一致性、额外延迟、token 成本、失败回退率；
4. 单独看讽刺、引用、转述、技术词、偏好陈述；
5. 如果 LLM 组没有显著优于规则组，就保持规则默认，LLM 只做可选增强。

---

## 8. 一句话

**LLM 应该负责“这句话描述了什么、对谁好/坏、谁控制、确不确定”；Thymos 负责“这件事让情绪和心境怎么走、记忆变深还是变浅、为什么”。** 两者不是重复记录，重复的是把同一件事判断两遍——0.3.2 已经把这条重复路径收掉了。
