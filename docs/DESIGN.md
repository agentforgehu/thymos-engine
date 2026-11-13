# 设计文档：Thymos 的形式化模型

> 版本：0.1.0（本地实验版）
>
> 本文档描述代码实际在算什么，而不是“应该算什么”。所有公式都能在 `src/thymos/` 找到对应实现；所有默认参数都在 `config.py` 与 `endocrine.py` 中。参数来源分为四类：
>
> - **L（文献先验）**：有研究支持的数量级或方向；
> - **E（工程估计）**：为了得到合理行为而选的参数，未经拟合；
> - **P（占位）**：明确需要未来校准；
> - **D（推导）**：由其他参数或公式推出。

---

## 1. 总体结构

Thymos 是一个离散时间步的连续状态系统。每个时刻维护：

```text
环境 E(t)  →  内分泌 H(t)  →  心境 M(t)
                    ↑              ↓
事件 x(t) → 评价 A(t) → 情绪 V(t) → 记忆编码
                                      ↓
                              记忆库 {m_i}
                                      ↓
                         提取 R_i, B_i, score_i
                                      ↓
                          再巩固 / 巩固 / 图式
```

核心设计约束：

1. **快慢分离**：情绪（小时）、心境（天）、内分泌（分钟–天）、适应负荷（周–月）各有自己的时间常数。
2. **双向耦合**：情绪影响内分泌，内分泌影响编码/提取，提取又改写情绪电荷。
3. **可解释**：每个输出都能分解为带名字的分量。
4. **有界**：环境与内分泌的调制都有上下限，避免单一因素支配。
5. **确定性**：给定 seed、时间戳、输入序列，输出一致。

---

## 2. 评价（Appraisal）

### 2.1 数据结构

```python
@dataclass
class Appraisal:
    goal_relevance: float   # 0..1
    desirability: float     # -1..1
    novelty: float          # 0..1
    certainty: float        # 0..1
    control: float          # 0..1
    agency: str             # self | other | circumstance
    norm_compliance: float  # -1..1
    stakes: float           # 0..1
    note: str
```

### 2.2 评价 → 情绪标签

规则按优先级：

```text
if novelty >= 0.72 and |d| < 0.35:            surprise
elif d >= 0.22:
    if certainty < 0.45:                      hope
    elif agency == other and norm > 0.25:      trust
    elif stakes >= 0.62:                       excitement
    elif stakes <= 0.32:                       contentment
    else:                                      joy
elif d <= -0.22:
    if norm <= -0.45:                          disgust
    elif agency == other and control >= 0.55 and stakes >= 0.40: anger
    elif certainty < 0.42 and control < 0.55:  anxiety (d > -0.70) / fear
    else:                                      sadness
elif goal_relevance >= 0.65:                   anticipation
else:                                          neutral
```

这是 OCC 规则的极简版，不是完整 OCC 树。它的作用是给动力学一个可读的标签，而不是做情绪分类的 SOTA。

### 2.3 评价 → PAD 冲量

```text
v = d * (0.55 + 0.45 * relevance)                 ∈ [-1, 1]
a = 0.16 + 0.52 * stakes * |d| + 0.34 * novelty    ∈ [0, 1]
p = (control - 0.5) * 1.6 * (0.5 + 0.5 * stakes)   ∈ [-1, 1]
I = |d| * (0.50 + 0.50 * relevance) * (0.55 + 0.45 * stakes)
    + 0.25 * novelty * (1 - |d|)                   ∈ [0.02, 1]
```

`v/a/p` 是情绪的连续读出；`I` 是强度。标签只用于解释，动力学用 `v/a/p/I`。

### 2.4 文本启发式评价（v0.1 基线）

`estimate_appraisal(text)` 使用：

- 中英文情感词典，带词权重；
- **逐次出现的否定判断**：分句内局部窗口 + 显式例外（`特别/别人/差不多/没想到`）+ 双重否定 + 词内例外；不再用前两个字符的盲匹配；
- 显式情绪词修正：
  - “生气/愤怒”→ agency=other，control≥0.68；
  - “害怕/恐惧”→ certainty≤0.35，control≤0.25；
  - “难过/伤心”→ control≤0.30；
- 高利害词、确定性词、控制感词、其他主体词；
- 返回 `note` 记录命中的词，便于审计。

这不是语义模型，只是让 quickstart 与测试在没有 LLM 时也能跑。生产环境可以把
`LLMAppraiser`（`chat(messages) -> str`，输出结构化 `Appraisal`）传给
`AgentMemory(appraiser=...)`，或由宿主直接传 `observe(..., appraisal=...)`；
`Appraisal` 仍是契约，规则只是 fallback。详见 [LLM_APPRAISAL.md](LLM_APPRAISAL.md)。

---

## 3. 情绪与心境动力学

### 3.1 事件时的即时更新

给定情绪冲量 `(v, a, p, I)` 与反应性 `ρ`：

```text
ρ = reactivity_gain * profile.emotional_reactivity
    * (1 + 0.35*(1 - serotonin) + 0.25*max(0, cortisol - 0.35))
    * sensitivity

Δv = v * I * 0.55 * ρ
Δa = a * I * 0.45 * ρ
Δp = p * I * 0.40 * ρ

V  ← clamp(V + Δv, -1, 1)
A  ← clamp(A + Δa, 0, 1)
D  ← clamp(D + Δp, -1, 1)
M  ← clamp(M + 0.18 * Δv, -1, 1)      # 心境即时反应
stress ← clamp(stress + 0.55*max(0, -Δv) + 0.12*max(0, Δa), 0, 1)
fatigue ← clamp(fatigue + 0.05*max(0, Δa), 0, 1)
```

### 3.2 连续演化

```text
dV/dt = -V / τ_e
dA/dt = (0.16 - A) / (0.75 * τ_e)
dD/dt = -D / τ_e

M* = M_setpoint + ΔM_weather + bias_hormone + ΔM_circadian + κV
dM/dt = (M* - M) / τ_m

A* = 0.34 + ΔA_weather + bias_arousal + 0.25*(A - 0.20)
dA_mood/dt = (A* - A_mood) / (0.45 * τ_m)

stress* = clamp(0.10 + 0.28*max(0, -M) + 0.10*A_mood, 0, 1)
dstress/dt = (stress* - stress) / τ_s

dfatigue/dt = +0.035*(1 + 0.5*sleep_debt)   （清醒）
             (0.08 - fatigue) / 3.0          （睡眠）

dL/dt = 0.020*max(0, stress - 0.55) - 0.0010*L
```

默认：

| 参数 | 默认 | 单位 | 来源 |
|---|---:|---|---|
| `τ_e` 情绪时间常数 | 1.1 | 小时 | E |
| `τ_m` 心境时间常数 | 10 | 小时 | L/E（聊天尺度工程折中；见 LANGUAGE_MOOD_AUDIT） |
| `κ` 情绪→心境耦合 | 0.45 | — | E |
| `mood_reactivity` | 0.40 | — | E（单条事件可见性；敏感性扫描） |
| `τ_s` 压力时间常数 | 5 | 小时 | E |
| `L` 适应负荷时间常数 | 240 | 小时 | E（慢性） |
| `mood_setpoint` | +0.06 | — | P |

> 心境不是“情绪的平均”，而是带稳态回拉与外部偏置的慢积分。`κ` 让当下情绪能影响心境目标，但不会立刻把心境拉到情绪值。0.3.1 把 `τ_m` 从 18h 调到 10h、`mood_reactivity` 从 0.18 调到 0.40，原因是旧参数下单条辱骂/感谢在聊天回合里几乎不可见；这是工程折中，不是生理常数。详见 [LANGUAGE_MOOD_AUDIT.md](LANGUAGE_MOOD_AUDIT.md)。

---

## 4. 内分泌层

### 4.1 变量与半衰期

八个变量均为 0–1 相对单位。`SPECS` 中的半衰期只决定一阶池的时间常数：

| 变量 | 中文 | 半衰期（模型） | 基线 | 依据 |
|---|---|---:|---:|---|
| adrenaline | 肾上腺素 | 2.5 min | 0.12 | L（血浆 1–3 min） |
| noradrenaline | 去甲肾上腺素 | 2.5 min | 0.18 | L（约 2 min） |
| cortisol | 皮质醇 | 75 min（清除） | 0.32 | L（60–90 min） |
| dopamine | 多巴胺 | 25 min | 0.40 | E（相位信号秒级，tonic 代理） |
| serotonin | 血清素 | 1800 min（30 h） | 0.50 | E（慢变量代理） |
| oxytocin | 催产素 | 5 min | 0.28 | L（血浆 3–5 min） |
| melatonin | 褪黑素 | 45 min | 0.15 | L（30–60 min） |
| endorphin | 内啡肽 | 25 min | 0.22 | E（20–30 min） |

### 4.2 HPA 级联

```text
feedback = 1 - 0.55*cortisol

CRH*  = 0.08 + 1.55*stress*feedback + 0.10*circ_cortisol + 0.12*sleep_debt
ACTH* = 0.06 + 1.10*CRH
Cort* = 0.080 + 1.15*ACTH + 0.09*circ_cortisol + 0.10*sleep_debt

τ_CRH = 6 min, τ_ACTH = 18 min
τ_cort = 30 min       if Cort* > cortisol   （快速分泌）
        108 min       otherwise             （缓慢清除）
```

稳态（stress=0）：

```text
CRH  = 0.08 + 0.10c
ACTH = 0.06 + 1.10*CRH
Cort = 0.080 + 1.15*ACTH + 0.09c
     ≈ 0.250 + 0.217c
```

其中 `c = circ_cortisol ∈ [0,1]`。因此：

```text
cortisol_excess = max(0, cortisol - (0.250 + 0.217*c))
```

**这是本模型最重要的校准之一**：晨峰是正常节律，不是压力。所有“皮质醇影响心境/提取”的计算都用 `cortisol_excess`，而不是原始值。

### 4.3 其他变量的目标

```text
adrenaline*     = 0.10 + 0.66*stress + 0.22*max(0, reward) + 0.10*light
noradrenaline*  = 0.15 + 0.55*stress + 0.12*|reward| + 0.08*alertness
dopamine*       = 0.34 + 0.45*reward + 0.10*|reward|
serotonin*      = 0.46 + 0.14*light + 0.06*social - 0.16*sleep_debt - 0.10*max(0, stress-0.35)
oxytocin*       = 0.22 + 0.52*social
melatonin*      = 0.06 + 0.72*circ_melatonin - 0.22*light
endorphin*      = 0.18 + 0.40*comfort
```

每个变量按 `first_order(current, target, dt, τ)` 更新，`τ = half_life / ln 2`。

### 4.4 输出调制器

```text
encoding_gain  = 0.70 + 0.90*(adr-0.12) + 0.50*cort_excess
                 + 0.45*(da-0.40) - 0.30*max(0, 0.45-ser)        ∈ [0.55, 2.10]

retrieval_gain = 1 - 0.70*cort_excess + 0.20*max(0, adr-0.15)   ∈ [0.50, 1.35]

mood_bias      = 0.80*(ser-0.50) + 0.45*(da-0.40)
                 - 0.45*min(0.30, cort_excess)
                 + 0.32*(oxy-0.28) + 0.22*(endo-0.22)             ∈ [-0.20, 0.20]

arousal_bias   = 0.55*(adr-0.12) + 0.35*(nor-0.18)
                 - 0.40*max(0, mel-0.30)                          ∈ [-0.25, 0.40]
```

### 4.5 昼夜节律

```text
shift = early:-1.0h | intermediate:0 | late:+1.5h
h' = (hour - shift) mod 24

cortisol  = 0.5 + 0.5*cos(2π(h'-8)/24)
melatonin = 0.5 + 0.5*cos(2π(h'-3)/24)
serotonin = 0.5 + 0.5*cos(2π(h'-14)/24)
alertness = 0.5 + 0.5*cos(2π(h'-11)/24)
```

这是“峰值归一化”的相位函数，不是绝对浓度。

---

## 5. 记忆编码

### 5.1 显著度

```text
salience = clamp(0.55*importance + 0.30*intensity + 0.15*novelty)
```

`importance` 可由调用方给出；未给出时默认 0.5。`intensity` 与 `novelty` 来自评价/情绪。

### 5.2 初始稳定度

```text
S0 = base_days
     * (1 + 0.90*|v|*a + 0.30*a)
     * (1 + 0.80*salience)
     * max(0.35, encoding_gain)
S0 = clamp(S0, 0.02, 3650)
```

含义：

- 情绪价越强、唤醒越高，S0 越大；
- 个人显著度越高，S0 越大；
- 编码时的内分泌（肾上腺素、皮质醇超额、多巴胺）通过 `encoding_gain` 放大；
- 上下限防止数值爆炸或“永不遗忘”。

### 5.3 记忆项

```python
MemoryItem:
    content, tags, embedding, context
    created_at, last_access, access_times[]
    stability, difficulty, reps, lapses
    valence, arousal, salience
    associations{id: weight}
    state: buffer | episodic | semantic | archived
    labile_until, consolidated_at
```

`context` 保存地点、天气桶、温度、时段、季节、昼夜；这些既是检索线索，也是实验分组的依据。

### 5.4 关联图

编码后，与当前记忆最相似的前 `k=3` 条（余弦 ≥ 0.26）建立双向边，权重为相似度。检索时，邻居的可提取度会通过 `spread` 加成：

```text
spread_i = min(1.5, Σ_j w_ij * R_j)
```

### 5.5 结构化事实层

情景记忆回答“发生过什么”，但不擅长回答“现在什么是真的”。`FactStore` 因此单独保存高价值事实：

```text
Fact:
    subject, predicate, value
    valid_from, valid_to        # 新值关闭旧值，不删除历史
    confidence, source, memory_id
```

抽取是规则兜底（姓名、城市、地址、宠物、过敏、项目、截止、职业、朋友、作息、咖啡），不依赖 LLM；宿主也可以通过 `FactStore.add()` 写入自己的结构化事实。渲染时当前事实优先，旧值只在用户明确问“以前/搬家前”时出现；被新事实取代的旧情景记忆会被过滤，避免同一事实的两个版本同时进入 prompt。

这让上下文预算可以很小：事实块通常只有十几行，却覆盖了最容易在长对话里丢失的稳定信息。预算接口见 `AgentMemory.context(..., max_chars=...)`；600/900 字符的 50 轮对照见 [BUDGET_CONTEXT.md](BUDGET_CONTEXT.md)。

---

## 6. 提取

### 6.1 三个量

**可提取度（FSRS）**：

```text
R(t, S) = (1 + F*t/S)^D
D = -0.5
F = 0.9^(1/D) - 1 ≈ 0.2346
```

**基础激活（ACT-R）**：

```text
B_i = ln( Σ_k max(1, now - t_k)^(-d) ),  d = 0.5
```

**情境相似度**：

```text
C = 0.55*semantic + 0.20*tags + 0.10*place
    + 0.08*weather + 0.07*time_bucket
```

其中 `semantic = max(0, cosine(query_embedding, memory_embedding))`，`tags = Jaccard(query_tags, memory_tags)`，place/weather/time 为 0/1 匹配。

### 6.2 激活与概率

```text
A_i = [0.45*(B_i/6) + 1.10*C_i
       + 0.35*(mood_fit_i - 0.5)*2
       + 0.22*emotional_i
       + 0.25*spread_i] * retrieval_gain

mood_fit_i = 1 - |valence_i - mood_valence| / 2
emotional_i = 0.5*arousal_i + 0.5*|valence_i|

p_i = sigmoid((A_i - τ) / s),  τ = -0.15, s = 0.30
```

### 6.3 排序分

```text
score_i = [0.55*R_i + 0.45*p_i] * (1 + 0.12*salience_i)
```

`R` 与 `p` 分开报告。UI 中同时显示“可提取度”和“激活/概率”，让用户看到“痕迹还在但想不起来”与“痕迹淡了但线索很强”的差别。

### 6.4 复习更新

对语义或标签确实匹配的命中（`semantic > 0.01` 或 `tags > 0`）：

```text
quality = clamp(0.45 + 0.55*p, 0.25, 0.98)

若 quality < 0.35:
    S ← S * (0.25 + 0.35*R)
否则:
    spacing = 0.05 + 0.95*(1 - R)
    S ← S * (1 + 1.70 * spacing * encoding_gain * quality)

S ← clamp(S, 0.02, 3650)
reps += 1
last_access = now
labile_until = now + 10 min
state: buffer → episodic
```

注意 `spacing` 的 0.05 下限：即使 `R≈1`（集中复习），也给一点增益，避免“同一天复习完全无效”的极端；但主要增益留给“快忘了但想起来了”的时刻。

### 6.5 再巩固

```text
if now <= labile_until:  strength = 1.0
else:                    strength = 0.5

valence ← valence + (target - valence) * strength
arousal ← arousal * (1 - 0.15*strength)
S       ← S * (1 + 0.05*strength)
```

10 分钟是**工程占位**；文献中的再巩固窗口受任务、强度、年龄等影响，不应被视为普遍常数。

---

## 7. 巩固与图式

一次睡眠巩固：

1. 以 `sleeping=True` 推进 `hours`（1 小时子步）；
2. 选 `salience * R` 最高的 30 条记忆重放；
3. 每条 `S ← S*(1 + 0.35*q)`，`q` 由睡眠债与疲劳决定；
4. 若 `reps ≥ 2` 且 `S ≥ 3`，状态转为 `semantic`；
5. 对重复出现的标签，若同标签有 ≥2 条已复习记忆，生成一条 `semantic` 图式：
   - 内容为“关于「tag」的反复经验：N 次相关事件，整体感受偏正面/负面/中性”；
   - 稳定度为组内最大 `S` 的 1.3 倍；
   - 与组内前 6 条建立 0.6 的关联；
   - 每次巩固最多生成 3 条，且同标签不重复生成。

图式不是“事实抽取”，而是**情感摘要**；这是它与 Mem0/Zep 的事实记忆的差异。

---

## 8. 环境

### 8.1 昼长与季节

```text
declination = 23.44° * sin(2π(284 + day_of_year)/365)
cos(H) = -tan(lat)*tan(declination)
daylight = 24 * acos(clamp(cos(H), -1, 1)) / π
```

季节按月份与半球判断。`is_day` 由本地时间与日出/日落窗口判断。

### 8.2 舒适温度

```text
comfort = 22 + comfort_offset + min(2.0, max(0, |lat|-35)*0.08)
```

### 8.3 天气 → 心境

```text
sunshine   = clamp(radiation/650)（白天）或 1 - cloud/100
temp_stress= clamp((|apparent - comfort| - 4)/12)
rain       = clamp(precipitation/4)
pressure_drop = clamp(-pressure_trend_3h/8)
daylight_anomaly = clamp((daylight_hours - 12)/6, -1, 1)

ΔM = sensitivity * (
      0.070*sunshine*light_sensitivity
    - 0.050*temp_stress
    - 0.035*rain
    - 0.030*pressure_drop
    + 0.025*daylight_anomaly*light_sensitivity
)
ΔM = clamp(ΔM, -0.12, 0.12)

ΔA = clamp(sensitivity*(0.050*temp_stress + 0.020*rain - 0.030*pressure_drop), -0.08, 0.08)
```

### 8.4 天气 → 内分泌驱动

```text
serotonin_drive = clamp(0.35 + 0.50*sunshine + 0.15*daylight_anomaly)
melatonin_drive = clamp(0.20 + 0.60*(1-sunshine) - 0.20*clamp(radiation/500))
energy          = clamp(0.55 + 0.30*sunshine - 0.30*temp_stress - 0.20*rain)
```

这些驱动只影响目标值，不直接改写心境；实际影响还要经过内分泌与心境动力学。

### 8.5 睡眠债

```text
deficit = clamp((7.5 - sleep_hours)/3, 0, 1)
awake   = min(18, (hour - wake_hour) mod 24)
pressure= clamp(awake/18)
sleep_debt = clamp(0.55*deficit + 0.45*pressure)
```

---

## 9. 数值积分与确定性

- `Thymos.tick(seconds)` 把时间切成**最多 1 小时**的子步；不足 1 小时的尾巴单独一步。
- 每个子步依次更新内分泌、情绪/心境、时间；环境按 TTL 刷新。
- 一阶系统使用精确离散解：

```text
H(t+dt) = target + (H(t) - target) * exp(-dt/τ)
```

而不是欧拉法，避免大 `dt` 不稳定。

- 随机源只有 `random.Random(seed)`；当前版本实际未使用随机采样，但保留给未来噪声/个体差异。
- 文本嵌入是 BLAKE2b 哈希 n-gram 投影，同一文本在任何机器上得到同一向量。
- 离线天气由 `blake2b(lat, lon, date)` 派生，确定且可复现。

---

## 10. 持久化

`Thymos.to_dict()` 输出：

```json
{
  "version": 1,
  "now": 1770000000.0,
  "profile": {...},
  "config": {...},
  "affect": {...},
  "endocrine": {"levels": {...}, "crh": ..., "acth": ...},
  "environment": {...},
  "memory": [...],
  "events": [...],
  "tick_count": 42
}
```

`save()` 使用临时文件 + `os.replace()` 原子写入；`load()` 用白名单字段反序列化，未知字段被忽略，便于向前兼容。

状态文件是普通 JSON，可 diff、可备份、可手工修正。没有数据库依赖。

---

## 11. 耦合矩阵

行 → 列表示“行变量影响列变量”。`+` 正相关，`-` 负相关，`~` 非线性/条件性。

|  | 情绪 | 心境 | 内分泌 | 编码 | 提取 | 巩固 |
|---|---|---|---|---|---|---|
| 评价 | + | +（经情绪） | +（应激/奖赏） | +（显著度） |  |  |
| 情绪 |  | +（κ） | +（脉冲） | +（S0） | +（心境一致性） |  |
| 心境 |  |  | ~（经行为） | ~（情绪一致性） | +（mood-congruence） |  |
| 内分泌 | +（反应性） | +（bias） |  | +（encoding_gain） | ~（retrieval_gain） | ~（睡眠） |
| 环境 | +（小） | +（小） | +（光照/节律） | ~（情境） | +（情境线索） |  |
| 记忆 | +（回忆再巩固） | +（回忆内容） | ~（回忆触发） |  | +（关联扩散） | +（重放） |
| 睡眠 |  | +（恢复） | +（HPA/褪黑素） |  |  | +（重放/图式） |

---

## 12. 参数表（默认值与来源）

| 参数 | 默认 | 单位 | 来源 | 说明 |
|---|---:|---|---|---|
| `base_stability_days` | 0.55 | 天 | E | 中性事件的初始稳定度 |
| `emotion_stability_gain` | 0.90 | — | E | 情绪价×唤醒对 S0 的放大 |
| `salience_stability_gain` | 0.80 | — | E | 显著度对 S0 的放大 |
| `max_stability_days` | 3650 | 天 | E | 10 年上限，防止数值爆炸 |
| `min_stability_days` | 0.02 | 天 | E | 约 29 分钟 |
| `fsrs_decay` | -0.5 | — | L/D | 使 R(S,S)=0.9 |
| `actr_decay` | 0.5 | — | L | ACT-R 常用值 |
| `retrieval_threshold` | -0.15 | — | E | 激活阈值 |
| `retrieval_noise` | 0.30 | — | E | sigmoid 斜率 |
| `association_top_k` | 3 | 条 | E | 稀疏关联 |
| `association_min_similarity` | 0.26 | — | E | 避免噪声边 |
| `context_weight` | 0.55 | — | E | 语义在情境相似度中的权重 |
| `mood_congruence_weight` | 0.35 | — | L/E | 心境一致性 |
| `emotion_boost_weight` | 0.22 | — | E | 情绪唤醒的记忆优势 |
| `activation_weight` | 0.45 | — | E | ACT-R 基础激活的权重 |
| `retrievability_weight` | 0.55 | — | E | R 与 p 的排序权重 |
| `reconsolidation_window_min` | 10 | 分钟 | P | 再巩固窗口占位 |
| `mood_setpoint` | +0.06 | — | P | 轻微正向基线 |
| `mood_tau_hours` | 10 | 小时 | L/E | 心境慢变量；聊天尺度折中 |
| `mood_reactivity` | 0.40 | — | E | 事件对心境的即时冲量 |
| `emotion_tau_hours` | 1.1 | 小时 | E | 快情绪回零 |
| `stress_tau_hours` | 5 | 小时 | E | 压力恢复 |
| `allostatic_tau_hours` | 240 | 小时 | E | 慢性负荷 |
| `reactivity_gain` | 1.0 | — | P | 全局反应性 |
| `retrieval_cortisol_penalty` | 0.70 | — | L/E | 皮质醇超额对提取的抑制 |
| `weather_ttl_seconds` | 1800 | 秒 | E | 天气缓存 |
| `consolidation_replay` | 30 | 条 | E | 每次睡眠重放上限 |
| `consolidation_gain` | 0.35 | — | E | 重放对稳定度的提升 |
| `semantic_abstraction_min_reps` | 2 | 次 | E | 图式抽象门槛 |
| `semantic_abstraction_limit` | 3 | 条 | E | 每次巩固最多图式 |

完整参数见 `src/thymos/config.py`；激素参数见 `src/thymos/endocrine.py`。

---

## 13. 已知偏差与失败模式

1. **词典评价的领域偏移**：否定、反讽、网络用语、混合情绪都会误判。它是基线，不是产品级 NLP。
2. **哈希嵌入的语义弱**：同义改写可能不相似；生产环境应替换为真实嵌入。
3. **参数不可识别**：模型有几十个参数，但实验只验证了少量定性行为；没有做全局敏感性分析。
4. **情绪标签的文化偏差**：OCC 的类别与中文语境不完全对应；“生气/愤怒”的 agency 规则过于整齐。
5. **皮质醇模型过度简化**：忽略了受体、性别、口服避孕药、慢性应激、HPA 负反馈个体差异。
6. **天气效应可能被高估或低估**：合成天气不是实测；真实天气的个体差异远大于均值。
7. **再巩固窗口过短/过长**：10 分钟是占位；不同任务可能需要不同窗口。
8. **长期稳定性**：数月的连续模拟没有做数值误差累积评估；`S` 的上限是工程护栏，不是理论结论。
9. **没有用户研究**：所有“像人”的判断都是设计者视角，不能替代被试数据。
10. **不是一个意识模型**：它没有自我模型、意图、主观体验；只是把若干认知/生理变量写成动力学。

---

## 14. 下一步的形式化工作

- 用真实复习日志拟合 `S0` 与 `spacing` 参数（而不是手调）。
- 做全局敏感性分析（Sobol / Morris），找出真正影响行为的参数。
- 把评价层替换为可校准的结构化模型，并保留规则基线做对照。
- 引入不确定性：每个状态带方差，检索返回分布而不是点估计。
- 做消融实验：去掉内分泌、去掉天气、去掉再巩固，量化各自贡献。
- 与真实用户日记/经验采样数据对照，但严格区分“拟合”与“验证”。

---

## 15. 参考

机制依据见 [RESEARCH.md](RESEARCH.md)；实验结果见 [EXPERIMENTS.md](EXPERIMENTS.md)；命名与定位见 [NAMING.md](NAMING.md)。
