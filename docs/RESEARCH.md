# 研究综述：记忆、情绪、内分泌与环境的可计算耦合

> 目的：为 Thymos 的每个机制找到文献依据，同时把“有证据的结论”“有争议的模型”“纯工程假设”分开。
>
> 一句话结论：**记忆的遗忘曲线、评价到情绪的映射、急性应激对编码/提取的双向影响、以及光照/天气的小幅情绪效应，都有可用的研究基础；但没有一个现成模型能同时覆盖它们。Thymos 的价值不在于“证明”某个理论，而在于把若干可检验的假设放进同一个可复现的动力学系统。**

---

## 0. 五条需要先记住的结论

1. **遗忘不是单一指数。** Ebbinghaus 的“节省量”曲线在双对数坐标下更接近幂律或拉伸指数；自由回忆、再认、节省量的曲线形状不同。把某一条曲线当成“人类遗忘定律”是过度概括。[1][2][3][4][5]
2. **复习的分布比次数更重要。** 间隔效应、测试效应、睡眠巩固和再巩固都有实验支持，但都有边界条件；“间隔越久越好”“回忆一定会改写记忆”都是错的。[6][7][8][9][10]
3. **评价维度可以映射到情绪，但离散标签是降维。** OCC 的 appraisal 结构适合工程实现；PAD/环形模型适合连续读出。两者都不是“情绪的真实分类”。[13][14][15][16]
4. **应激对记忆是双向的。** 急性应激/皮质醇可能增强情绪性材料的编码，却损害需要前额叶控制的提取；效应取决于时间点、剂量、性别、情绪效价和任务。[26][27][28][29]
5. **天气对情绪的平均效应很小，个体差异很大。** 光照/季节对部分人群（尤其高纬度、季节性情绪障碍）更强；把“下雨=难过”写死是伪科学。[35][36][37][38][39][40]

---

## 1. 记忆：从遗忘曲线到可计算调度

### 1.1 Ebbinghaus 的原始数据

Ebbinghaus（1885）用无意义音节做自身实验，测量“节省量”（relearning savings）：重新学习到同样标准所需时间比初次学习少多少。经典数据点大致为：

| 距初次学习 | 节省量 |
|---|---:|
| 20 分钟 | 58.2% |
| 1 小时 | 44.2% |
| 8.8 小时 | 35.8% |
| 1 天 | 33.7% |
| 2 天 | 27.8% |
| 6 天 | 25.4% |
| 31 天 | 21.1% |

这组数据有几个常被忽略的特征：

- **起点不是 100%**：20 分钟时已经只剩 58%，因为测量的是节省量而非“记得/不记得”。
- **长尾很厚**：31 天仍有 21%，说明存在一个缓慢消退的成分。
- **这是单被试、无意义材料、节省量范式**，不能直接外推到日常事件记忆。

Murre 与 Dros（2015）复现并重新分析了这组数据，认为幂律（power law）比简单指数更能描述其形状。[2]

### 1.2 指数、幂律、拉伸指数

三种常见形式：

```text
指数：      R(t) = a + (1-a) * exp(-t / τ)
幂律：      R(t) = (1 + t / (b * S))^(-c)
拉伸指数：  R(t) = a + (1-a) * exp(-(t / τ)^β)
```

- **指数**假设每个时刻的遗忘风险与剩余量成正比，数学上最简洁，但通常低估长尾。
- **幂律**在自由回忆和日常事件记忆中常拟合更好；Anderson 与 Schooler（1991）把“记忆概率随需求/近因的幂律衰减”解释为环境统计结构的反映。[3]
- **Wixted 与 Ebbesen（1991）** 比较指数与幂律，发现幂律更适合保存函数。[4]
- **Averell 与 Heathcote（2011）** 进一步指出，曲线形状取决于测量指标与聚合方式，不能脱离任务谈“正确曲线”。[5]

**Thymos 的选择**：核心使用 FSRS 的幂律族，因为它同时给出“稳定度”这一可操作参数；但在实验 1 中明确比较三种形式，并报告 FSRS 对 Ebbinghaus 节省量拟合较差（RMSE 0.190），避免把调度器当成心理学定律。

### 1.3 FSRS：把“稳定度”变成可调度量

FSRS（Free Spaced Repetition Scheduler）是 Anki 社区发展出的开源调度算法。其核心 retrievability 形式为：

```text
R(t, S) = (1 + FACTOR * t / S)^DECAY
FACTOR  = 0.9^(1/DECAY) - 1
```

当 `DECAY = -0.5` 时，`FACTOR = 0.9^-2 - 1 ≈ 0.2346`，且 `R(S, S) = 0.9`：

- `S`（stability）被定义为**可提取度降到 90% 所需的天数**；
- 这比“半衰期”更适合间隔重复，因为调度目标通常是 90% 而不是 50%；
- 复习成功会提高 `S`，失败会降低 `S`；间隔越接近遗忘边缘，成功复习带来的增长越大（spacing effect 的算法化）。

FSRS 本身是一个工程模型，参数由大规模复习日志拟合，不是从神经机制推导出来的。Thymos 借用其曲线与稳定度语义，但把 `S` 的初始值改为由情绪显著度、唤醒和内分泌编码增益共同决定。[12]

### 1.4 ACT-R 的基础激活

ACT-R 用基础水平激活 `B_i` 描述一条记忆痕迹的可用性：

```text
B_i = ln( Σ_k t_k^(-d) )
```

- `t_k` 是第 k 次呈现/提取距现在的时间；
- `d` 通常取 0.5；
- 频率越高、越近，`B_i` 越大；
- 提取概率由 `P = sigmoid((B_i - τ) / s)` 给出，`τ` 是阈值，`s` 是噪声。

它与 FSRS 的区别很重要：

- **FSRS 的 R** 问“痕迹还剩多少”；
- **ACT-R 的 B** 问“此刻有多容易被唤起”。

一条记忆可以“痕迹很强但线索不对”，也可以“痕迹很弱但刚刚被反复提到”。Thymos 同时计算两者，并在 UI 中分开显示，避免把“记不住”和“想不起”混为一谈。[11]

### 1.5 间隔、测试、睡眠、再巩固

**间隔效应**：Cepeda 等（2006）综合大量 verbal recall 研究，发现分布练习优于集中练习，且最优间隔随目标保持时间增长。它不是“越久越好”，而是“与你要保留多久相匹配”。[6]

**测试效应**：Roediger 与 Karpicke（2006）发现，提取练习比重复阅读带来更好的长期保持，即使短期感觉更差。[7]

**睡眠巩固**：McGaugh（2000）与 Rasch、Born（2013）综述了巩固与睡眠重放在陈述性/程序性记忆中的作用；睡眠不是简单“保存”，而是重组与抽象。[9][10]

**再巩固**：Nader 等（2000）显示，已巩固的恐惧记忆在提取后需要蛋白质合成才能再次稳定；这开启了“回忆会改写记忆”的研究。但再巩固不是无条件发生：记忆必须被“提醒”到不稳定状态，且存在年龄、强度、边界条件等限制。Thymos 的 10 分钟可改写窗口是**工程占位**，不是普遍生理常数。[8]

### 1.6 对设计的含义

- 用 `S` 表示稳定度，用 `R` 表示当前可提取度，用 `B` 表示线索相关可用性；三者分开。
- 复习不是“+1 次”，而是根据当时的 `R` 决定增益；在 `R` 接近 1 时增益很小。
- 记忆不是只读文件；提取后进入短暂可改写窗口，情绪电荷可以被重新评价。
- 睡眠巩固要能提升稳定度、并把重复出现的主题抽象为语义图式。
- 任何“遗忘曲线”都必须标注测量范式；不能拿节省量曲线直接预测日常事件回忆。

---

## 2. 情绪与心境：评价、维度与时间尺度

### 2.1 OCC 评价模型

Ortony、Clore 与 Collins（1988）提出 OCC 模型：情绪不是由事件本身直接触发，而是由对事件的**评价**触发。核心维度包括：

- 目标相关性（这件事与我的目标有关吗？）
- 可取性（对我好还是坏？）
- 新异性/意外性
- 确定性
- 控制感
- 责任/agency（是我、他人还是环境造成的？）
- 规范/道德符合度

OCC 用规则树把评价模式映射到 22 类情绪（joy、distress、hope、fear、anger、gratitude 等）。它的优势是**可审计**：每个情绪标签都能追溯到评价维度；劣势是离散标签过于整齐，且文化差异大。[13]

### 2.2 Russell 环形模型与 PAD

Russell（1980）用“愉悦—唤醒”二维环形空间组织情绪词；Mehrabian 与 Russell（1974）进一步提出 PAD（pleasure-arousal-dominance）三维模型。PAD 的优势：

- 连续、可插值、可随时间衰减；
- 适合做动力学状态，而不是每帧重新分类；
- dominance 能区分“愤怒”（高支配）与“恐惧”（低支配）。

Thymos 的做法是：**评价 → 离散标签（用于解释） + PAD 冲量（用于动力学）**。标签是读出，不是状态本身。[14][15]

### 2.3 ALMA、EMA、WASABI

- **ALMA**（Gebhard, 2005）是分层情绪模型：长期人格、中期心境、短期情绪；用 PAD 空间和情绪强度衰减。[18]
- **EMA**（Marsella & Gratch, 2009）把 appraisal 做成过程模型，强调评价、应对、再评价的循环。[17]
- **WASABI**（Becker-Asano & Wachsmuth, 2010）用 PAD 加上“认知评价”产生 primary/secondary emotions，并显式建模情绪强度与衰减。[19]

这些系统证明了“评价 + 连续维度 + 衰减”在虚拟人里可行。它们的共同短板是：**记忆通常只是一个查找表，没有遗忘曲线；环境通常只是背景贴图。**

### 2.4 情绪持续多久？

Verduyn 等（2009）用经验采样研究情绪持续时间，发现持续时间与事件的重要性、评价和应对方式有关，短则几分钟，长则数小时甚至更久；不存在一个统一的“情绪半衰期”。[20]

Thymos 的默认值（5–480 分钟，按标签区分）是**工程分层**：

- surprise 5 分钟（极短）；
- fear 12 分钟、anger 25 分钟（高唤醒、快速）；
- joy 45 分钟；
- sadness 300 分钟（低唤醒、拖尾）；
- trust 480 分钟（慢变量）。

这些数字用于让不同情绪在动力学上可区分，不应被引用为“科学常数”。

### 2.5 心境：慢变量与情绪惯性

心境不是“情绪的平均值”，而是更慢、更弥散、带有稳态回拉的状态。Kuppens 等（2010）用“情绪惯性”描述情绪状态对前一刻状态的依赖；Houben 等（2015）的元分析发现，情绪惯性/变异性与心理健康相关，但效应受测量层次影响。[21][22]

Thymos 用一阶动力学：

```text
dV/dt = -V / τ_e                      （快情绪回零）
dM/dt = (M* - M) / τ_m                （慢心境回目标）
M*    = setpoint + weather + hormone + circadian + κV
```

- `τ_e` 默认 1.1 小时；
- `τ_m` 默认 18 小时；
- `κ` 让当下情绪以 45% 的权重进入心境目标；
- 事件还会给心境一个即时的 18% 冲量，避免“心境永远慢半拍”的不真实感。

这套结构不是从某个实验直接拟合的，而是把“快情绪/慢心境/稳态”三个定性事实写成最小动力学。

### 2.6 对设计的含义

- 不要用一次分类结果代表用户“此刻的情绪”；保留连续状态和不确定性。
- 离散标签只用于解释和交互，动力学用 PAD。
- 情绪与心境必须有不同时间常数；否则系统要么过度反应，要么像木头。
- 评价维度要能被用户/开发者检查和覆盖；不能只给一个黑箱分数。
- 文化差异与个体差异必须作为参数，而不是默认值。

---

## 3. 内分泌：把“激素”降维成可计算的调制器

> **重要边界**：Thymos 的激素变量是 0–1 的**相对单位**，不是血液/唾液浓度。它们用于表达“相对高低与时间尺度”，不能用于诊断、用药或任何医学推断。

### 3.1 HPA 轴与皮质醇

下丘脑—垂体—肾上腺（HPA）轴：

```text
CRH → ACTH → cortisol
```

- 皮质醇有昼夜节律：清晨觉醒后 30–45 分钟出现觉醒反应（CAR），随后下降，夜间最低。
- 急性心理应激可在 20–40 分钟内抬高皮质醇，通常在 60–90 分钟内回落；反应强度受“不可控性”和“社会评价威胁”调节（Dickerson & Kemeny, 2004）。[23]
- 血浆皮质醇半衰期约 60–90 分钟；唾液皮质醇反映游离部分。[24]
- 糖皮质激素通过基因组与非基因组机制影响海马、杏仁核和前额叶；慢性升高与海马树突萎缩、前额叶功能受损相关。[25][26]

Thymos 用三级级联 + 负反馈 + 非对称时间常数（快速分泌 30 分钟、缓慢清除 108 分钟）来近似，而不是直接给皮质醇一个“压力值”。

### 3.2 儿茶酚胺

- **肾上腺素**：肾上腺髓质释放，半衰期约 1–3 分钟；与急性警觉、心率、能量动员相关。
- **去甲肾上腺素**：交感神经与蓝斑，半衰期约 2 分钟；与唤醒、注意、警觉相关。

在模型中，它们是最快的变量，负责“事件发生后的前几分钟”。这也是为什么急性应激实验里，肾上腺素先升、皮质醇后升。

### 3.3 多巴胺、血清素、催产素、褪黑素、内啡肽

| 变量 | 主要功能（简化） | 半衰期/时间尺度 | Thymos 用法 | 证据强度 |
|---|---|---|---|---|
| 多巴胺 | 奖励预测误差、动机 | 相位信号秒级；-tonic 更慢 | 奖励/动机增益 | 中（机制复杂，模型高度简化） |
| 血清素 | 情绪稳定、睡眠/食欲 | 血小板/外周约小时级；脑内周转更慢 | 慢心境偏置 | 中（外周≠中枢） |
| 催产素 | 社会联结、信任 | 血浆 3–5 分钟；中枢效应更久 | 社交事件增益 | 中（社会效应情境依赖强） |
| 褪黑素 | 昼夜节律、睡眠压力 | 30–60 分钟 | 夜间/光照抑制 | 强（节律生物学） |
| 内啡肽 | 镇痛、舒适/奖赏 | 20–30 分钟 | 舒适/安抚 | 弱–中（外周测量困难） |

**必须说明**：多巴胺的“25 分钟半衰期”不是生理常数，而是工程上的“紧张度代理”；血清素的 30 小时也是为了产生慢变量。把它们当成血浆半衰期会误导。[30][31][32][33]

### 3.4 皮质醇对记忆的双向影响

这是 Thymos 最需要谨慎处理的部分。

- **编码增强**：情绪唤醒与杏仁核—海马相互作用可增强情绪性材料的编码与巩固（Cahill & McGaugh, 1998）。[28]
- **提取损害**：急性应激与高皮质醇常损害前额叶依赖的提取/执行功能，尤其在高负荷、需要控制搜索的任务中（Arnsten, 2009；de Kloet et al., 2005）。[26][29]
- **时间点决定方向**：应激后立即提取可能受儿茶酚胺影响，延迟提取更受皮质醇基因组效应影响；情绪效价、性别、习惯化都调节结果（Joëls et al., 2011）。[27]
- **不是“压力越大记性越差”**：慢性应激、睡眠剥夺、抑郁症状等混杂因素很多；实验室效应量通常中小。

Thymos 的实现：

```text
encoding_gain = 0.70 + 0.90*(adr-0.12) + 0.50*cortisol_excess + 0.45*(da-0.40) - 0.30*max(0, 0.45-ser)
retrieval_gain = 1 - 0.70*cortisol_excess + 0.20*max(0, adr-0.15)
```

其中 `cortisol_excess` 是**扣除昼夜预期后的超额**，避免把正常晨峰误判为压力。两个增益都被限制在合理区间（0.55–2.10、0.50–1.35）。

### 3.5 适应负荷（allostatic load）

McEwen 与 Stellar（1993）提出 allostasis：适应不是“回到固定点”，而是通过改变设定点来维持稳定；长期反复激活会累积“适应负荷”，增加疾病风险。[34]

Thymos 用一个 0–1 的慢变量 `allostatic_load` 表示：

```text
dL/dt = 0.020 * max(0, stress - 0.55) - 0.0010 * L
```

它不直接产生症状，只作为“慢性压力背景”影响内分泌基线与心境偏置。这是一个**叙事性/结构性变量**，不是临床指标。

### 3.6 对设计的含义

- 区分“正常节律”与“应激超额”；否则每天早晨都会“压力爆表”。
- 激素用相对单位，UI 必须标注“相对水平”，不能给绝对值。
- 不同激素用不同时间常数；不能让八个变量同一个速度。
- 编码与提取用不同增益；急性应激可能“记得更牢但想不起来”。
- 所有内分泌参数都要可关闭、可替换、可审计。

---

## 4. 环境：天气、光照、季节与地点

### 4.1 天气对情绪：平均很小，个体差异很大

- **Howarth 与 Hoffman（1984）** 发现湿度、温度、日照等与心境有相关，但效应不大，且受人格与季节调节。[35]
- **Denissen 等（2008）** 用日记法发现，天气对每日心境的平均效应很小；阳光/温度的作用被人格与季节调节，而非普遍规律。[36]
- **Kööts 等（2011）** 在爱沙尼亚样本中发现，天气对积极/消极情绪的影响存在但较弱，且与个体敏感性有关。[37]
- **Klimstra 等（2011）** 强调“有人晴天更好，有人阴天更好”的个体差异；平均效应会掩盖方向相反的子群。[38]

因此 Thymos 把天气写成**有界的小偏置**（±0.12），乘以个体 `weather_sensitivity`，并允许完全关闭。我们不写“下雨=悲伤”的规则。

### 4.2 光照、血清素与季节性

- **Lambert 等（2002）** 发现健康人脑内血清素周转随季节/日照变化，夏季较高、冬季较低；这为“光照→血清素→心境”提供了机制线索。[39]
- **Rosenthal 等（1984）** 描述了季节性情绪障碍（SAD）：高纬度、冬季、光照不足是风险因素；光疗有效。[40]
- 但对大多数人，季节效应是**亚临床**的；不能把 SAD 的机制套到所有人。

Thymos 用昼长、太阳辐射、季节和纬度计算 `daylight_anomaly` 与 `serotonin_drive`，影响血清素目标与褪黑素目标；效应同样有界。

### 4.3 热、舒适温度与认知

- 高温与攻击性/烦躁有相关（Anderson, 1989 的元分析），但效应受情境、文化、空调等调节。[41]
- 热应激会损害复杂认知任务表现，尤其是持续注意与工作记忆（Hancock & Vasmatzidis, 2003）。[42]
- 舒适温度存在**地域适应**：长期生活在热带的人，中性温度更高（de Dear & Brager, 2002 的适应性舒适模型）。[43]

Thymos 用 `comfort_temperature = 22 + comfort_offset + latitude_adaptation`，计算 `temp_stress`，并且只让它贡献很小的负偏置。

### 4.4 气压与头痛/疲劳

“气压下降导致头痛/关节痛”在民间很常见，但证据质量参差，效应小且异质。Thymos 只保留一个极小的 `pressure_drop` 项，并在文档中标为“弱证据/可关闭”，不把它当作确定机制。

### 4.5 Open-Meteo 与隐私

- Open-Meteo 提供免费、无需 API Key 的天气 API，支持 current/daily 字段与 WMO 天气代码；非商业使用免费，商业使用需查看其条款。[44]
- 请求只需要经纬度；Thymos 默认发送**保留两位小数**的坐标（约 1 km 精度），不发送用户 ID。
- 结果缓存到本地 `.thymos/weather_cache.json`，默认 30 分钟 TTL；离线时使用确定性合成天气。
- 若用户不愿联网，`--offline` 或 `ThymosConfig(offline=True)` 完全禁用网络。

### 4.6 对设计的含义

- 环境效应必须小、可关闭、可解释；不能让天气“决定”一个人的情绪。
- 地点/天气只作为检索情境线索与心境偏置，不作为诊断。
- 高纬度/季节性风险人群需要单独的敏感性参数，而不是全局默认。
- 任何位置数据都要最小化、可缓存、可删除。

---

## 5. 既有 AI 记忆与情绪系统

### 5.1 Generative Agents 的 memory stream

Park 等（2023）的 Generative Agents 用“记忆流 + 检索 + 反思”：

```text
retrieval_score = α_recency * recency + α_importance * importance + α_relevance * relevance
recency = 0.995 ^ hours_since_access
```

- 近因衰减是指数式的，不是 FSRS 的稳定度；
- importance 由 LLM 打分；
- relevance 用嵌入余弦；
- 反思会把多条记忆抽象成更高层结论。

它证明了“记忆流 + 反思”能产生可信的日常行为，但没有显式的内分泌、心境动力学或真实天气耦合。[45]

### 5.2 MemGPT / Letta

MemGPT（Packer et al., 2023）把 LLM 当作操作系统：主上下文（main context）与外部存储（archival）之间分页；模型通过函数调用自己管理记忆。Letta 是其工程化延续。

- 优势：上下文管理、可扩展、Agent 自编辑；
- 局限：记忆的“重要性”主要由模型判断，没有遗忘曲线；没有情绪/内分泌层；回忆不会改变记忆的情感电荷。[46]

### 5.3 Mem0

Mem0（Chhikara et al., 2025）从对话中抽取事实/偏好，做去重、合并、图记忆，强调生产可用与低延迟。

- 优势：抽取与冲突消解、图结构、部署成熟；
- 局限：以“事实正确”为目标，不建模“感受如何随时间变化”；没有环境耦合。[47]

### 5.4 Zep / Graphiti

Zep（Rasmussen et al., 2025）用时序知识图（bi-temporal）组织对话与实体关系，支持时间有效性查询。

- 优势：时间感知、关系推理、企业级；
- 局限：时间主要是“事实有效期”，不是“可提取度”；没有情绪动力学。[48]

### 5.5 A-MEM

A-MEM（Xu et al., 2025）受 Zettelkasten 启发，让每条记忆自动生成关键词、标签、上下文与链接，并在新记忆到来时动态更新旧记忆的网络。

- 优势：自组织、关联丰富；
- 局限：链接由 LLM 生成，缺少可解释的遗忘/情感机制；没有生理层。[49]

### 5.6 ALMA / WASABI 等情感 Agent

ALMA 与 WASABI 证明了 appraisal + PAD + 衰减在虚拟人中可行；但它们通常：

- 记忆是脚本/查找表，没有遗忘曲线；
- 环境是静态背景；
- 没有把内分泌作为独立层；
- 很少提供可复现实验与开源实现。

### 5.7 空白在哪里

| 维度 | 现有系统 | Thymos 的尝试 |
|---|---|---|
| 遗忘 | 近因衰减或上下文淘汰 | FSRS 稳定度 + ACT-R 激活 |
| 情绪 | 分类/标签或简单 valence | OCC 评价 + PAD + 快慢双尺度 |
| 内分泌 | 基本缺失 | 8 变量 + HPA 级联 + 昼夜节律 |
| 环境 | 基本缺失 | 天气/光照/季节/地点，效应有界 |
| 再巩固 | 基本缺失 | 提取后可改写窗口 |
| 可复现 | 少见 | 确定性 + 104 项测试 + 5 个实验 |
| 可解释 | 部分 | 每次检索给出 components |

Thymos 不宣称比这些系统“更强”。它补的是**动力学与可解释性**这一块，且明确是研究沙盒。

---

## 6. 从文献到设计的十条推论

1. **把“记得”拆成三个量**：痕迹稳定度、当前可提取度、线索相关激活。
2. **复习增益取决于当时的可提取度**，而不是简单计数。
3. **情绪是快变量，心境是慢变量**；两者必须有不同时间常数。
4. **评价维度要显式**，标签只是读出；允许用户覆盖评价。
5. **内分泌用相对单位**，区分昼夜预期与应激超额。
6. **编码与提取用不同增益**；应激可能增强编码、损害提取。
7. **再巩固窗口是条件性的**，默认短、可关闭、可审计。
8. **环境效应小且有界**，个体差异参数化，默认允许关闭。
9. **所有随机性可复现**；所有状态可导出；所有判断带 components。
10. **不诊断、不替代、不承诺意识**；把系统定位为模拟与交互设计工具。

---

## 7. 参考文献

1. Ebbinghaus, H. (1885/1913). *Memory: A Contribution to Experimental Psychology*. https://psychclassics.yorku.ca/Ebbinghaus/index.htm
2. Murre, J. M. J., & Dros, J. (2015). Replication and analysis of Ebbinghaus' forgetting curve. *PLoS ONE*, 10(7), e0120644. https://doi.org/10.1371/journal.pone.0120644
3. Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science*, 2(6), 396–408. https://doi.org/10.1111/j.1467-9280.1991.tb00174.x
4. Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science*, 2(6), 409–415. https://doi.org/10.1111/j.1467-9280.1991.tb00178.x
5. Averell, L., & Heathcote, A. (2011). The form of the forgetting curve and the fate of memories. *Journal of Mathematical Psychology*, 55(1), 25–35. https://doi.org/10.1016/j.jmp.2010.08.004
6. Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354
7. Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
8. Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406, 722–726. https://doi.org/10.1038/35021052
9. McGaugh, J. L. (2000). Memory—a century of consolidation. *Science*, 287(5451), 248–251. https://doi.org/10.1126/science.287.5451.248
10. Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiological Reviews*, 93(2), 681–766. https://doi.org/10.1152/physrev.00032.2012
11. Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036–1060. https://doi.org/10.1037/0033-295X.111.4.1036
12. Open Spaced Repetition. *FSRS4Anki: The Algorithm*. https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm
13. Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*. Cambridge University Press. https://doi.org/10.1017/CBO9780511571299
14. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178. https://doi.org/10.1037/0022-3514.39.6.1161
15. Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental Psychology*. MIT Press.
16. Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking. In K. R. Scherer, A. Schorr, & T. Johnstone (Eds.), *Appraisal Processes in Emotion*. Oxford University Press. https://doi.org/10.1093/oso/9780195130072.003.0006
17. Marsella, S. C., & Gratch, J. (2009). EMA: A process model of appraisal dynamics. *Cognitive Systems Research*, 10(1), 70–90. https://doi.org/10.1016/j.cogsys.2008.03.003
18. Gebhard, P. (2005). ALMA: A layered model of affect. *AAMAS 2005*, 29–36. https://doi.org/10.1145/1082473.1082540
19. Becker-Asano, C., & Wachsmuth, I. (2010). Affective computing with primary and secondary emotions in a virtual human. *Autonomous Agents and Multi-Agent Systems*, 20, 32–49. https://doi.org/10.1007/s10458-009-9091-z
20. Verduyn, P., Delvaux, E., Van Coillie, H., Tuerlinckx, F., & Van Mechelen, I. (2009). Predicting the duration of emotional experience: Two experience sampling studies. *Emotion*, 9(1), 83–91. https://doi.org/10.1037/a0014171
21. Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psychological Science*, 21(7), 984–991. https://doi.org/10.1177/0956797610372634
22. Houben, M., Van Den Noortgate, W., & Kuppens, P. (2015). The relation between short-term emotion dynamics and psychological well-being: A meta-analysis. *Psychological Bulletin*, 141(4), 901–930. https://doi.org/10.1037/bul0000015
23. Dickerson, S. S., & Kemeny, M. E. (2004). Acute stressors and cortisol responses: A theoretical integration and synthesis of laboratory research. *Psychological Bulletin*, 130(3), 355–391. https://doi.org/10.1037/0033-2909.130.3.355
24. Kirschbaum, C., & Hellhammer, D. H. (1989). Salivary cortisol in psychobiological research: An overview. *Neuropsychobiology*, 22(3), 150–169. https://doi.org/10.1159/000118590
25. Sapolsky, R. M., Romero, L. M., & Munck, A. U. (2000). How do glucocorticoids influence stress responses? *Endocrine Reviews*, 21(1), 55–89. https://doi.org/10.1210/edrv.21.1.0390
26. de Kloet, E. R., Joëls, M., & Holsboer, F. (2005). Stress and the brain: From adaptation to disease. *Nature Reviews Neuroscience*, 6, 463–475. https://doi.org/10.1038/nrn1683
27. Joëls, M., Fernandez, G., & Roozendaal, B. (2011). Stress and emotional memory: A matter of timing. *Trends in Cognitive Sciences*, 15(6), 280–288. https://doi.org/10.1016/j.tics.2011.04.005
28. Cahill, L., & McGaugh, J. L. (1998). Mechanisms of emotional arousal and lasting declarative memory. *Trends in Neurosciences*, 21(7), 294–299. https://doi.org/10.1016/S0166-2236(98)01265-8
29. Arnsten, A. F. T. (2009). Stress signalling pathways that impair prefrontal cortex structure and function. *Nature Reviews Neuroscience*, 10, 410–422. https://doi.org/10.1038/nrn2648
30. Schultz, W. (1998). Predictive reward signal of dopamine neurons. *Journal of Neurophysiology*, 80(1), 1–27. https://doi.org/10.1152/jn.1998.80.1.1
31. Young, S. N. (2007). How to increase serotonin in the human brain without drugs. *Journal of Psychiatry & Neuroscience*, 32(6), 394–399. https://pubmed.ncbi.nlm.nih.gov/18076562/
32. Uvnäs-Moberg, K. (1998). Oxytocin may mediate the benefits of positive social interaction and emotions. *Psychoneuroendocrinology*, 23(8), 819–835. https://doi.org/10.1016/S0306-4530(98)00039-0
33. Brzezinski, A. (1997). Melatonin in humans. *New England Journal of Medicine*, 336(3), 186–195. https://doi.org/10.1056/NEJM199701163360306
34. McEwen, B. S., & Stellar, E. (1993). Stress and the individual: Mechanisms leading to disease. *Archives of Internal Medicine*, 153(18), 2093–2101. https://doi.org/10.1001/archinte.1993.00410200105011
35. Howarth, E., & Hoffman, M. S. (1984). A multidimensional approach to the relationship between mood and weather. *British Journal of Psychology*, 75(1), 15–23. https://doi.org/10.1111/j.2044-8295.1984.tb02749.x
36. Denissen, J. J. A., Butalid, L., Penke, L., & van Aken, M. A. G. (2008). The effects of weather on daily mood: A multilevel approach. *Emotion*, 8(5), 662–667. https://doi.org/10.1037/1528-3542.8.5.662
37. Kööts, L., Realo, A., & Allik, J. (2011). The influence of the weather on affective experience: An experience sampling study. *Journal of Individual Differences*, 32(2), 74–84. https://doi.org/10.1027/1614-0001/a000041
38. Klimstra, T. A., Frijns, T., Keijsers, L., Denissen, J. J. A., Raaijmakers, Q. A. W., van Aken, M. A. G., Koot, H. M., van Lier, P. A. C., & Meeus, W. H. J. (2011). Come rain or come shine: Individual differences in how weather affects mood. *Emotion*, 11(6), 1495–1499. https://doi.org/10.1037/a0025013
39. Lambert, G. W., Reid, C., Kaye, D. M., Jennings, G. L., & Esler, M. D. (2002). Effect of sunlight and season on serotonin turnover in the brain. *The Lancet*, 360(9348), 1840–1842. https://doi.org/10.1016/S0140-6736(02)11737-5
40. Rosenthal, N. E., Sack, D. A., Gillin, J. C., et al. (1984). Seasonal affective disorder: A description of the syndrome and preliminary findings with light therapy. *Archives of General Psychiatry*, 41(1), 72–80. https://doi.org/10.1001/archpsyc.1984.01790120076008
41. Anderson, C. A. (1989). Temperature and aggression: Ubiquitous effects of heat on occurrence of human violence. *Psychological Bulletin*, 106(1), 74–96. https://doi.org/10.1037/0033-2909.106.1.74
42. Hancock, P. A., & Vasmatzidis, I. (2003). Effects of heat stress on cognitive performance: The current state of knowledge. *International Journal of Hyperthermia*, 19(3), 355–372. https://doi.org/10.1080/0265673021000054630
43. de Dear, R. J., & Brager, G. S. (2002). Thermal comfort in naturally ventilated buildings: Revisions to ASHRAE Standard 55. *Energy and Buildings*, 34(6), 549–561. https://doi.org/10.1016/S0378-7788(01)00126-8
44. Open-Meteo. *Weather API Documentation*. https://open-meteo.com/en/docs
45. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive simulacra of human behavior. *UIST '23*. https://arxiv.org/abs/2304.03442
46. Packer, C., Wooders, S., Lin, K., et al. (2023). MemGPT: Towards LLMs as operating systems. https://arxiv.org/abs/2310.08560
47. Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025). Mem0: Building production-ready AI agents with scalable long-term memory. https://arxiv.org/abs/2504.19413
48. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). Zep: A temporal knowledge graph architecture for agent memory. https://arxiv.org/abs/2501.13956
49. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). A-MEM: Agentic memory for LLM agents. https://arxiv.org/abs/2502.12110
50. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). Cognitive architectures for language agents. https://arxiv.org/abs/2309.02427
