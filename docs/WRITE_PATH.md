# 写路径：准入、合并与审计

> 0.3.5 新增。目标不是“把所有聊天都存下来”，而是让每条记忆的进入、合并、跳过都有可检查的理由。

## 1. 为什么需要写路径

只有检索和遗忘还不够。一个长期陪伴 Agent 会遇到三类写问题：

1. **低价值重复**：寒暄、确认、天气式闲聊，如果每轮都写，记忆库很快被噪声占满。
2. **同一件事的多次说法**：用户今天说“我叫林墨，住杭州”，明天说“我住在杭州，名字是林墨”。这两条应合并，而不是并存。
3. **新旧冲突**：项目截止日期从 4 月 15 日改到 5 月 1 日；如果旧值继续被检索，模型会答错。

因此写路径不是“存储层”的附属，而是控制平面的一部分。它必须回答：**这条要不要存、要不要并、并到哪、为什么。**

## 2. 四种决策

`AgentMemory.observe()` 现在会生成一个 `last_write` 审计对象：

| action | 含义 | 典型条件 |
|---|---|---|
| `store` | 新建一条记忆 | 通过重要性门槛，且没有可合并候选 |
| `merge` | 并入已有记忆 | 语义相似度 ≥ 阈值，时间窗口内，且没有数字/日期冲突 |
| `reinforce` | 强化已有记忆 | 规范化文本完全相同（跨会话也生效） |
| `skip` | 不写 | 低于 `min_importance`，或纯寒暄 |

每条记录包含 `reason`、`importance`、`similarity`、`memory_id`。`mind.snapshot()["last_write"]` 和 `events` 里都能看到最近一次决策。

### 2.1 精确重复

`normalize_key()` 只做表层规范化：去空白、去标点、转小写。它不把“项目黄了”与“项目失败了”当成同一条；语义合并交给嵌入。

精确重复会：

- 不新增条目；
- `merged_count += 1`；
- 更新 `last_merged_at`、`last_access`、`access_times`；
- 仍走一次 `perceive(store_memory=False)`，所以这一轮的情绪/心境/内分泌照常更新。

这一点很重要：**“不写新记忆”不等于“这一轮没有发生”。** 用户重复一句辱骂，不应该因为去重而让心境无感。

### 2.2 语义合并

语义合并需要同时满足：

- 已有记忆的年龄 ≤ `write_merge_window_hours`（默认 72 小时）；
- 余弦相似度 ≥ `write_merge_similarity`（默认 0.90）；
- 两条文本中的数字/日期集合不冲突（例如 `4 月 15 日` vs `5 月 1 日` 不合并）；
- 目标条目仍是 active。

合并保留原条目，不删除历史：

- `content` 更新为较新的说法；
- `merged_count` 累加，`merged_from` 记录被并入的文本（最多 8 条）；
- `importance`、`salience` 取较大值；
- `valence` / `arousal` 做指数滑动平均；
- `stability` 略微增加，`last_merged_at` 更新。

### 2.3 跳过

低于 `min_importance` 的回合不会进入 `perceive`，但会写一条 `skip` 审计记录。纯寒暄（`嗯`、`晚安`、`谢谢`）默认也跳过；如果宿主显式传 `min_importance=0`，仍可强制保留。

## 3. 配置

```python
from thymos import AgentMemory, ThymosConfig

cfg = ThymosConfig(
    write_admission_enabled=True,
    write_min_importance=0.35,
    write_merge_similarity=0.90,
    write_merge_window_hours=72.0,
    write_semantic_merge_enabled=True,
)
memory = AgentMemory(".thymos/agent.sqlite", config=cfg)
```

- `write_admission_enabled=False`：恢复旧行为，所有通过 `min_importance` 的回合都新建条目。
- `write_semantic_merge_enabled=False`：只做精确去重，不做语义合并。
- 默认阈值偏保守。中文短句的嵌入相似度很容易到 0.8–0.9，0.90 是为了避免把“今天很开心”和“今天很累”合并；代价是部分改写不会合并。

## 4. 当前实测

### 4.1 单元与行为测试

`tests/test_write_path.py` 覆盖：

- 精确重复 → `reinforce`，条目数不增；
- 跨 JSON 重开 → 仍然去重；
- 语义改写 → `merge`（用确定性嵌入模拟）；
- 数字冲突 → 不合并；
- 低重要性 → `skip`；
- 合并后事实的 `memory_id` 指向存活条目；
- `perceive(store_memory=False)` 仍更新心境；
- 关闭准入后恢复旧行为。

### 4.2 24 条混合记忆的检索排序

同一批 12 个查询、24 条记忆（12 个目标 + 12 个干扰项），完整 `Thymos.recall()`：

| 嵌入 | R@1 | R@3 | MRR |
|---|---:|---:|---:|
| 哈希 n-gram（零依赖默认） | 0.333 | 0.417 | 0.417 |
| bge-small-zh-v1.5（本地 GGUF） | 0.583 | 0.750 | 0.661 |

这组数字说明两件事：

1. 真实中文嵌入明显优于哈希；哈希只适合作为零依赖基线。
2. 排序不能只看 sigmoid 后的概率。此前 `score` 在多个候选都饱和到 0.98 时，会被显著度乘数翻转；0.3.5 改为用未压缩的 activation 排序，`probability` 只保留给复习质量和遥测。

### 4.3 100 轮离线写路径

`experiments/longrun_100.py --offline` 的 100 轮场景：

- 67 条记忆 / 100 轮（0.67 条/轮）；
- `store` 67、`skip` 31、`reinforce` 2；
- 事实探针上下文 100/100；
- 心境范围约 0.37，压力峰值约 0.27。

这不是“效果很好”的证明；它只说明在作者构造的长场景里，准入没有把记忆库撑爆，也没有把关键事实挤掉。真实模型的 100 轮结果见 [LONGRUN_100.md](LONGRUN_100.md)。

## 5. 已知边界

- **语义合并依赖嵌入质量。** 哈希嵌入下，改写通常不会合并；这不是 bug，是零依赖基线的上限。
- **事实冲突仍以规则为主。** `FactStore` 对单值状态（城市、地址、职业、截止日期、睡眠、宠物位置）有“显式更新线索”守卫：一句裸的旧值复述不会覆盖新值；但复杂的多项目、多对象冲突仍可能出错。
- **合并会改写内容。** 原始说法保存在 `merged_from`，但不是完整的事件日志；如果宿主需要法务级审计，应在外层另存原始消息。
- **阈值未做真人标注。** 0.90 是工程默认值，不是从数据学出来的最优值。
