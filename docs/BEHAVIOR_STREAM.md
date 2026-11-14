# 行为映射与连续事件流

这两块解决的是同一个产品问题：引擎不能只输出“我现在有点难过”，也不能只会
一条一条吃消息。宿主需要知道**这次该怎么回**，以及**这一晚上整体发生了什么**。

## 1. 情绪到行为：`BehaviorMapper`

引擎负责评价、情绪、心境和内分泌；行为出口由产品决定。`BehaviorMapper` 是一个
很小的协议：

```python
from thymos import AgentMemory, BehaviorDirective, Profile, ThymosConfig

class MyBrandMapper:
    def map(self, affect, *, intent="mixed", emotion=None, context=None):
        if intent in {"factual", "technical"}:
            return BehaviorDirective(
                tone="neutral",
                verbosity="short",
                initiative="none",
                boundary="none",
                suggested_action="answer_directly",
                confidence=0.9,
                reason="task first",
                affect_label=emotion or affect.dominant_emotion,
            )
        if affect.stress >= 0.75:
            return BehaviorDirective(
                tone="gentle",
                verbosity="short",
                initiative="reflect",
                boundary="deescalate",
                suggested_action="acknowledge_then_ask",
                confidence=0.85,
                reason="high stress",
                affect_label=emotion or affect.dominant_emotion,
            )
        return BehaviorDirective(
            tone="warm",
            verbosity="medium",
            initiative="ask",
            boundary="none",
            suggested_action="ask_one_follow_up",
            confidence=0.6,
            reason="brand default",
            affect_label=emotion or affect.dominant_emotion,
        )

memory = AgentMemory(
    ".thymos/bot.sqlite",
    profile=Profile(name="小助手", location_name="杭州"),
    config=ThymosConfig(offline=True),
    behavior_mapper=MyBrandMapper(),
)

directive = memory.behavior("今晚有点累")
print(directive.to_dict())

plan = memory.plan("今晚有点累，陪我聊聊")
print(plan["behavior"])  # same directive, included in the audit plan
```

默认的 `DefaultBehaviorMapper` 是保守的产品先验，不是“人类行为模型”：

| 条件 | 默认行为 |
|---|---|
| 压力 ≥ 0.75，或 anger/fear/disgust | gentle / short / reflect / deescalate |
| factual / technical / history | neutral / short / answer_directly |
| sadness/anxiety 或心境 ≤ -0.30 | gentle / short / validate_then_offer_one_step |
| joy/excitement/trust 或心境 ≥ 0.35 | warm / medium / match_energy |
| 其他 | neutral / medium / answer_then_check_in |

它只输出**建议**，不直接改 LLM 的 prompt。宿主可以把它映射成系统提示、工具选择、
回复长度或人工接管条件。`confidence` 低时应该允许宿主忽略它。

## 2. 连续事件流：`observe_stream`

`AgentMemory.observe` 一次处理一条消息；`observe_stream` 处理一个有时间的窗口：

```python
from thymos import AgentMemory, Event

memory = AgentMemory(".thymos/bot.sqlite", now=1770000000.0)

window = memory.observe_stream(
    [
        Event("今天加班到九点，累死了", at=1770000000, tags=["工作"]),
        Event("领导还临时改需求，真的很烦", at=1770001800, tags=["工作"]),
        Event("不过同事帮我买了饭", at=1770003600, tags=["同事"]),
        Event("现在好多了", at=1770005400, tags=["恢复"]),
        Event("听起来今天挺不容易的", at=1770005460, role="assistant"),
    ],
    window="今晚",
)

print(window.summary)
print(window.memory_id)
print(memory.context("今晚怎么样", max_chars=800))
```

行为：

- 每个 `user` 事件仍走正常的 `observe` 写路径：事实、去重、合并、心境都生效；
- `assistant`/`system` 事件进入窗口记录，但不会被当成用户事实；
- 时间戳缺失时按 `step_seconds` 单调补齐；早于当前时钟的时间戳会被夹到当前，
  不会让心境倒流；
- 窗口结束时生成一条 `kind="semantic"` 的摘要记忆，tag 为 `window`，并在
  `context()` 中可检索；
- 摘要包含：事件数、用户/助手轮数、时间范围、情绪标签分布、心境起止、压力峰值、
  话题、最大转折；
- 摘要不是诊断，也不替代每条原始记忆；原始记忆仍然保留。

`WindowSummary` 的 `to_dict()` 默认不包含逐条 `records`，避免把整晚文本塞进审计
日志；需要回放时传 `include_records=True`。

## 3. 边界

- `observe_stream` 是**窗口批处理**，不是异步消息队列。宿主可以在一晚结束时调用，
  也可以每 10 分钟把缓冲区 flush 一次；后一种做法会产生多个窗口摘要。
- 摘要只基于规则统计和现有评价，不做 LLM 总结；需要 LLM 总结时，宿主可以读取
  `WindowSummary` 后自行生成，并把结果作为新的 `Event` 或外部记忆写入。
- 窗口标签来自引擎当前评价，不是人工标注。真实陪伴质量评估见
  [evals/companion_zh/README.md](../evals/companion_zh/README.md)。
- 行为映射不会自动执行安全策略；高风险场景仍需宿主自己的危机识别与人工接管。

## 4. 真实对话回归

`tests/test_real_dialogue_regression.py` 默认读取
`evals/companion_zh/data/companion_windows.jsonl`，重放 8 段 CPED 真实中文多轮
对话，并检查：

- 每个真实窗口都能被 `observe_stream` 摄入；
- 窗口摘要被写入、能被 `context()` 检索；
- SQLite 保存/重载后摘要仍在；
- 同一输入两次运行的动力学与摘要一致（忽略随机记忆 ID）；
- 156 条真实 CPED 语句都能进入评价管线，不抛异常。

它断言的是**管线不变量**，不是情绪准确率。真实数据上的准确率仍由
`experiments/eval_companion_zh.py` 单独报告，且不应拿来调词表。

要换成自己的真实会话，设置 `THYMOS_REAL_DIALOGUE_FIXTURE` 指向 JSONL 文件或
目录；格式与 PII 注意事项见
[tests/fixtures/README.md](../tests/fixtures/README.md)。
