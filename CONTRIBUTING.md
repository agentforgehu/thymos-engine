# 贡献指南

Thymos 是一个研究性项目，最需要的贡献不是“更多功能”，而是**更诚实的检验**：反例、参数批评、复现失败、边界条件。

## 1. 开发环境

```bash
git clone https://github.com/agentforgehu/thymos-engine.git
cd thymos-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python experiments/run_experiments.py
```

要求 Python 3.10+。运行时零依赖；开发依赖只有 pytest / numpy / matplotlib。

## 2. 提交前检查

```bash
pytest
python -m thymos --offline demo
python experiments/run_experiments.py
```

- 新功能必须有测试；
- 新参数必须写入 `docs/DESIGN.md` 的参数表，并标注来源（L/E/P/D）；
- 改变默认行为必须更新 `CHANGELOG.md`；
- 新实验必须能在固定 seed 下复现，并把指标写入 `docs/experiments.json`。

## 3. 代码风格

- 标准库优先；不要为了一个小功能引入重依赖。
- 类型注解尽量完整；公共 API 必须有 docstring。
- 纯函数优先；状态变更集中在 `engine.py`。
- 不要用全局随机；使用 `self.rng`。
- 不要 `print` 调试；用 `events` 或返回值。
- 中文注释可以，代码标识符用英文。

## 4. 参数与证据

任何参数改动都要回答三个问题：

1. **它对应什么构念？**（例如 `mood_tau_hours` = 心境回到目标的时间常数）
2. **依据是什么？**（文献先验 / 工程估计 / 占位 / 数据推导）
3. **改动后哪个实验/测试会变？**

如果只是“看起来更好”，请把改动放在 PR 描述里，并附带前后对比。不要偷偷调参让 demo 更好看。

## 5. 新增情绪规则

1. 在 `appraisal.py` 的 `appraise()` 里加规则，保持优先级清晰；
2. 在 `EMOTION_HALF_LIFE` 里给半衰期，并说明依据；
3. 在 `tests/test_appraisal.py` 里加正例/反例；
4. 在 `docs/DESIGN.md` 更新规则树；
5. 如果引入新标签，更新 dashboard 的颜色映射。

## 6. 新增激素/环境变量

1. 在 `endocrine.py` 的 `SPECS` 加变量，写明半衰期、基线、依据；
2. 在 `step()` 里给目标值与时间常数；
3. 在 `snapshot()` 与 dashboard 中展示；
4. 在 `docs/RESEARCH.md` 加证据段落；
5. 加测试：基线、上升/回落、边界。

## 7. 数据与伦理

- 不要提交真实用户的记忆/日记/位置数据；
- 测试用合成文本；
- 如果做用户研究，必须预先说明用途、获得同意、通过伦理审查；
- 不要用本系统做诊断、危机干预、情感操控；
- 不要在 Issue 里贴他人的私密内容。

## 8. Issue 模板

**Bug**

- 复现步骤
- 期望行为 / 实际行为
- 状态文件（脱敏）
- Python 版本 / OS
- 相关日志（不要贴原始私密内容）

**参数/模型讨论**

- 你认为哪个机制不对？
- 有什么反例/文献？
- 建议怎么改？会影响哪些实验？

**功能请求**

- 使用场景
- 为什么现有 API 不够
- 是否愿意实现/测试

## 9. PR 流程

1. Fork + 分支（`feat/...`、`fix/...`、`docs/...`）；
2. 写测试；
3. 更新文档/CHANGELOG；
4. 确保 CI 全绿；
5. PR 描述写清“改了什么、为什么、怎么验证”；
6. 至少一位维护者 review。

## 10. 好的第一批 Issue

- 给 `estimate_appraisal` 加反讽/双重否定测试；
- 把 `retrieval.py` 的 `time_bucket_for` 去重；
- 给 dashboard 加“导出状态”按钮；
- 写一个 `examples/agent_loop.py` 的真实 LLM 版本；
- 用公开 Anki 日志拟合 `S0`；
- 复现实验 1 并报告不同拟合方法的 RMSE。

## 11. 行为准则

参与即表示同意 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。对事不对人；批评模型，不攻击作者。

## 12. 许可

贡献的代码默认以 MIT 许可证发布。
