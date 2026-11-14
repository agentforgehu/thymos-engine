# 中文陪伴评估集（companion_zh）

这不是一套“让 Thymos 看起来正确”的自测剧本。它从公开中文情绪对话数据
**CPED** 的 test split 中抽取真实语句与多轮窗口，并把数据来源、标签来源和
限制写清楚。

当前版本解决的是“没有外部数据”的问题，不解决“已经有真实用户盲评”的问题。

## 数据来源

- 数据集：CPED: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset
- 论文：Chen et al., 2022, <https://arxiv.org/abs/2205.14727>
- 仓库：<https://github.com/scutcyr/CPED>（Apache-2.0）
- 使用 split：**test**，不参与任何调参
- 采样：按 CPED 的 13 个 emotion 标签分层，每类 12 条，共 **156 条**
- 窗口：从 test split 中选 8 段真实多轮对话，每段取同一说话人的 5–6 轮；
  窗口标签由 CPED 的 sentiment/emotion 标注派生，**不是**人工陪伴质量评分

原始 CSV 不随仓库分发；仓库只保留抽样后的语句、来源 ID 和标签。来源与许可见
[NOTICE.md](NOTICE.md)。

## 文件

| 文件 | 用途 |
|---|---|
| `data/cped_emotion_sample.jsonl` | 156 条真实中文语句，带 CPED sentiment/emotion 金标 |
| `data/companion_windows.jsonl` | 8 段真实多轮窗口，带来源、事件和派生标签 |
| `annotation_protocol.md` | 陪伴质量人工标注协议（下一步用，不是当前已完成） |
| `results/lexical_baseline.json` | 规则评价基线在真实数据上的结果 |
| `experiments/eval_companion_zh.py` | 可复现运行器 |

### 单条样本 schema

```json
{
  "id": "cped-test-27_000_001",
  "source": {
    "dataset": "CPED",
    "split": "test",
    "dialogue_id": "27_000",
    "utterance_id": "27_000_001",
    "repository": "https://github.com/scutcyr/CPED",
    "license": "Apache-2.0 (repository); see NOTICE.md"
  },
  "scene": "car",
  "speaker": "林宛瑜",
  "text": "谢谢你哦",
  "gold": {"sentiment": "neutral", "emotion": "neutral"}
}
```

### 窗口 schema

`events` 是真实语句；`at` 是评估用的相对时间戳，不表示原始视频时间。
`expected.majority_sentiment` 是窗口内 CPED 金标的多数票，不是人工对“这一晚”
的总体判断。

## 当前基线（规则评价，未针对 CPED 调参）

运行：

```bash
PYTHONPATH=src python experiments/eval_companion_zh.py
```

结果（156 条语句、8 个窗口）：

| 指标 | 结果 |
|---|---:|
| 单条 emotion accuracy | **0.0962** |
| 单条 emotion macro-F1 | **0.0482** |
| 单条 sentiment accuracy | **0.1731** |
| 单条 sentiment macro-F1 | **0.1999** |
| 窗口多数情绪方向 accuracy | **0.50**（8 个窗口，样本很小） |
| 窗口摘要可被 `context()` 取回 | **1.00** |

这组数字不好看，但它是真实数据上的起点。规则评价擅长处理“闭嘴/谢谢你/项目黄了”
这类显式线索，不擅长电视对白里的省略、反讽和语境依赖。**不要**为了让这组数字
变好而按 CPED 样本改词表；那会把评估集变成训练集。下一步应比较规则、LLM
appraiser 和人工标注，而不是继续加词。

## 它不是什么

- 不是真实陪伴用户数据；
- 不是人工盲评；
- 不是 CPED 全量基准，只是 156 条分层样本 + 8 个窗口；
- 不是“陪伴质量”分数。CPED 的 emotion/sentiment 标签只说明说话人的情绪类别，
  不说明一个陪伴机器人是否该共情、追问、给建议或保持沉默。

## 下一步：陪伴质量人工标注

1. 用 `annotation_protocol.md` 的双人标注，至少 200 个真实/授权窗口；
2. 先算标注者一致性（Krippendorff's alpha），低于 0.6 先改定义，不急着评模型；
3. 盲评时隐藏系统身份，A/B 条件 token 对齐；
4. 主指标不是“情绪标签准确率”，而是“用户是否感到被理解、被记得、没有被伤害”；
5. 报告失败案例，不只报告均值。

## 许可与引用

如果你使用这些派生样本，请保留 [NOTICE.md](NOTICE.md) 并引用 CPED 原论文。
CPED 仓库为 Apache-2.0；原始视频/字幕权利归原权利人，样本仅用于研究与评估。
