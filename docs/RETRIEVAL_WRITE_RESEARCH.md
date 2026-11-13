# 中文召回与写路径：证据笔记

> 这是 0.4.0 迭代的研究底稿，不是最终实验报告。外部结论只用于约束设计，不替代本项目自己的基准。

## 1. 中文检索

C-Pack / C-MTEB 把中文 embedding 评测拆成 6 类任务、35 个数据集，并提供了 bge 系列模型 [1]。`BAAI/bge-small-zh-v1.5` 的模型卡给出的 C-MTEB 数字是：平均 57.82，Retrieval 61.77，STS 49.11；向量维度 512 [2]。它不是“最强中文模型”，但足够做本地、可复现的基线，且 GGUF 只有约 26 MB，适合本项目的本地优先约束。

模型卡还有两个工程细节 [2]：

- 短查询检索建议加指令 `为这个句子生成表示以用于检索相关文章：`；passage 不加指令。
- v1.5 在不用指令时只有轻微下降；是否加指令应以自己的任务为准，不能把“官方建议”当成普遍最优。
- 相似度分数是相对排序，不是绝对阈值；不要拿 0.5 当“相似/不相似”的固定分界。

混合检索的工程共识是：BM25 擅长精确词、名称、编号；向量擅长改写和近义。两者应该融合，而不是用一个替代另一个 [3]。但融合方式必须验证：简单加权可能让常见字/二元组压过语义分数，RRF 也可能把弱词法排名带进来。本项目的 12 条小基准上，直接加权没有稳定提升 bge，因此先保留“独立词法通道 + 可配置权重”，用更大的评测决定默认权重。

## 2. 写路径

Generative Agents 的检索分数由 recency、importance、relevance 三项组成，说明“什么值得进入记忆”与“什么值得被取出”是两个问题 [4]。本项目已有重要性和遗忘曲线，但缺少写路径的显式准入/合并审计。

Mem0 当前公开算法强调 single-pass ADD-only extraction、实体链接、semantic + BM25 + entity 多信号检索 [5]。这给本项目一个反向约束：**不要为了去重而静默覆盖历史**。更安全的做法是保留原始条目，合并时记录 `merged_count`、`last_merged_at` 和来源，而不是把旧记忆删掉。

A-MEM 用 Zettelkasten 式 note construction、link generation 和 memory evolution：新记忆可以触发旧记忆的上下文/属性更新 [6]。这支持“合并是演化，不是覆盖”，也说明合并需要可审计字段。

LongMemEval 把长期记忆拆成信息抽取、多会话推理、时间推理、知识更新、拒答五项能力 [7]。MemoryAgentBench 则强调准确检索、test-time learning、长程理解和选择性遗忘 [8]。因此本项目的 100 轮测试不能只看“召回命中”，还要看：

- 旧事实是否被新事实替代；
- 重复/相似事件是否被合并，而不是无限堆积；
- 该拒答时是否拒答；
- 跨会话重开后是否仍一致；
- 选择性遗忘是否仍然工作。

## 3. 本项目的设计结论

1. 保留哈希嵌入作为零依赖默认；增加 `LlamaCppEmbedder` / `OpenAIEmbedder` 作为真实中文嵌入路径，并用本地 bge-small 验证。
2. 检索采用 semantic + BM25 + exact phrase + tags/context 多通道；权重可配置，默认不假设某一个模型最好。
3. 写路径分为 `store`、`merge`、`reinforce`、`skip` 四种决策；每次决策带原因、相似度、重要性和目标 memory id。
4. 精确重复跨会话去重；语义相似只在时间窗口内合并，数字/日期冲突不合并；合并保留原条目并累加 `merged_count`。
5. 评测明确区分“作者构造的 100 轮自测”与“真人标注/真实用户”。前者验证稳定性，不证明产品效度。

## 参考文献

[1]: https://arxiv.org/abs/2309.07597 "C-Pack: Packed Resources For General Chinese Embeddings"
[2]: https://huggingface.co/BAAI/bge-small-zh-v1.5 "BAAI bge-small-zh-v1.5 model card"
[3]: https://help.aliyun.com/zh/milvus/use-cases/full-text-retrieval-by-milvus-bm25-algorithm-and-application-of-hybrid-retrieval-to-rag-system "Milvus BM25 与混合检索"
[4]: https://arxiv.org/abs/2304.03442 "Generative Agents: Interactive Simulacra of Human Behavior"
[5]: https://github.com/mem0ai/mem0 "Mem0: The Memory Layer for AI Agents"
[6]: https://arxiv.org/abs/2502.12110 "A-MEM: Agentic Memory for LLM Agents"
[7]: https://arxiv.org/abs/2410.10813 "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
[8]: https://arxiv.org/abs/2507.05257 "Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions"
