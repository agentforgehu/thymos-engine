# NOTICE — CPED-derived evaluation sample

The files under `data/` are a small, stratified evaluation sample derived from
the **CPED** test split.

## Citation

```bibtex
@article{chen2022cped,
  title={{CPED}: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset for Conversational AI},
  author={Yirong Chen and Weiquan Fan and Xiaofen Xing and Jianxin Pang and Minlie Huang and Wenjing Han and Qianfeng Tie and Xiangmin Xu},
  journal={arXiv preprint arXiv:2205.14727},
  year={2022},
  url={https://arxiv.org/abs/2205.14727}
}
```

## Source and license

- Repository: <https://github.com/scutcyr/CPED>
- License of the repository: **Apache-2.0**
- Paper: <https://arxiv.org/abs/2205.14727>
- Files used: `data/CPED/test_split.csv` from the repository

The repository's license and the paper's copyright statement should be read
together. CPED was constructed from Chinese TV shows; rights in the original
video/subtitles remain with their respective owners. This sample is included
only for non-commercial research and evaluation, with source IDs retained so
that errors can be traced back to the original corpus.

If you redistribute these derived files, keep this NOTICE and the citation.
If you need the full corpus, obtain it from the CPED repository and follow its
terms.

## What was changed

- Selected a deterministic stratified sample of 156 utterances from the test
  split (12 per CPED emotion label).
- Selected 8 real multi-turn windows from the test split; kept the original
  utterance text and source IDs, added evaluation timestamps and a derived
  majority-sentiment label.
- Added evaluation-only metadata (`id`, `source`, `scene`, `speaker`, `gold`).
- No text was generated or paraphrased by an LLM.
