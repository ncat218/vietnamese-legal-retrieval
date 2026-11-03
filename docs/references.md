# References and attribution

## Research context and ownership

The data preparation, retrieval, ranking and experiment notebooks for this research project were developed by **[ncat218](https://github.com/ncat218)**. The author confirmed responsibility for the notebook collection used to prepare this public case study. The public modules are a focused reimplementation, not an export of the full notebook archive.

Pretrained models, training libraries and the competition data remain separate third-party components. Public source availability does not confer rights to redistribute model weights or competition datasets.

## Method and implementation references

- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*: [paper](https://doi.org/10.1561/1500000019).
- Cormack, Clarke and Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*: [paper](https://doi.org/10.1145/1571941.1572114).
- BGE-M3: [official model card](https://huggingface.co/BAAI/bge-m3).
- DEk21-v2 used in this branch: [CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2](https://huggingface.co/CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2), historical revision `99a2963b2f51fa7a570a3e7f550d7993b9de90a8`.
- Jina v3.5: [official model card](https://huggingface.co/jinaai/jina-reranker-v3.5), historical revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`. The model is listwise; consult the pinned implementation and model terms before execution.
- Qwen3-Reranker-0.6B: [official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Historical execution used checkpoint-file attestation; this public package does not invent a missing immutable revision.
- LightGBM: [official project](https://github.com/microsoft/LightGBM), [native Python training API, v4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html).

## Presentation reference

The solution-write-up format was inspired by [sakaku and tascj's RSNA Knee Abnormality Detection first-place write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution): explain design decisions, separate evaluation scopes, report component comparisons and describe inference constraints. No medical task method, prose or code is copied into this LegalIR implementation.

## Data and examples

All documents in `src/legalir/data/` and all inputs in `examples/` are authored fictional/synthetic examples. They are not authoritative laws, real legal advice, competition examples or actual neural model predictions. Aggregate metric exports in `results/` preserve the recorded internal evaluation values; their hashes and source scope are documented separately.
