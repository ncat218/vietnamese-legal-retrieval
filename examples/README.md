# Synthetic examples

Everything in this directory is authored for software demonstration. Document IDs, query IDs, labels, feature values and inherited scores do not come from the competition.

- `replay_input.jsonl`: exact15-document shortlist, inherited score/position records and baselineTop5 for one synthetic query.
- `gold.json`: synthetic relevance judgments for policy/metric tests only.
- `query_ids.json`, `corpus_ids.json`: allowlists used in official-shaped submission export.
- `train_features.jsonl`:24 synthetic queries,12 duplicate groups,12 candidates each. Feature columns carry familiar expert names, but their values are generated artificial numbers, not actual BM25/dense/Qwen/Jina outputs.

For strict replay, the protected baseline prefix is retained even when those documents are outside the shortlist. The trainer uses supplied duplicate-group IDs for OOF isolation. Follow [`docs/reproduce.md`](../docs/reproduce.md) for commands and schemas.
