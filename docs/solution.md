# From high-recall candidates to a reliable Top5

**Vietnamese Legal Retrieval — solution write-up**
Author: [ncat218](https://github.com/ncat218) · UIT DSC 2026 · Public edition, November 2026

## 1. Task and motivation

Given a Vietnamese legal question, return exactly five relevant document/article IDs. The principal competition metric discussed in this project is Recall@5: for each query, divide the number of relevant documents retrieved in five positions by the total number of relevant documents, then average over queries.

This creates two distinct problems. First, the candidate generator must retain the applicable documents. Second, the final ranker must distinguish closely related provisions in a very small output budget. The experiments repeatedly showed that solving the first problem does not automatically solve the second.

The public case study focuses on the **05A/06A bridge**, the branch reported at public Recall@5 **0.933833334**. Historical public scores were supplied by the project owner and recorded during the research. This repository does not independently rescore the public leaderboard, claim a competition rank, or present a different team's champion as this branch's achievement.

## 2. Document preparation and retrieval

The historical preparation stage normalized document and question identities, legal metadata and text, produced short/long chunks with document mappings, and recorded corpus and split identities. The reusable corpus fingerprint is recorded in the original artifact chain; raw competition documents and labels are not redistributed here.

Three complementary retrieval lanes generate natural Top200 candidates:

| Lane | Useful signal |
|---|---|
| BM25 | Exact wording, terminology and legal references |
| BGE-M3 dense retrieval | Semantic similarity when wording differs |
| DEk21-v2 dense retrieval | A complementary Vietnamese representation |

The D5 candidate pool uses equal-weight RRF:

```text
RRF(d) = sum over sources containing d of 1 / (60 + source_rank(d))
```

Source ranks are one-based; absent documents contribute zero. The frozen D5 implementation uses float64 accumulation in its declared source order and breaks ties by document ID. Top50 is the neural shortlist input. Candidates are not selected or inserted using gold labels.

The small BM25 implementation in this public package is an executable baseline over Unicode word tokens. It does not reconstruct the historical legal-aware indexes or pretrained dense encoders. `reciprocal_rank_fusion` is reusable with externally supplied source rankings.

## 3. Focused evidence and the first ranker

For every query–document candidate, the historical 04A preparation selects two short chunks using lexical overlap without reading answers. The downstream Jina stage uses bounded title/chunk evidence. These limits were runtime choices made for the T4 environment, not demonstrated optimal evidence budgets.

04B uses **jinaai/jina-reranker-v3.5**, pinned to revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`. Its protocol scores each entire Top50 list jointly. A separate frozen LambdaMART combines Jina and retrieval features and produces Top15.

The historical query/document budgets were 256 tokens each, with a 16,384-token prompt cap. These are model-token budgets; the public demo's word windows are not a substitute for the model's tokenizer or historical legal-unit chunker. Exact candidate order, evidence, tokenizer, truncation and model identity must match before reusing a neural cache.

Jina is a listwise model. Splitting Top50 into smaller batches, scoring an extra tail separately or reusing Top80 scores for Top10 changes the contract. This public package deliberately consumes precomputed final scores instead of hiding that difference behind a generic pointwise adapter.

## 4. The inherited 17E branch

The stronger pre-existing baseline has its own lineage:

```text
Protected multi-source candidate pool (17B/17C)
→ evidence selection and Qwen3-Reranker-0.6B (17D)
→ retrieval/Qwen features and a three-seed LambdaMART ensemble (17E)
→ protected first two documents inherited from 14A, then Top5
```

Qwen contributes logits, relevance, margins, ranks and query-relative transformations. Pure Qwen final ordering was weak in the historical experiments, so its value is studied as a feature source. Pool-relative transformations mean that a score generated in one candidate context cannot automatically be reconstructed from a smaller shortlist.

The final 05A/06A stage uses this frozen 17E score. It does not retrain a new final model over the 15 surviving documents. The baseline's protected first two documents can come from outside the new Top15; the effective final candidate union therefore has at most 17 documents.

## 5. 05A: a cache-safe pilot

For each query:

1. Obtain the frozen Jina/LambdaMART Top15.
2. Check that an inherited 17E score exists for **every** candidate.
3. If any score is missing, return the **entire exact baseline Top5** and log the missing pairs.
4. Otherwise preserve baseline Top2 and fill positions 3–5 from the shortlist, excluding duplicates.

Tie-breaking is frozen: score 17E descending, candidate_position17C ascending, then document ID ascending. Missing candidates are not dropped and their scores are not replaced by zero. The implementation is in [`ranking.py`](../src/legalir/ranking.py).

The validated 05A artifact reports seven missing pairs across seven internal EVAL queries. Historical project analysis records 53 missing public pairs across 40 of 1,000 public queries. The pilot therefore measures the cache-safe policy, not a fully scored new method on every query.

## 6. 06A: complete the scoring path

06A prepares evidence for missing query–document pairs, applies the exact existing Qwen checkpoint/prompt/scoring rule, checks numerical parity with previous scores, and rebuilds feature transformations in the declared context of base200 plus newly required candidates. The frozen 17E models remain unchanged.

This is more demanding than assigning a score to each isolated new document. Query-relative ranks and normalization depend on the rest of the scoring context. The backfill also included work for a separate expansion branch; runtime should not be attributed entirely to the small set of missing 05A pairs.

The public strict policy refuses incomplete inputs. It is a replay of the final decision rule over supplied scores, **not the historical Qwen backfill or a complete reconstruction from raw documents**.

## 7. Evaluation design

The 04B ranking experiment used a shared 4,000-query TRAIN / 3,000-query EVAL split, with duplicate-group-aware TRAIN OOF model selection. EVAL predictions were frozen before evaluation. That EVAL had previously been used for retrieval development, so it is not described as a completely unseen holdout.

05A compares an older inherited model on the **intersection** of 04B EVAL and legacy fold4 evaluation, removing duplicate components that appeared in the inherited model's training folds. This yields **295 queries**. This scope differs from 04B's 3,000 queries and from the public leaderboard.

This public reference trainer creates new group-preserving OOF folds, saves every fold model and held-out prediction, and fits a final model only after OOF scoring. It does not certify the provenance of supplied upstream features. Cached supervised scores are not made OOF by training a cross-fit head on top of them.

## 8. Results

### Historical public results

| Method | Public Recall@5 | Evidence class |
|---|---:|---|
| D5/Jina/LambdaMART direct Top5, 04C | 0.909833334 | Owner-reported historical leaderboard |
| Frozen inherited 17E | 0.9332 | Corrected, rounded historical value |
| 05A cache-safe Top15 bridge | **0.933833334** | Owner-reported historical leaderboard |
| 06A backfilled bridge | **0.933833334** | Owner-reported historical leaderboard |

The difference from the rounded baseline is approximately +0.063 percentage points. No public significance test is available. Equal 05A/06A scores do not prove equal predictions.

### Artifact-verified 05A internal comparison — 295 queries

| Arm | Mean Recall@5 | Improved / worsened vs baseline |
|---|---:|---:|
| Baseline 17E | 0.931638418 | — |
| D5/Jina/LambdaMART direct Top5 | 0.912429379 | 4 / 10 |
| Cache-safe Top15 → 17E bridge | **0.935028249** | **1 / 0** |

The bridge's mean delta is +0.003389831 with a saved paired bootstrap interval [0, 0.010204082]. The interval includes zero, and one query drives the gain. This supports the pilot's engineering feasibility, not a broad, robust improvement claim.

### Artifact-verified 04B comparison — a different 3,000-query scope

| Arm | Mean Recall@5 |
|---|---:|
| D5 retrieval order | 0.906888889 |
| D5 + raw Jina order | 0.771166667 |
| Retrieval-feature LambdaMART | 0.908277778 |
| Retrieval + Jina LambdaMART | **0.914805556** |

Candidate Recall@50 is 0.981444444 for all four arms. This experiment changes ordering over the same pool. It illustrates why the entire scoring/fusion policy needs evaluation. It does not isolate a universal weakness of the pretrained Jina model.

The exported aggregate tables have their source-archive identities and metric hashes in [`results/provenance.json`](../results/provenance.json). No per-query gold labels, query text or submission predictions are published as evidence.

## 9. Engineering decisions and trade-offs

- **Two independent GPU workers:** shard queries across T4×2, avoiding distributed-training coordination for long scoring jobs. Keep each listwise candidate list intact on one worker.
- **Query-level checkpointing:** commit completed queries with model, input and protocol identity; resume only compatible work.
- **Explicit failure modes:** incomplete scores, changed pools and failed parity checks stop strict inference.
- **Separate experiment cost from deployment cost:** cached inference, all-candidate diagnostics and prospective selective scoring have different compute budgets.
- **Artifact lineage:** keep input/model revisions, raw scores, per-query movement, manifests and readiness alongside aggregate metrics.

The deployment budget discussed in the competition is <4B logical model parameters. The complete inherited 05A/06A upstream ledger is not fully attested. Reusing a cache or quantizing a model does not remove its checkpoint from the dependency count.

## 10. Lessons and limits

The most durable result is the diagnosis: the system needs both candidate coverage and discrimination at the final boundary. Adding models or expanding pools is useful only if the final policy converts available evidence into better Top5 decisions.

Conservative protected-prefix policies retain strong baseline behavior but cannot fix errors in the locked prefix. Fixed Top15 compression can discard relevant documents. Frozen rankers may see a shifted candidate distribution. Small, repeatedly viewed evaluation scopes limit the strength of improvement claims.

The public package demonstrates core policies, baseline retrieval, validation and an optional trainable head. **Historical full-corpus inference, pretrained scorer execution and exact leaderboard reproduction are not currently packaged.** Reconstructing them requires the original indexes, caches, feature construction, checkpoint identities and official data access. See [Reproduction](reproduce.md) for a concrete dependency matrix.

## 11. Author and references

I developed the project's data/retrieval/ranking notebooks and experiment analysis. The public edition is a focused reimplementation of reusable components, supported by aggregate evidence from the research artifacts. [References and model attribution](references.md).
