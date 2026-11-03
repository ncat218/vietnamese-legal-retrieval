# Run, inspect and reproduce

## Three executable paths

| Path | What actually runs | Historical score reproduced? |
|---|---|---|
| `legalir demo` / `search` | CPU BM25 and lexical evidence over local documents | No |
| `legalir replay` | 05A/06A final selection over supplied frozen scores | No neural inference; no score claim |
| `legalir train` / `predict` | Native LightGBM head with saved group-isolated OOF outputs | New reference implementation, not original 17E |

No command downloads competition data, contacts an inference API or loads hidden checkpoints. Output directories are new runs; existing completed runs are not overwritten.

## Installation

Python 3.11+ is required. Core search, replay, metrics, manifests and tests use the standard library. Training additionally requires LightGBM 4.6.0 and NumPy.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install and inspect:

```bash
python -m pip install -e .
legalir --help
legalir demo
python -m unittest discover -s tests -v
```

The optional path is `python -m pip install -e ".[train]"`. The native LightGBM API avoids a scikit-learn wrapper. This public reference dependency set is distinct from the original GPU notebook environment.

## Search your own corpus

UTF-8 JSONL, one document per line:

```json
{"document_id":"my-doc-001","title":"Example title","text":"Your document text"}
```

```bash
legalir search --corpus path/to/documents.jsonl --query "hợp đồng lao động" --top-k 5
```

IDs are unique strings. Unicode normalization retains Vietnamese accents. Search uses a basic BM25 baseline with deterministic document-ID ties. Its word tokenization, word-window chunks and lexical evidence are reference choices; they do not reconstruct the original preparation/indexing stages.

## Replay the final selection policy

[`examples/replay_input.jsonl`](../examples/replay_input.jsonl) is a complete synthetic schema example. Each line has:

| Field | Contract |
|---|---|
| `question_id` | Unique string query ID |
| `shortlist` | Exactly 15 unique document IDs, supplied by the upstream Jina/LambdaMART stage |
| `baseline_top5` | Exactly 5 unique IDs; first 2 are protected anchors |
| `inherited_scores` | Map documentID → `{score, candidate_position}` |

`candidate_position` is the original 17C tie-break position. It must not be replaced by Jina rank or the new shortlist position. Scores must be finite. The 05A mode returns the **entire baseline5** if any shortlist score is missing; strict06A fails instead. Both keep the exact protected prefix and never insert a gold document.

```bash
legalir replay --input examples/replay_input.jsonl --mode 06a_strict --output outputs/replay-001
legalir evaluate --predictions outputs/replay-001/predictions.json --gold examples/gold.json --output outputs/replay-001-metrics.json
legalir export --predictions outputs/replay-001/predictions.json --query-ids examples/query_ids.json --corpus-ids examples/corpus_ids.json --output outputs/submission-001
```

Prediction and gold files map questionID to lists of document IDs. Evaluation refuses incomplete query sets, duplicate IDs or an output length other than K. Gold sets must be nonempty. The exported competition-shaped JSON is `questionID → {"answer": [fiveIDs]}`. Export checks query coverage, corpus IDs and ZIP roundtrip; it never submits to a competition.

`evaluate` reports query-macro Recall, nDCG and MRR, plus all-loss queries and per-query metrics. Python `legalir.evaluation.compare` additionally records gained/lost gold and duplicate-group bootstrap intervals. These diagnostics require permitted labeled evaluation data, not public/private leaderboard labels.

## Train the optional reference LambdaMART head

The committed feature fixture contains 24 queries, 12 duplicate groups, 12 candidates/query and deterministic artificial labels/features. This verifies fitting/OOF plumbing only. It does not contain real neural model scores.

```bash
python -m pip install -e ".[train]"
legalir train --features examples/train_features.jsonl --config configs/reference_ranker.json --output outputs/ranker-001
```

Each JSONL query record contains `question_id`, a supplied `group_id`, `partition: "train"`, and candidates with `document_id`, binary `label` and numeric `features`. The configuration fixes the feature allowlist and hyperparameters before fitting. Labels, query IDs, fold IDs and group IDs are rejected as features. The trainer accepts TRAIN records only.

Output files include:

```text
protocol_before_fit.json  # Input/config SHA256, package versions, exact fold map
fold_0.txt ...            # One model per held-out fold
oof_scores.json          # Every candidate predicted by its held-out model
fold_audit.json          # Query/group counts and overlap checks
final_model.txt          # Fit on all TRAIN after OOF completion
readiness.json           # PASS_REFERENCE_HEAD_TRAINED, not a benchmark result
manifest.json            # Hash and size of every artifact
```

For **TRAIN performance**, rank candidates using `oof_scores.json`, never the final model's in-sample predictions. For a separate EVAL set, use its own feature JSONL:

```bash
legalir predict --features path/to/eval_features.jsonl --model outputs/ranker-001 --output outputs/eval_predictions.json
legalir evaluate --predictions outputs/eval_predictions.json --gold path/to/eval_gold.json
```

The supplied duplicate-group IDs must be correct; the trainer does not discover near duplicates. Group isolation applies to the new head fits. Any supervised upstream features must separately have cross-fit provenance. Raw pointwise scorer inference, cached in-sample ranker outputs and OOF meta-features are different evidence classes.

## Exact historical reproduction: what is still required

| Historical stage | Required assets not redistributed here |
|---|---|
| Preparation/retrieval | Official corpus/query access; exact legal-unit/chunk mapping; BM25/dense indexes and encoder identities |
| 04B Top50→Top15 | Exact ordered Top50/evidence, pinned Jina revision/tokenization, complete inference cache, trained04B model/feature schema |
| 14A/17C/17D/17E | Inherited protected anchors, full candidate context, Qwen checkpoint/prompt identities, feature transformations and frozen three-seed models |
| 06A backfill | Missing-pair evidence, numerical parity probes and the exact base200-plus-additional-candidate context |
| Validation | Frozen query IDs, duplicate groups, splits and label permissions |
| Deployment | Complete upstream checkpoint/parameter ledger and model/data rights |

[`configs/historical_05a_06a.json`](../configs/historical_05a_06a.json) documents the historical contract. It is **not an executable end-to-end neural inference config**. [`configs/reference_ranker.json`](../configs/reference_ranker.json) trains the new reference head and must not be substituted into the historical score claim.

The historical T4×2 scorers used independent query workers, query-level checkpoints and identity-checked resume. They retained Kaggle CUDA PyTorch. Reconstructing those scoring jobs is a distinct task from executing this public CPU package.

## Reproducibility principles

- Freeze pools, prompts, feature semantics, seeds and scoring rules before evaluation.
- Keep duplicate groups within partitions and held-out fitting units.
- Do not insert gold into candidates or choose public/private policies from labels.
- Keep raw outputs needed for movement and lost-gold analysis in authorized storage.
- Preserve completed outputs; version substantial successors.
- Report candidate recall, internal metrics and owner-reported leaderboard results separately.
- Count every deployed model dependency, including those that generated cached scores. The historical full-pipeline parameter ledger is not yet fully attested.
