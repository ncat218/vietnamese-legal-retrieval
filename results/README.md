# Results: evidence and interpretation

The public repository distinguishes **historical owner-reported public scores**, **artifact-verified internal metrics**, and **synthetic software tests**. These evidence classes are not interchangeable.

## Public leaderboard history

| Branch | Public Recall@5 | Status |
|---|---:|---|
| 04C direct D5/Jina/LambdaMART Top5 | 0.909833334 | Historical owner report |
| 17E inherited baseline | 0.9332 | Historical corrected value, rounded |
| 05A cache-safe bridge | 0.933833334 | Historical owner report |
| 06A backfilled bridge | 0.933833334 | Historical owner report |

The public score has not been independently rescored here. A current official leaderboard export is not bundled. The approximate increase over rounded17E is +0.000633334 Recall, or +0.0633 percentage points. This is not a significance claim. The separate team-champion branch is excluded from this solution's headline.

## 05A matched internal comparison

[`05a_internal_eval.csv`](05a_internal_eval.csv) is an exact aggregate export from the validated original handoff: **295 queries**, previously viewed internal evaluation. The complete 16-entry artifact manifest was independently hash/size verified before exporting only this table.

| Arm | Recall@5 | Improved / worsened | Mean delta vs baseline |
|---|---:|---:|---:|
| BASE17E | 0.931638418 | 0 / 0 | 0 |
| D5_JINA_LM_TOP5 | 0.912429379 | 4 / 10 | −0.019209040 |
| H1_TOP15_BASE17E_CACHE_SAFE | **0.935028249** | **1 / 0** | **+0.003389831** |

The pilot's paired bootstrap 95% interval is [0,0.010204082], including zero. The readiness records seven missing pairs, seven fallback queries, and `strict_h1_fully_scored=false`. This internal result must not be described as fully scored 06A or a new unseen holdout.

## 04B: a different evaluation scope

[`04b_internal_eval.csv`](04b_internal_eval.csv) contains aggregate all-query and600-query block rows from the 22-entry manifest-verified 04B handoff. Its all-query scope is 3,000 EVAL queries, not the 295-query 05A intersection.

| Arm | Recall@5 | Candidate Recall@50 |
|---|---:|---:|
| D5 | 0.906888889 | 0.981444444 |
| D5_JINA | 0.771166667 | 0.981444444 |
| D5_LM | 0.908277778 | 0.981444444 |
| D5_JINA_LM | 0.914805556 | 0.981444444 |

All arms retain the same candidate pool. These rows assess ranking under the frozen evidence/scoring contract; they are not evidence that the pretrained Jina model universally degrades retrieval. EVAL had previously informed retrieval experiments, which limits independent generalization claims.

## What the software tests establish

The CPU demo, synthetic replay, OOF integration and policy-parity checks establish implementation behavior on controlled examples. They do not establish competition accuracy or neural model parity. [validation.json](validation.json) records the checks performed on the public package, and [provenance.json](provenance.json) records source archive and metric hashes.

No competition gold IDs, query text, prediction files, dataset ZIPs or model weights are redistributed in this directory.
