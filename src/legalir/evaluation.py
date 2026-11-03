"""Query-macro metrics, complete-query comparison and grouped bootstrap."""
import math
import random
from statistics import mean

from .ranking import validate_ids


def evaluate(predictions: dict[str, list[str]], gold: dict[str, list[str]], *, k: int = 5) -> dict:
    if k < 1 or not gold or set(predictions) != set(gold):
        raise ValueError("Require positive k and exact, nonempty query coverage")
    records = []
    for query_id in sorted(gold):
        validate_ids(predictions[query_id], count=k)
        validate_ids(gold[query_id])
        truth = set(gold[query_id])
        if not truth:
            raise ValueError(f"Gold set must be nonempty for {query_id}")
        predicted = predictions[query_id]
        hits = [int(doc_id in truth) for doc_id in predicted]
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(truth), k) + 1))
        records.append({"question_id": query_id, "recall": sum(hits) / len(truth),
                        "ndcg": dcg / ideal, "mrr": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0),
                        "relevant_retained": sum(hits), "all_loss": not any(hits)})
    return {"k": k, "queries": len(records), "mean_recall": mean(row["recall"] for row in records),
            "mean_ndcg": mean(row["ndcg"] for row in records),
            "mean_mrr": mean(row["mrr"] for row in records),
            "all_loss_queries": sum(row["all_loss"] for row in records), "per_query": records}


def compare(baseline: dict, challenger: dict, gold: dict, *, groups: dict | None = None,
            k: int = 5, resamples: int = 2000, seed: int = 20261103) -> dict:
    base = {row["question_id"]: row for row in evaluate(baseline, gold, k=k)["per_query"]}
    trial = {row["question_id"]: row for row in evaluate(challenger, gold, k=k)["per_query"]}
    if groups is None or set(groups) != set(gold) or any(not g for g in groups.values()) or resamples < 1:
        raise ValueError("Complete duplicate-group assignments and positive resamples required")
    movements, grouped = [], {}
    for q in sorted(gold):
        delta = trial[q]["recall"] - base[q]["recall"]
        grouped.setdefault(groups[q], []).append(delta)
        truth = set(gold[q])
        movements.append({"question_id": q, "recall_delta": delta,
                          "gained_relevant": sorted((set(challenger[q]) - set(baseline[q])) & truth),
                          "lost_relevant": sorted((set(baseline[q]) - set(challenger[q])) & truth)})
    rng = random.Random(seed)
    group_ids = sorted(grouped)
    samples = []
    for _ in range(resamples):
        values = [value for group in rng.choices(group_ids, k=len(group_ids)) for value in grouped[group]]
        samples.append(mean(values))
    samples.sort()

    def quantile(p):
        position = p * (len(samples) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return samples[lo] + (samples[hi] - samples[lo]) * (position - lo)

    return {"mean_delta": mean(row["recall_delta"] for row in movements),
            "improved_queries": sum(row["recall_delta"] > 0 for row in movements),
            "worsened_queries": sum(row["recall_delta"] < 0 for row in movements),
            "ci95": [quantile(0.025), quantile(0.975)], "bootstrap_unit": "duplicate_group",
            "resamples": resamples, "seed": seed, "movements": movements}
