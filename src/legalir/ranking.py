"""The 05A cache-safe / 06A fully-scored bridge, independently reimplemented.

Inputs are frozen shortlist IDs, inherited 17E scores/17C positions and exact
baseline Top5. This module does not reconstruct the upstream learned features.
"""
import math


def validate_ids(ids: list[str], *, count: int | None = None) -> None:
    if not isinstance(ids, list) or any(not isinstance(d, str) or not d for d in ids):
        raise ValueError("Expected a list of nonempty string IDs")
    if len(ids) != len(set(ids)) or count is not None and len(ids) != count:
        raise ValueError("Unexpected cardinality or duplicate document IDs")


def select_top5(shortlist: list[str], baseline: list[str], scores: dict, *,
                mode: str = "06a_strict", shortlist_k: int = 15) -> dict:
    validate_ids(shortlist, count=shortlist_k)
    validate_ids(baseline, count=5)
    if mode not in ("05a_cache_safe", "06a_strict"):
        raise ValueError("Unknown bridge policy")
    for doc_id, entry in scores.items():
        value, position = entry["score"], entry["candidate_position"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite inherited score for {doc_id}")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise ValueError(f"Invalid inherited candidate position for {doc_id}")
    missing = sorted(set(shortlist) - set(scores))
    if missing:
        if mode == "06a_strict":
            raise ValueError("Missing inherited scores; backfill required: " + ", ".join(missing))
        return {"top5": baseline.copy(), "fallback": True, "missing_score_ids": missing,
                "mode": mode}
    order = sorted(shortlist, key=lambda d: (-scores[d]["score"], scores[d]["candidate_position"], d))
    result = baseline[:2]
    for doc_id in order:
        if doc_id not in result:
            result.append(doc_id)
        if len(result) == 5:
            break
    validate_ids(result, count=5)
    return {"top5": result, "fallback": False, "missing_score_ids": [], "mode": mode}
