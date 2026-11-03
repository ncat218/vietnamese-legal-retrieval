"""Deterministic query folds that keep supplied duplicate groups intact."""
import random


def group_folds(records: list[dict], *, n_folds: int = 5, seed: int = 20261103) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("At least two folds required")
    groups, seen = {}, set()
    for record in records:
        qid, group = record["question_id"], record["group_id"]
        if not isinstance(qid, str) or not qid or qid in seen or not isinstance(group, str) or not group:
            raise ValueError("Unique string question IDs and nonempty duplicate-group IDs required")
        seen.add(qid)
        groups.setdefault(group, []).append(qid)
    if len(groups) < n_folds:
        raise ValueError("Fewer duplicate groups than folds")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: -len(groups[group]))
    loads = [0] * n_folds
    result = {}
    for group in order:
        fold = min(range(n_folds), key=lambda f: (loads[f], f))
        result.update({q: fold for q in groups[group]})
        loads[fold] += len(groups[group])
    return result
