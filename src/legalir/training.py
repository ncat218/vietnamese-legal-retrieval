"""Optional native-LightGBM ranking head with group-isolated OOF outputs.

This reference trainer operates on supplied feature tables. It is not an exact
retraining of the historical 17E feature pipeline or competition checkpoints.
"""
from importlib.metadata import version
import math
from pathlib import Path

from .artifacts import read_json, read_jsonl, sha256, write_json, write_manifest
from .splits import group_folds


def _dependencies():
    try:
        import numpy as np
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[train]"') from exc
    return np, lgb


def _matrix(records, features, *, labels):
    matrix, targets, sizes, identities = [], [], [], []
    if not records:
        raise ValueError("Nonempty feature records required")
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature names must be nonempty and unique")
    forbidden = ("gold", "label", "answer", "target", "relevant", "fold", "question_id", "document_id", "group_id")
    if any(any(word in name.casefold() for word in forbidden) for name in features):
        raise ValueError("Labels, identities and folds cannot be ranking features")
    if len({row["question_id"] for row in records}) != len(records):
        raise ValueError("Duplicate question IDs in feature table")
    for record in sorted(records, key=lambda row: row["question_id"]):
        candidates = sorted(record["candidates"], key=lambda row: row["document_id"])
        ids = [row["document_id"] for row in candidates]
        if len(ids) < 5 or len(set(ids)) != len(ids) or any(not isinstance(d, str) or not d for d in ids):
            raise ValueError("At least five uniquely identified candidates required per query")
        sizes.append(len(candidates))
        for candidate in candidates:
            values = [candidate["features"][name] for name in features]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Ranking features must be finite numbers")
            matrix.append(values)
            identities.append((record["question_id"], candidate["document_id"]))
            if labels:
                label = candidate["label"]
                if label not in (0, 1):
                    raise ValueError("This reference head expects binary relevance labels")
                targets.append(label)
            elif record.get("partition") in ("public", "private") and "label" in candidate:
                raise ValueError("Public/private inference records must not contain labels")
    return matrix, targets, sizes, identities


def train(feature_path, config_path, output):
    np, lgb = _dependencies()
    records, config = read_jsonl(feature_path), read_json(config_path)
    if not records or any(record.get("partition") != "train" for record in records):
        raise ValueError("Training input must contain TRAIN records only")
    features = config["features"]
    # Validate every record before creating a run or fitting a model.
    _matrix(records, features, labels=True)
    folds = group_folds(records, n_folds=config["n_folds"], seed=config["seed"])
    params = dict(config["parameters"])
    if params.get("objective") != "lambdarank":
        raise ValueError("Expected a fixed LambdaMART/lambdarank configuration")
    if params.get("num_threads", 0) < 1 or config["num_boost_round"] < 1:
        raise ValueError("Positive fixed thread and boosting-round budgets required")
    params.update(seed=config["seed"], deterministic=True, force_col_wise=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"implementation": "reference_head_v1", "input_sha256": sha256(feature_path),
                "configuration_sha256": sha256(config_path), "config": config,
                "query_folds": folds, "packages": {"lightgbm": version("lightgbm"), "numpy": version("numpy")},
                "selection": "fixed configuration; no EVAL input, no early stopping or policy search",
                "upstream_feature_provenance": "Supplied by caller; not independently certified OOF",
                "historical_score_reproduced": False}
    write_json(output / "protocol_before_fit.json", protocol)
    oof, audits = [], []

    def fit(subset):
        x, y, sizes, _ = _matrix(subset, features, labels=True)
        dataset = lgb.Dataset(np.asarray(x, dtype=float), label=y, group=sizes, feature_name=features)
        return lgb.train(params, dataset, num_boost_round=config["num_boost_round"])

    for fold in range(config["n_folds"]):
        fitting = [row for row in records if folds[row["question_id"]] != fold]
        heldout = [row for row in records if folds[row["question_id"]] == fold]
        fitting_groups = {row["group_id"] for row in fitting}
        heldout_groups = {row["group_id"] for row in heldout}
        if fitting_groups & heldout_groups or not fitting or not heldout:
            raise ValueError("Duplicate-group leakage or empty OOF partition")
        model = fit(fitting)
        model.save_model(str(output / f"fold_{fold}.txt"))
        x, _, _, identities = _matrix(heldout, features, labels=False)
        scores = model.predict(np.asarray(x, dtype=float))
        for (qid, doc_id), score in zip(identities, scores):
            oof.append({"question_id": qid, "document_id": doc_id, "score": float(score), "fold": fold})
        audits.append({"fold": fold, "train_queries": len(fitting), "heldout_queries": len(heldout),
                       "train_group_count": len(fitting_groups), "heldout_group_count": len(heldout_groups),
                       "overlapping_groups": 0})
    expected = sum(len(row["candidates"]) for row in records)
    if len(oof) != expected or len({(r["question_id"], r["document_id"]) for r in oof}) != expected:
        raise RuntimeError("Incomplete or duplicate OOF coverage")
    fit(records).save_model(str(output / "final_model.txt"))
    write_json(output / "oof_scores.json", sorted(oof, key=lambda row: (row["question_id"], row["document_id"])))
    write_json(output / "fold_audit.json", audits)
    write_json(output / "readiness.json", {"status": "PASS_REFERENCE_HEAD_TRAINED", "queries": len(records),
               "oof_pairs": expected, "upstream_oof_attested": False, "historical_score_reproduced": False})
    write_manifest(output)
    return {"status": "PASS_REFERENCE_HEAD_TRAINED", "queries": len(records), "oof_pairs": expected}


def predict(feature_path, model_directory):
    np, lgb = _dependencies()
    directory = Path(model_directory)
    protocol = read_json(directory / "protocol_before_fit.json")
    manifest = read_json(directory / "manifest.json")
    listed = [item["path"] for item in manifest["artifacts"]]
    if len(listed) != len(set(listed)) or not {"final_model.txt", "protocol_before_fit.json"} <= set(listed):
        raise ValueError("Incomplete or duplicate model manifest entries")
    for item in manifest["artifacts"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("Unsafe or missing model artifact path")
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Model run integrity check failed")
    records = read_jsonl(feature_path)
    x, _, _, identities = _matrix(records, protocol["config"]["features"], labels=False)
    model = lgb.Booster(model_file=str(directory / "final_model.txt"))
    rows = {}
    for (qid, doc_id), score in zip(identities, model.predict(np.asarray(x, dtype=float))):
        rows.setdefault(qid, []).append((doc_id, float(score)))
    return {q: [doc_id for doc_id, score in sorted(values, key=lambda pair: (-pair[1], pair[0]))[:5]]
            for q, values in sorted(rows.items())}
