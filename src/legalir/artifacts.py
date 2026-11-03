"""Atomic JSON exports and reproducible file identities."""
import hashlib
import json
from pathlib import Path
import zipfile

from .ranking import validate_ids


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(directory):
    directory = Path(directory)
    entries = [{"path": p.relative_to(directory).as_posix(), "size_bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in sorted(directory.rglob("*"))
               if p.is_file() and p != directory / "manifest.json"]
    write_json(directory / "manifest.json", {"format": "legalir-public-v1", "artifacts": entries})


def export_submission(predictions, expected_query_ids, directory, *, corpus_ids=None):
    expected = list(expected_query_ids)
    validate_ids(expected)
    if not expected or set(predictions) != set(expected):
        raise ValueError("Submission must cover exactly the expected question IDs")
    for q, ids in predictions.items():
        validate_ids(ids, count=5)
        if corpus_ids is not None and not set(ids) <= set(corpus_ids):
            raise ValueError(f"Unknown document ID for {q}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = {q: {"answer": predictions[q]} for q in expected}
    write_json(directory / "submission.json", payload)
    with zipfile.ZipFile(directory / "submission.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        archive.write(directory / "submission.json", "submission.json")
    with zipfile.ZipFile(directory / "submission.zip") as archive:
        if archive.namelist() != ["submission.json"] or json.loads(archive.read("submission.json")) != payload:
            raise RuntimeError("Submission ZIP roundtrip failed")
    write_manifest(directory)
