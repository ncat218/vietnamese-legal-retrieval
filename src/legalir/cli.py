"""Run `legalir --help` for the CPU demo, policy replay and reference trainer."""
import argparse
from importlib.resources import files
import json
from pathlib import Path

from .artifacts import export_submission, read_json, read_jsonl, write_json, write_manifest
from .evaluation import evaluate
from .ranking import select_top5, validate_ids
from .retrieval import BM25
from .text import chunks, select_evidence


def demo(query):
    corpus_path = files("legalir").joinpath("data/demo_corpus.jsonl")
    documents = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = BM25(documents)
    results = []
    for hit in index.search(query, top_k=5):
        doc = index.documents[hit["document_id"]]
        results.append(dict(hit, title=doc["title"], evidence=select_evidence(query, chunks(doc["text"]))))
    return {"mode": "illustrative_cpu_bm25", "competition_score": None,
            "notice": "Fictional example documents; not legal advice or a replay of competition models.",
            "query": query, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vietnamese legal retrieval — auditable building blocks")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="Offline BM25 search over fictional Vietnamese examples")
    d.add_argument("--query", default="người lao động chấm dứt hợp đồng")
    s = commands.add_parser("search", help="BM25 search over your own JSONL documents")
    s.add_argument("--corpus", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    r = commands.add_parser("replay", help="Apply 05A/06A final selection to supplied frozen scores")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--mode", choices=["05a_cache_safe", "06a_strict"], default="06a_strict")
    e = commands.add_parser("evaluate", help="Query-macro metrics with exact query coverage")
    e.add_argument("--predictions", required=True)
    e.add_argument("--gold", required=True)
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--output")
    t = commands.add_parser("train", help="Fit an optional OOF LambdaMART reference head on TRAIN features")
    t.add_argument("--features", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--output", required=True)
    p = commands.add_parser("predict", help="Top5 with a trained reference head; not the historical pipeline")
    p.add_argument("--features", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    x = commands.add_parser("export", help="Validate IDs and export the official-format Top5 JSON/ZIP")
    x.add_argument("--predictions", required=True)
    x.add_argument("--query-ids", required=True)
    x.add_argument("--corpus-ids", required=True)
    x.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.query)
        elif args.command == "search":
            result = BM25(read_jsonl(args.corpus)).search(args.query, top_k=args.top_k)
        elif args.command == "replay":
            records = read_jsonl(args.input)
            if not records or len({r["question_id"] for r in records}) != len(records):
                raise ValueError("Nonempty, uniquely identified replay queries required")
            validate_ids([record["question_id"] for record in records])
            audit, predictions = {}, {}
            for record in records:
                selection = select_top5(record["shortlist"], record["baseline_top5"], record["inherited_scores"], mode=args.mode)
                audit[record["question_id"]] = selection
                predictions[record["question_id"]] = selection["top5"]
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            write_json(out / "predictions.json", predictions)
            write_json(out / "selection_audit.json", audit)
            write_json(out / "readiness.json", {"status": "PASS_POLICY_REPLAY", "mode": args.mode,
                       "queries": len(records), "neural_inference": False, "historical_score_reproduced": False})
            write_manifest(out)
            result = {"queries": len(records), "fallback_queries": sum(row["fallback"] for row in audit.values())}
        elif args.command == "evaluate":
            result = evaluate(read_json(args.predictions), read_json(args.gold), k=args.k)
            if args.output:
                if Path(args.output).exists():
                    raise ValueError("Choose a new metrics output path")
                write_json(args.output, result)
        elif args.command == "train":
            from .training import train
            result = train(args.features, args.config, args.output)
        elif args.command == "predict":
            from .training import predict
            result = predict(args.features, args.model)
            if Path(args.output).exists():
                raise ValueError("Choose a new predictions output path")
            write_json(args.output, result)
        else:
            export_submission(read_json(args.predictions), read_json(args.query_ids), args.output,
                              corpus_ids=read_json(args.corpus_ids))
            result = {"status": "PASS_SUBMISSION_FORMAT", "competition_uploaded": False}
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"legalir: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
