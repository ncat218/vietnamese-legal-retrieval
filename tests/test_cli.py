import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from legalir.cli import main

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def call(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(arguments)
        return json.loads(output.getvalue())

    def test_offline_demo_returns_five_and_is_labeled(self):
        result = self.call(["demo", "--query", "người lao động chấm dứt hợp đồng"])
        self.assertEqual(len(result["results"]), 5)
        self.assertEqual(result["mode"], "illustrative_cpu_bm25")
        self.assertIsNone(result["competition_score"])
        self.assertIn("termination", result["results"][0]["document_id"])

    def test_replay_evaluate_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "replay"
            result = self.call(["replay", "--input", str(ROOT / "examples/replay_input.jsonl"), "--output", str(out)])
            self.assertEqual(result["fallback_queries"], 0)
            metrics = self.call(["evaluate", "--predictions", str(out / "predictions.json"),
                                 "--gold", str(ROOT / "examples/gold.json")])
            self.assertEqual(metrics["mean_recall"], 1.0)
            exported = self.call(["export", "--predictions", str(out / "predictions.json"),
                                  "--query-ids", str(ROOT / "examples/query_ids.json"),
                                  "--corpus-ids", str(ROOT / "examples/corpus_ids.json"),
                                  "--output", str(Path(temporary) / "export")])
            self.assertEqual(exported["status"], "PASS_SUBMISSION_FORMAT")


if __name__ == "__main__":
    unittest.main()
