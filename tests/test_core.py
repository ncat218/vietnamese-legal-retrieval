import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

from legalir.artifacts import export_submission, read_json, sha256
from legalir.evaluation import compare, evaluate
from legalir.ranking import select_top5
from legalir.retrieval import BM25, reciprocal_rank_fusion
from legalir.splits import group_folds
from legalir.text import chunks, normalize, select_evidence

ROOT = Path(__file__).resolve().parents[1]


class TextRetrievalTests(unittest.TestCase):
    def test_unicode_equivalence_and_accents(self):
        self.assertEqual(normalize("  lao động\n"), normalize("lao động"))
        docs = [{"document_id": "b", "text": "hợp đồng lao động"},
                {"document_id": "a", "text": "hợp đồng lao động"},
                {"document_id": "c", "text": "đất đai"}]
        hits = BM25(docs).search("lao động", top_k=2)
        self.assertEqual([h["document_id"] for h in hits], ["a", "b"])
        self.assertGreater(hits[0]["score"], 0)

    def test_duplicate_corpus_ids_rejected(self):
        with self.assertRaises(ValueError):
            BM25([{"document_id": "a", "text": "x"}, {"document_id": "a", "text": "y"}])

    def test_rrf_missing_source_and_tie(self):
        result = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "c"]})
        self.assertEqual(result[0]["document_id"], "b")
        self.assertAlmostEqual(result[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(result[1]["source_ranks"], {"bm25": 1})
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"lane": ["a", "a"]})

    def test_evidence_is_label_free_and_stable(self):
        selected = select_evidence("hợp đồng", ["hồ sơ đất", "hợp đồng lao động", "hợp đồng dân sự"])
        self.assertEqual([r["chunk_index"] for r in selected], [1, 2])
        self.assertEqual(chunks("a b c d e f", words=4, overlap=1), ["a b c d", "d e f"])
        with self.assertRaises(ValueError):
            chunks("a", words=2, overlap=2)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / "examples/replay_input.jsonl").read_text(encoding="utf-8"))

    def select(self, **kwargs):
        r = self.record
        return select_top5(r["shortlist"], r["baseline_top5"], r["inherited_scores"], **kwargs)

    def test_outside_shortlist_anchors_and_exact_tail(self):
        result = self.select()
        expected = self.record["baseline_top5"][:2] + list(reversed(self.record["shortlist"]))[:3]
        self.assertEqual(result["top5"], expected)
        self.assertFalse(result["fallback"])

    def test_pilot_falls_back_entire_query(self):
        del self.record["inherited_scores"][self.record["shortlist"][0]]
        result = self.select(mode="05a_cache_safe")
        self.assertEqual(result["top5"], self.record["baseline_top5"])
        self.assertTrue(result["fallback"])
        with self.assertRaisesRegex(ValueError, "backfill"):
            self.select()

    def test_ties_use_original_position_before_id(self):
        for i, d in enumerate(self.record["shortlist"]):
            self.record["inherited_scores"][d] = {"score": 1.0, "candidate_position": 20 - i}
        result = self.select()
        self.assertEqual(result["top5"][2:], list(reversed(self.record["shortlist"]))[:3])

    def test_duplicate_shortlist_or_nonfinite_score_fails(self):
        d = self.record["shortlist"][0]
        self.record["inherited_scores"][d]["score"] = math.nan
        with self.assertRaises(ValueError):
            self.select()
        self.record["inherited_scores"][d]["score"] = 1.0
        self.record["shortlist"][-1] = d
        with self.assertRaises(ValueError):
            self.select()


class EvaluationArtifactTests(unittest.TestCase):
    def test_macro_recall_is_not_micro_recall(self):
        predictions = {"q1": ["a", "x", "y", "z", "w"], "q2": ["b", "x", "y", "z", "w"]}
        gold = {"q1": ["a"], "q2": ["b", "c", "d"]}
        result = evaluate(predictions, gold)
        self.assertAlmostEqual(result["mean_recall"], (1 + 1 / 3) / 2)
        self.assertNotAlmostEqual(result["mean_recall"], 2 / 4)

    def test_incomplete_queries_and_duplicate_predictions_fail(self):
        with self.assertRaises(ValueError):
            evaluate({}, {"q": ["a"]})
        with self.assertRaises(ValueError):
            evaluate({"q": ["a"] * 5}, {"q": ["a"]})

    def test_movement_and_group_bootstrap(self):
        base = {"q": ["a", "b", "c", "d", "e"]}
        trial = {"q": ["a", "b", "c", "d", "f"]}
        result = compare(base, trial, {"q": ["f"]}, groups={"q": "g"}, resamples=10)
        self.assertEqual(result["ci95"], [1, 1])
        self.assertEqual(result["movements"][0]["gained_relevant"], ["f"])
        self.assertEqual(result["improved_queries"], 1)

    def test_groups_do_not_cross_oof_folds(self):
        rows = [{"question_id": str(i), "group_id": str(i // 2)} for i in range(12)]
        folds = group_folds(rows, n_folds=3)
        for i in range(0, 12, 2):
            self.assertEqual(folds[str(i)], folds[str(i + 1)])
        self.assertEqual(set(folds.values()), {0, 1, 2})
        self.assertEqual(folds, group_folds(rows, n_folds=3))

    def test_submission_roundtrip_and_manifest(self):
        predictions = {"q": list("abcde")}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "submission"
            export_submission(predictions, ["q"], target, corpus_ids=list("abcde"))
            with zipfile.ZipFile(target / "submission.zip") as archive:
                self.assertEqual(json.loads(archive.read("submission.json")), {"q": {"answer": list("abcde")}})
            for entry in read_json(target / "manifest.json")["artifacts"]:
                self.assertEqual(sha256(target / entry["path"]), entry["sha256"])
            with self.assertRaises(FileExistsError):
                export_submission(predictions, ["q"], target)
            with self.assertRaises(ValueError):
                export_submission(predictions, ["other"], Path(temp) / "bad")


if __name__ == "__main__":
    unittest.main()
