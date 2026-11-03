import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from legalir.artifacts import read_json
from legalir.training import _matrix, predict, train

ROOT = Path(__file__).resolve().parents[1]
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


class FeatureSafetyTests(unittest.TestCase):
    def test_labels_and_ids_are_rejected_as_features(self):
        record = {"question_id": "q", "candidates": []}
        with self.assertRaises(ValueError):
            _matrix([record], ["gold_hit"], labels=True)
        with self.assertRaises(ValueError):
            _matrix([record], ["document_id"], labels=True)


@unittest.skipUnless(HAS_LIGHTGBM, "Install the train extra for the real LightGBM integration test")
class TrainingIntegrationTests(unittest.TestCase):
    def test_native_lightgbm_oof_models_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            result = train(ROOT / "examples/train_features.jsonl", ROOT / "configs/reference_ranker.json", out)
            self.assertEqual(result["oof_pairs"], 24 * 12)
            audits = read_json(out / "fold_audit.json")
            self.assertEqual(len(audits), 3)
            self.assertTrue(all(a["overlapping_groups"] == 0 for a in audits))
            oof = read_json(out / "oof_scores.json")
            self.assertEqual(len({(r["question_id"], r["document_id"]) for r in oof}), 288)
            protocol = read_json(out / "protocol_before_fit.json")
            self.assertEqual(protocol["packages"]["lightgbm"], "4.6.0")
            predictions = predict(ROOT / "examples/train_features.jsonl", out)
            self.assertEqual(len(predictions), 24)
            self.assertTrue(all(len(set(ids)) == 5 for ids in predictions.values()))
            with self.assertRaises(FileExistsError):
                train(ROOT / "examples/train_features.jsonl", ROOT / "configs/reference_ranker.json", out)

    def test_eval_partition_cannot_enter_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = json.loads((ROOT / "examples/train_features.jsonl").read_text().splitlines()[0])
            record["partition"] = "eval"
            path = Path(temporary) / "eval.jsonl"
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                train(path, ROOT / "configs/reference_ranker.json", Path(temporary) / "bad")


if __name__ == "__main__":
    unittest.main()
