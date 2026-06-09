import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.eval import print_evaluation
from scripts.evaluation import load_history, summarize_history, write_json_atomic


class HistorySummaryTests(unittest.TestCase):
    def test_print_evaluation_includes_model_parameters(self):
        result = {
            "storage": "storage/example",
            "checkpoint": "storage/example/checkpoints/best.pt",
            "model": {
                "name": "mlp",
                "parameters": {
                    "hidden_sizes": [128, 64],
                    "use_dropout": True,
                    "dropout_p": 0.2,
                },
            },
            "best_epoch": 3,
            "best_monitor": "train_f1",
            "best_metric": 0.9,
            "test_metrics": {"loss": 0.1, "accuracy": 0.95, "f1": 0.94},
            "history_average": {"train_loss": 0.2, "train_f1": 0.89},
            "last_epoch": {"epoch": 3},
            "total_epochs": 3,
            "total_training_seconds": 1.0,
        }

        output = StringIO()
        with redirect_stdout(output):
            print_evaluation(result)

        rendered = output.getvalue()
        self.assertIn("Model Parameters (mlp)", rendered)
        self.assertIn('"hidden_sizes": [', rendered)
        self.assertIn('"use_dropout": true', rendered)

    def test_history_average_excludes_epoch_and_sample_counts(self):
        records = [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "train_samples": 100,
                "valid_f1": 0.7,
                "valid_samples": 20,
                "epoch_seconds": 2.0,
            },
            {
                "epoch": 2,
                "train_loss": 0.3,
                "train_samples": 100,
                "valid_f1": 0.9,
                "valid_samples": 20,
                "epoch_seconds": 3.0,
            },
        ]

        summary = summarize_history(records)

        self.assertEqual(summary["average"]["train_loss"], 0.4)
        self.assertEqual(summary["average"]["valid_f1"], 0.8)
        self.assertNotIn("epoch", summary["average"])
        self.assertNotIn("train_samples", summary["average"])
        self.assertEqual(summary["total_epochs"], 2)
        self.assertEqual(summary["total_training_seconds"], 5.0)
        self.assertEqual(summary["last_epoch"]["epoch"], 2)

    def test_history_json_round_trip(self):
        records = [{"epoch": 1, "train_loss": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            write_json_atomic(path, records)
            loaded = load_history(path)

        self.assertEqual(loaded, records)

    def test_empty_history_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_history([])


if __name__ == "__main__":
    unittest.main()
