import json
import tempfile
import unittest
from pathlib import Path

from tui_model_config import (
    cleanup_selection,
    discover_previous_runs,
    materialize_selection,
    parse_parameter_value,
    selection_from_default,
    selection_from_run,
)


class TuiModelConfigTests(unittest.TestCase):
    def test_parse_parameter_value_preserves_type(self):
        self.assertEqual(parse_parameter_value("[1, 2]", [3]), [1, 2])
        self.assertEqual(parse_parameter_value("0.25", 0.1), 0.25)
        with self.assertRaises(ValueError):
            parse_parameter_value("1", True)

    def test_previous_run_discovery_and_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "20260610_mnist_mlp"
            run_dir.mkdir()
            payload = {
                "training": {
                    "dataset_path": "mnist",
                    "epochs": 5,
                },
                "model": {
                    "name": "mlp",
                    "parameters": {"flatten": True},
                },
                "source": {"type": "default", "path": "models/config.json"},
            }
            (run_dir / "config.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            runs = discover_previous_runs(directory)
            selection = selection_from_run(runs[0])

        self.assertEqual(len(runs), 1)
        self.assertEqual(selection.name, "mlp")
        self.assertEqual(selection.training["epochs"], 5)

    def test_materialized_selection_preserves_original_source(self):
        selection = selection_from_default("mlp")
        original_source = selection.source_path

        path = materialize_selection(selection)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cleanup_selection(selection)

        self.assertEqual(payload["source"]["type"], "tui_override")
        self.assertEqual(payload["source"]["path"], original_source)
        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
