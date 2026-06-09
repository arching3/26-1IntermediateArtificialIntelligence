import json
import tempfile
import unittest
from pathlib import Path

from models.config import (
    load_all_model_configs,
    load_model_configuration,
    save_model_config,
    validate_model_name,
)


class ModelConfigTests(unittest.TestCase):
    def test_default_config_uses_underscore_model_names(self):
        configs = load_all_model_configs()

        self.assertEqual(
            tuple(configs),
            ("mlp", "simple_cnn", "resnet", "mlp_mixer"),
        )

    def test_model_name_rejects_dash(self):
        with self.assertRaises(ValueError):
            validate_model_name("mlp-mixer")

    def test_run_snapshot_can_be_loaded(self):
        payload = {
            "training": {"epochs": 3},
            "model": {
                "name": "mlp",
                "parameters": {"flatten": True, "hidden_sizes": [16]},
            },
            "source": {"type": "default", "path": "models/config.json"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_model_configuration("mlp", path)

        self.assertEqual(config.source_type, "previous_run")
        self.assertEqual(config.parameters["hidden_sizes"], [16])

    def test_save_model_config_creates_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            backup = root / "config.backup.json"
            path.write_text(
                json.dumps({"mlp": {"flatten": True}}),
                encoding="utf-8",
            )

            save_model_config(
                "mlp",
                {"flatten": False},
                config_path=path,
                backup_path=backup,
            )

            self.assertTrue(backup.exists())
            self.assertFalse(load_all_model_configs(path)["mlp"]["flatten"])

    def test_tui_override_preserves_original_source(self):
        payload = {
            "training": {},
            "model": {
                "name": "mlp",
                "parameters": {"flatten": True},
            },
            "source": {
                "type": "tui_override",
                "path": "/original/config.json",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "override.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_model_configuration("mlp", path)

        self.assertEqual(config.source_type, "tui_override")
        self.assertEqual(config.source_path, "/original/config.json")


if __name__ == "__main__":
    unittest.main()
