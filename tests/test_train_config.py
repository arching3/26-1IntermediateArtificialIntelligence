import unittest

from scripts.train import (
    TrainingConfig,
    _run_config_payload,
    parse_config,
    resolve_model_configuration,
)


class TrainingConfigTests(unittest.TestCase):
    def test_model_specific_cli_arguments_are_removed(self):
        config = parse_config(["--dataset", "mnist", "--model-name", "mlp"])

        self.assertFalse(hasattr(config, "n_layers"))
        self.assertFalse(hasattr(config, "use_dropout"))
        self.assertFalse(hasattr(config, "dropout_p"))
        self.assertFalse(hasattr(config, "flatten"))
        self.assertFalse(hasattr(config, "normalize"))

    def test_default_model_configuration_is_loaded(self):
        config = parse_config(["--dataset", "mnist", "--model-name", "mlp"])

        model_configuration = resolve_model_configuration(config)

        self.assertEqual(model_configuration.name, "mlp")
        self.assertTrue(model_configuration.parameters["flatten"])
        self.assertEqual(model_configuration.source_type, "default")

    def test_run_config_separates_training_and_model_settings(self):
        config = parse_config(["--dataset", "mnist", "--model-name", "mlp"])
        config.model_parameters = {"flatten": True, "hidden_sizes": [32]}
        config.model_source = {"type": "default", "path": "models/config.json"}

        payload = _run_config_payload(config)

        self.assertNotIn("model_parameters", payload["training"])
        self.assertEqual(payload["model"]["name"], "mlp")
        self.assertEqual(payload["model"]["parameters"]["hidden_sizes"], [32])
        self.assertEqual(payload["source"]["type"], "default")


if __name__ == "__main__":
    unittest.main()
