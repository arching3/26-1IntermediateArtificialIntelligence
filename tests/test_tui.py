import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tui


class FakeWindow:
    def __init__(
        self,
        height: int = 20,
        width: int = 100,
        keys: list[int] | None = None,
    ):
        self.height = height
        self.width = width
        self.lines: dict[int, str] = {}
        self.keys = list(keys or [])
        self.getch_calls = 0

    def getmaxyx(self):
        return self.height, self.width

    def addstr(self, y, x, text, attr=0):
        current = self.lines.get(y, "")
        if len(current) < x:
            current += " " * (x - len(current))
        self.lines[y] = current[:x] + text

    def refresh(self):
        pass

    def nodelay(self, enabled):
        pass

    def getch(self):
        self.getch_calls += 1
        return self.keys.pop(0)


class PathBrowserTests(unittest.TestCase):
    def test_path_suggestions_include_all_entries_and_hide_dotfiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(12):
                (root / f"script_{index:02d}.py").write_text("", encoding="utf-8")
            (root / "models").mkdir()
            (root / ".hidden.py").write_text("", encoding="utf-8")

            suggestions = tui.path_suggestions(f"{root}{os.sep}")

        labels = [suggestion.label for suggestion in suggestions]
        self.assertEqual(len(labels), 13)
        self.assertIn("script_11.py", labels)
        self.assertNotIn(".hidden.py", labels)
        self.assertTrue(suggestions[0].is_dir)

    def test_parent_path_removes_partial_name_before_parent_directory(self):
        self.assertEqual(tui.parent_path("data/lo"), f"data{os.sep}")
        self.assertEqual(tui.parent_path(f"data{os.sep}"), f".{os.sep}")

    def test_path_browser_renders_one_path_column(self):
        window = FakeWindow(height=8)
        suggestion = tui.PathSuggestion(
            label="train.py",
            completion="/different/location/train.py",
            is_dir=False,
        )

        tui.draw_path_browser(window, 1, [suggestion], 0, window.width)

        rendered = "\n".join(window.lines.values())
        self.assertIn("FILE train.py", rendered)
        self.assertNotIn(suggestion.completion, rendered)


class ArgumentFormTests(unittest.TestCase):
    def test_imported_model_choices_are_parsed(self):
        specs, _ = tui.parse_script_args("scripts/train.py")
        model_spec = next(spec for spec in specs if spec.dest == "model_name")

        self.assertEqual(
            model_spec.choices,
            ["mlp", "simple_cnn", "resnet", "mlp_mixer"],
        )

    def test_rebuild_fields_preserves_existing_values(self):
        specs, _ = tui.parse_script_args("scripts/train.py")
        fields = tui.make_fields("train", [])
        fields[0].value = "scripts/train.py"
        fields[1].value = "cifar100"

        rebuilt = tui.rebuild_fields("train", specs, fields)

        self.assertEqual(rebuilt[0].value, "scripts/train.py")
        self.assertEqual(rebuilt[1].value, "cifar100")
        model_field = next(field for field in rebuilt if field.label == "--model_name")
        self.assertEqual(
            model_field.choices,
            ("mlp", "simple_cnn", "resnet", "mlp_mixer"),
        )

    def test_build_command_injects_dataset_once(self):
        specs, _ = tui.parse_script_args("scripts/train.py")
        fields = tui.make_fields("train", specs)
        fields[0].value = "scripts/train.py"
        fields[1].value = "fashion"

        command, error = tui.build_command("train", fields, specs)

        self.assertIsNone(error)
        self.assertEqual(command.count("--dataset"), 1)
        dataset_index = command.index("--dataset")
        self.assertEqual(command[dataset_index + 1], "fashion")

    def test_model_selection_populates_train_form(self):
        specs, _ = tui.parse_script_args("scripts/train.py")
        fields = tui.make_fields("train", specs)
        selection = tui.ModelConfigSelection(
            name="mlp",
            parameters={"flatten": True},
            source_type="default",
            source_path="models/config.json",
            training={
                "dataset_path": "cifar100",
                "epochs": 7,
                "batch_size": 32,
            },
            action="clone_configuration",
        )
        try:
            tui.apply_model_selection("train", fields, selection)

            self.assertEqual(fields[1].value, "cifar100")
            values = {
                field.arg.dest: field.value
                for field in fields
                if field.arg
            }
            self.assertEqual(values["model_name"], "mlp")
            self.assertEqual(values["epochs"], "7")
            self.assertEqual(values["batch_size"], "32")
            self.assertTrue(values["model_config_path"].endswith(".json"))
        finally:
            tui.cleanup_selection(selection)

    def test_eval_form_contains_only_storage_path(self):
        fields = tui.make_fields("eval", [])

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].label, "storage path")
        self.assertTrue(fields[0].required)

    def test_eval_command_uses_fixed_script_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            fields = tui.make_fields("eval", [])
            fields[0].value = directory

            command, error = tui.build_command("eval", fields, [])

        self.assertIsNone(error)
        self.assertEqual(Path(command[1]), tui.EVAL_SCRIPT_PATH)
        self.assertEqual(command[2], "--storage")
        self.assertEqual(command[3], str(Path(directory).resolve()))


class ErrorScreenTests(unittest.TestCase):
    def test_finished_screen_ignores_keys_until_q(self):
        window = FakeWindow(keys=[ord("x"), ord("q")])

        with patch("tui.draw_run"):
            tui.draw_finished(window, "Train", ["error"], 1)

        self.assertEqual(window.getch_calls, 2)
        rendered = "\n".join(window.lines.values())
        self.assertIn("Press q to return to menu.", rendered)

    def test_fatal_error_screen_ignores_keys_until_q(self):
        window = FakeWindow(keys=[10, ord("q")])

        with patch("tui.draw_run"):
            tui.draw_fatal_error(window, RuntimeError("failure"))

        self.assertEqual(window.getch_calls, 2)
        rendered = "\n".join(window.lines.values())
        self.assertIn("Press q to close the TUI.", rendered)


if __name__ == "__main__":
    unittest.main()
