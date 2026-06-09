#!/usr/bin/env python3
"""Curses TUI for running ML train/eval/inference scripts."""

from __future__ import annotations

import ast
import curses
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models import MODEL_NAMES
from models.config import save_model_config
from tui_model_config import (
    ModelConfigSelection,
    cleanup_selection,
    discover_previous_runs,
    materialize_selection,
    parse_parameter_value,
    selection_from_checkpoint,
    selection_from_default,
    selection_from_run,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency.
    psutil = None


DATASET_ARG_CANDIDATES = (
    "--dataset",
    "--dataset_path",
    "--dataset-path",
    "--data",
    "--data_path",
    "--data-path",
    "--input",
    "--input_path",
    "--input-path",
)
MODEL_ARG_CANDIDATES = (
    "--model",
    "--model_path",
    "--model-path",
    "--model_fn",
    "--weights",
    "--weight_path",
    "--weight-path",
    "--checkpoint",
    "--checkpoint_path",
    "--checkpoint-path",
    "--ckpt",
)
DATASET_CHOICES = ("cifar100", "fashion", "mnist")
EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "eval.py"


@dataclass
class ArgSpec:
    flags: list[str]
    dest: str
    required: bool = False
    default: Any = None
    type_name: str = ""
    action: str = ""
    choices: list[str] = field(default_factory=list)
    help: str = ""

    @property
    def primary_flag(self) -> str:
        long_flags = [flag for flag in self.flags if flag.startswith("--")]
        return long_flags[0] if long_flags else self.flags[0]

    @property
    def is_bool_flag(self) -> bool:
        return self.action in {"store_true", "store_false"}

    @property
    def default_text(self) -> str:
        if self.default is None:
            return ""
        return str(self.default)


@dataclass
class FormField:
    label: str
    value: str = ""
    required: bool = False
    arg: ArgSpec | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathSuggestion:
    label: str
    completion: str
    is_dir: bool


def safe_literal(node: ast.AST, constants: dict[str, Any] | None = None) -> Any:
    constants = constants or {}
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return constants.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return node.attr
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "set", "tuple"}
            and len(node.args) == 1
        ):
            value = safe_literal(node.args[0], constants)
            if isinstance(value, str) and value == ast.unparse(node.args[0]):
                return ast.unparse(node)
            if isinstance(value, dict):
                value = value.keys()
            try:
                return {"list": list, "set": set, "tuple": tuple}[node.func.id](value)
            except (TypeError, ValueError):
                pass
        if isinstance(node, ast.Call):
            return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"
        return ast.unparse(node) if hasattr(ast, "unparse") else None


def literal_default(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def keyword_map(
    node: ast.Call,
    constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg:
            values[kw.arg] = safe_literal(kw.value, constants)
    return values


def dest_from_flags(flags: list[str], explicit_dest: str | None = None) -> str:
    if explicit_dest:
        return explicit_dest
    long_flags = [flag for flag in flags if flag.startswith("--")]
    source = long_flags[0] if long_flags else flags[0]
    return source.lstrip("-").replace("-", "_")


def collect_literal_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        value = safe_literal(value_node, constants)
        if isinstance(value, str) and value == ast.unparse(value_node):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def collect_imported_constants(
    tree: ast.Module,
    script_path: Path,
) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
            continue

        module_parts = node.module.split(".")
        search_roots = (script_path.parent, script_path.parent.parent)
        candidates = tuple(
            candidate
            for search_root in search_roots
            for module_path in (search_root.joinpath(*module_parts),)
            for candidate in (module_path.with_suffix(".py"), module_path / "__init__.py")
        )
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            continue

        try:
            source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            source_constants = collect_literal_constants(source_tree)
        except (OSError, SyntaxError, UnicodeError):
            continue

        for alias in node.names:
            if alias.name in source_constants:
                constants[alias.asname or alias.name] = source_constants[alias.name]
    return constants


def parse_argparse_ast(script_path: Path) -> list[ArgSpec]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    specs: list[ArgSpec] = []
    constants = collect_imported_constants(tree, script_path)
    constants.update(collect_literal_constants(tree))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        flags = []
        for arg in node.args:
            value = safe_literal(arg, constants)
            if isinstance(value, str) and value.startswith("-"):
                flags.append(value)
        if not flags:
            continue

        kws = keyword_map(node, constants)
        default = kws.get("default")
        for kw in node.keywords:
            if kw.arg == "default":
                default = literal_default(kw.value)
        action = str(kws.get("action", ""))
        type_value = kws.get("type", "")
        if isinstance(type_value, str):
            type_name = type_value
        else:
            type_name = str(type_value) if type_value else ""
        choices = kws.get("choices", [])
        if isinstance(choices, (list, tuple)):
            choices = [str(choice) for choice in choices]
        else:
            choices = []

        specs.append(
            ArgSpec(
                flags=flags,
                dest=dest_from_flags(flags, kws.get("dest")),
                required=bool(kws.get("required", False)),
                default=default,
                type_name=type_name,
                action=action,
                choices=choices,
                help=str(kws.get("help", "")) if kws.get("help") else "",
            )
        )

    return specs


def parse_help_options(script_path: Path) -> list[ArgSpec]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path.resolve()), "--help"],
            cwd=str(script_path.resolve().parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    specs: list[ArgSpec] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        option_part = line.split("  ", 1)[0]
        flags = []
        for chunk in option_part.split(","):
            flag = chunk.strip().split(" ", 1)[0]
            if flag.startswith("-"):
                flags.append(flag)
        if flags:
            specs.append(ArgSpec(flags=flags, dest=dest_from_flags(flags)))
    return specs


def parse_script_args(script_path_text: str) -> tuple[list[ArgSpec], str]:
    path = Path(script_path_text).expanduser()
    if not path.exists() or not path.is_file():
        return [], f"Script not found: {path}"

    try:
        specs = parse_argparse_ast(path)
        if specs:
            return specs, f"Loaded {len(specs)} argparse options from AST."
    except Exception as exc:
        ast_error = str(exc)
    else:
        ast_error = "No argparse add_argument calls found."

    specs = parse_help_options(path)
    if specs:
        return specs, f"Loaded {len(specs)} options from --help fallback."
    return [], f"Could not parse arguments. {ast_error}"


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text[: max(0, width - 1)]


def addstr_safe(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = win.getmaxyx()
    if 0 <= y < height and 0 <= x < width:
        try:
            win.addstr(y, x, clip(text, width - x), attr)
        except curses.error:
            pass


def draw_box(win: curses.window, title: str = "") -> None:
    win.erase()
    win.box()
    if title:
        addstr_safe(win, 0, 2, f" {title} ", curses.A_BOLD)


def path_suggestions(
    value: str,
    limit: int | None = None,
) -> list[PathSuggestion]:
    raw_value = value
    expanded = os.path.expanduser(raw_value)
    if not raw_value:
        typed_dir = ""
        list_dir = "."
        prefix = ""
    elif raw_value.endswith(os.sep):
        typed_dir = raw_value
        list_dir = expanded
        prefix = ""
    else:
        typed_dir = os.path.dirname(raw_value)
        list_dir = os.path.dirname(expanded) or "."
        prefix = os.path.basename(raw_value)

    try:
        entries = list(os.scandir(list_dir))
    except OSError:
        return []

    show_hidden = prefix.startswith(".")
    matches: list[tuple[int, str, PathSuggestion]] = []
    for entry in entries:
        name = entry.name
        if not show_hidden and name.startswith("."):
            continue
        if prefix and not name.startswith(prefix):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        typed_base = typed_dir if typed_dir else ""
        completion = os.path.join(typed_base, name) if typed_base else name
        if raw_value.startswith("~"):
            user_home = os.path.expanduser("~")
            expanded_completion = os.path.abspath(os.path.expanduser(completion))
            if expanded_completion.startswith(user_home):
                completion = "~" + expanded_completion[len(user_home) :]
        if is_dir:
            completion += os.sep
        label = name + (os.sep if is_dir else "")
        suggestion = PathSuggestion(label, completion, is_dir)
        matches.append((0 if is_dir else 1, label.lower(), suggestion))

    matches.sort()
    suggestions = [suggestion for _, _, suggestion in matches]
    return suggestions if limit is None else suggestions[:limit]


def parent_path(value: str) -> str:
    raw_value = value or "."
    use_absolute_path = os.path.isabs(raw_value) or raw_value.startswith("~")
    working_value = os.path.expanduser(raw_value) if use_absolute_path else raw_value

    if raw_value.endswith(os.sep):
        target = os.path.dirname(os.path.normpath(working_value))
    else:
        target = os.path.dirname(working_value)
    target = target or "."

    if raw_value.startswith("~"):
        home = os.path.expanduser("~")
        if target == home:
            return f"~{os.sep}"
        if target.startswith(home + os.sep):
            return "~" + target[len(home) :] + os.sep
    return target.rstrip(os.sep) + os.sep


def draw_path_browser(
    stdscr: curses.window,
    y: int,
    suggestions: list[PathSuggestion],
    selected: int,
    width: int,
) -> None:
    height, _ = stdscr.getmaxyx()
    visible_count = max(0, height - y - 1)
    start = max(0, selected - visible_count + 1)
    visible_suggestions = suggestions[start : start + visible_count]
    position = f" [{selected + 1}/{len(suggestions)}]" if suggestions else " [0/0]"

    addstr_safe(stdscr, y, 0, " " * (width - 1))
    addstr_safe(
        stdscr,
        y,
        0,
        "Path browser: Up/Down select | Tab/Right complete | Left parent | Enter accept"
        + position,
    )
    for row, suggestion in enumerate(visible_suggestions):
        index = start + row
        line_y = y + 1 + row
        addstr_safe(stdscr, line_y, 0, " " * (width - 1))
        attr = curses.A_REVERSE if index == selected else 0
        marker = "DIR " if suggestion.is_dir else "FILE"
        addstr_safe(stdscr, line_y, 2, f"{marker:<4} {suggestion.label}", attr)


def visible_input(value: str, width: int) -> tuple[str, int]:
    available = max(1, width - 1)
    if len(value) <= available:
        return value, len(value)
    visible = value[-available:]
    return visible, len(visible)


def prompt_line(stdscr: curses.window, prompt: str, initial: str = "", path_browser: bool = False) -> str | None:
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    curses.noecho()
    value = initial
    selected_suggestion = 0
    while True:
        height, width = stdscr.getmaxyx()
        prompt_y = max(1, height - 12) if path_browser else height - 3
        for clear_y in range(prompt_y, height):
            addstr_safe(stdscr, clear_y, 0, " " * (width - 1))
        suggestions = path_suggestions(value) if path_browser else []
        if selected_suggestion >= len(suggestions):
            selected_suggestion = max(0, len(suggestions) - 1)
        addstr_safe(stdscr, prompt_y, 0, prompt)
        displayed_value, cursor_x = visible_input(value, width)
        addstr_safe(stdscr, prompt_y + 1, 0, displayed_value)
        if path_browser:
            draw_path_browser(stdscr, prompt_y + 3, suggestions, selected_suggestion, width)
        try:
            stdscr.move(prompt_y + 1, min(cursor_x, width - 2))
        except curses.error:
            pass
        stdscr.refresh()
        key = stdscr.getch()
        if key in (27,):
            curses.noecho()
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            return None
        if path_browser and key in (curses.KEY_UP,):
            if suggestions:
                selected_suggestion = (selected_suggestion - 1) % len(suggestions)
            continue
        if path_browser and key in (curses.KEY_DOWN,):
            if suggestions:
                selected_suggestion = (selected_suggestion + 1) % len(suggestions)
            continue
        if path_browser and key in (9, curses.KEY_RIGHT):
            if suggestions:
                value = suggestions[selected_suggestion].completion
                selected_suggestion = 0
            continue
        if path_browser and key == curses.KEY_LEFT:
            value = parent_path(value)
            selected_suggestion = 0
            continue
        if key in (curses.KEY_ENTER, 10, 13):
            curses.noecho()
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            return value
        if key in (curses.KEY_BACKSPACE, 127, 8):
            value = value[:-1]
            selected_suggestion = 0
        elif 32 <= key <= 126:
            value += chr(key)
            selected_suggestion = 0


def menu(stdscr: curses.window, title: str, items: list[str]) -> int | None:
    selected = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        addstr_safe(stdscr, 1, 2, title, curses.A_BOLD)
        addstr_safe(stdscr, height - 2, 2, "Up/Down or j/k: move | Enter: select | q: quit")
        visible_rows = max(1, height - 6)
        start = max(0, selected - visible_rows + 1)
        for row, item in enumerate(items[start : start + visible_rows]):
            index = start + row
            attr = curses.A_REVERSE if index == selected else 0
            addstr_safe(stdscr, 4 + row, 4, item, attr)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(items)
        elif key in (curses.KEY_ENTER, 10, 13):
            return selected


def prompt_choice(
    stdscr: curses.window,
    title: str,
    choices: tuple[str, ...],
    initial: str = "",
) -> str | None:
    selected = choices.index(initial) if initial in choices else 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        addstr_safe(stdscr, 1, 2, title, curses.A_BOLD)
        addstr_safe(stdscr, height - 2, 2, "Up/Down or j/k: move | Enter: select | Esc/q: cancel")
        visible_rows = max(1, height - 6)
        start = max(0, selected - visible_rows + 1)
        for row, choice in enumerate(choices[start : start + visible_rows]):
            index = start + row
            attr = curses.A_REVERSE if index == selected else 0
            addstr_safe(stdscr, 4 + row, 4, choice, attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(choices)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(choices)
        elif key in (curses.KEY_ENTER, 10, 13):
            return choices[selected]


def message_screen(stdscr: curses.window, title: str, message: str) -> None:
    stdscr.erase()
    height, _ = stdscr.getmaxyx()
    addstr_safe(stdscr, 1, 2, title, curses.A_BOLD)
    for row, line in enumerate(message.splitlines()):
        addstr_safe(stdscr, 3 + row, 2, line)
    addstr_safe(stdscr, height - 2, 2, "Press any key to continue.")
    stdscr.refresh()
    stdscr.getch()


def choose_model_configuration(
    stdscr: curses.window,
) -> ModelConfigSelection | None:
    source = prompt_choice(
        stdscr,
        "Model configuration source",
        ("default", "previous_run", "checkpoint"),
    )
    if source is None:
        return None

    try:
        if source == "default":
            model_name = prompt_choice(
                stdscr,
                "Select model",
                MODEL_NAMES,
            )
            if model_name is None:
                return None
            selection = selection_from_default(model_name)
        elif source == "previous_run":
            runs = discover_previous_runs()
            if not runs:
                message_screen(
                    stdscr,
                    "Previous runs",
                    "No storage/*/config.json files were found.",
                )
                return None
            index = menu(
                stdscr,
                "Select previous run",
                [run.label for run in runs],
            )
            if index is None:
                return None
            selection = selection_from_run(runs[index])
        else:
            checkpoint_path = prompt_line(
                stdscr,
                "checkpoint path: ",
                path_browser=True,
            )
            if not checkpoint_path:
                return None
            selection = selection_from_checkpoint(checkpoint_path)
    except Exception as error:
        message_screen(stdscr, "Configuration error", str(error))
        return None

    return edit_model_configuration(stdscr, selection)


def edit_model_configuration(
    stdscr: curses.window,
    selection: ModelConfigSelection,
) -> ModelConfigSelection | None:
    parameter_names = list(selection.parameters)
    if not parameter_names:
        message_screen(
            stdscr,
            "Model configuration",
            f"Model '{selection.name}' has no parameters.",
        )
        return selection

    selected = 0
    status = (
        f"source={selection.source_type} | Enter: edit | "
        "u: use | s: save as default | q: cancel"
    )
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        addstr_safe(
            stdscr,
            1,
            2,
            f"Model Config: {selection.name}",
            curses.A_BOLD,
        )
        addstr_safe(stdscr, 2, 2, status)
        addstr_safe(
            stdscr,
            height - 2,
            2,
            "Up/Down: move | Enter: edit JSON value | u: use | s: save default | q: cancel",
        )

        visible_rows = max(1, height - 6)
        start = max(0, selected - visible_rows + 1)
        for row, name in enumerate(parameter_names[start : start + visible_rows]):
            index = start + row
            value = json.dumps(selection.parameters[name], ensure_ascii=True)
            attr = curses.A_REVERSE if index == selected else 0
            addstr_safe(
                stdscr,
                4 + row,
                2,
                f"{name:<28} {clip(value, max(10, width - 34))}",
                attr,
            )
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(parameter_names)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(parameter_names)
        elif key in (curses.KEY_ENTER, 10, 13):
            name = parameter_names[selected]
            current = selection.parameters[name]
            raw_value = prompt_line(
                stdscr,
                f"{name} (JSON): ",
                json.dumps(current, ensure_ascii=True),
            )
            if raw_value is None:
                continue
            try:
                value = parse_parameter_value(raw_value, current)
            except ValueError as error:
                status = str(error)
            else:
                cleanup_selection(selection)
                selection.parameters[name] = value
                selection.dirty = True
                status = f"Updated {name}."
        elif key == ord("s"):
            try:
                save_model_config(selection.name, selection.parameters)
            except Exception as error:
                status = f"Save failed: {error}"
            else:
                selection.source_type = "default"
                selection.source_path = str(
                    Path(__file__).resolve().parent / "models" / "config.json"
                )
                selection.dirty = False
                cleanup_selection(selection)
                status = "Saved as model default. Previous config was backed up."
        elif key == ord("u"):
            actions = ["use_model_parameters"]
            if selection.training:
                actions.append("clone_configuration")
            if selection.resume_path and not selection.dirty:
                actions.extend(("resume_training", "evaluate"))
            action = prompt_choice(
                stdscr,
                "Use configuration",
                tuple(actions),
                selection.action,
            )
            if action is not None:
                selection.action = action
                return selection


def infer_path_arg(specs: list[ArgSpec], candidates: tuple[str, ...]) -> ArgSpec | None:
    normalized = {candidate.replace("-", "_"): candidate for candidate in candidates}
    for spec in specs:
        names = spec.flags + [f"--{spec.dest}", f"--{spec.dest.replace('_', '-')}"]
        for name in names:
            if name in candidates or name.replace("-", "_") in normalized:
                return spec
    return None


def make_fields(mode: str, specs: list[ArgSpec]) -> list[FormField]:
    if mode == "eval":
        return [FormField("storage path", required=True)]

    script_labels = {
        "train": "train script path",
        "print_inference": "print script path",
    }
    fields = [FormField(script_labels.get(mode, "script path"), required=True)]
    if mode == "train":
        fields.append(FormField("dataset", "mnist", required=True, choices=DATASET_CHOICES))
    else:
        fields.append(FormField("pretrained weight path"))
        fields.append(FormField("input data path"))

    for spec in specs:
        if mode == "train" and infer_path_arg([spec], DATASET_ARG_CANDIDATES):
            continue
        if mode == "print_inference" and (
            infer_path_arg([spec], MODEL_ARG_CANDIDATES)
            or infer_path_arg([spec], DATASET_ARG_CANDIDATES)
        ):
            continue
        if spec.is_bool_flag:
            default = "yes" if spec.action == "store_true" and spec.default else "no"
            fields.append(FormField(spec.primary_flag, default, spec.required, spec))
        else:
            fields.append(
                FormField(
                    spec.primary_flag,
                    spec.default_text,
                    spec.required,
                    spec,
                    tuple(spec.choices),
                )
            )
    fields.append(FormField("extra args"))
    return fields


def rebuild_fields(
    mode: str,
    specs: list[ArgSpec],
    current_fields: list[FormField],
) -> list[FormField]:
    values = {field.label: field.value for field in current_fields}
    fields = make_fields(mode, specs)
    for field in fields:
        if field.label in values:
            field.value = values[field.label]
    return fields


def set_field_value(
    fields: list[FormField],
    dest: str,
    value: Any,
) -> bool:
    for field in fields:
        if field.arg and field.arg.dest == dest:
            if isinstance(value, bool):
                field.value = "yes" if value else "no"
            elif value is not None:
                field.value = str(value)
            return True
    return False


def apply_model_selection(
    mode: str,
    fields: list[FormField],
    selection: ModelConfigSelection | None,
) -> None:
    if selection is None:
        return

    set_field_value(fields, "model_name", selection.name)
    if mode == "train" and selection.training.get("dataset_path"):
        fields[1].value = str(selection.training["dataset_path"])

    if selection.action in {"clone_configuration", "resume_training"}:
        for name, value in selection.training.items():
            if name not in {
                "model_name",
                "dataset_path",
                "resume",
                "run_name",
                "model_config_path",
                "input_shape",
                "num_classes",
                "run_dir",
            }:
                set_field_value(fields, name, value)

    if selection.action == "resume_training" and selection.resume_path:
        set_field_value(fields, "resume", selection.resume_path)
        set_field_value(fields, "model_config_path", selection.resume_path)
    else:
        set_field_value(
            fields,
            "model_config_path",
            materialize_selection(selection),
        )

    if selection.action == "evaluate" and selection.evaluation_path:
        model_arg = infer_path_arg(
            [field.arg for field in fields if field.arg],
            MODEL_ARG_CANDIDATES,
        )
        if model_arg:
            set_field_value(fields, model_arg.dest, selection.evaluation_path)


def field_hint(field: FormField) -> str:
    if field.choices:
        return "choices=" + ",".join(field.choices)
    if not field.arg:
        return "Enter to edit"
    parts = []
    if field.arg.required:
        parts.append("required")
    if field.arg.type_name:
        parts.append(f"type={field.arg.type_name}")
    if field.arg.choices:
        parts.append("choices=" + ",".join(field.arg.choices))
    if field.arg.action:
        parts.append(f"action={field.arg.action}")
    return " | ".join(parts) or "optional"


def is_path_field(field: FormField) -> bool:
    if field.choices:
        return False
    text = field.label.lower().replace("-", "_")
    if any(token in text for token in ("path", "script", "weight", "checkpoint", "input data", "dataset")):
        return True
    if field.arg:
        dest = field.arg.dest.lower()
        return any(token in dest for token in ("path", "model_fn", "weight", "checkpoint", "data", "dataset", "input"))
    return False


def form_screen(
    stdscr: curses.window,
    mode: str,
    model_selection: ModelConfigSelection | None = None,
) -> tuple[list[str], str] | None:
    fields = make_fields(mode, [])
    specs: list[ArgSpec] = []
    selected = 0
    status = (
        "Select a completed training storage directory."
        if mode == "eval"
        else "Enter script path, then press r to auto-load argparse options."
    )

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        addstr_safe(stdscr, 1, 2, f"{mode} setup", curses.A_BOLD)
        addstr_safe(stdscr, 2, 2, status)
        addstr_safe(stdscr, height - 2, 2, "Enter: edit | r: reload args | x: execute | Esc/q: back")

        visible_rows = max(1, height - 6)
        start = max(0, selected - visible_rows + 1)
        for row, field in enumerate(fields[start : start + visible_rows]):
            index = start + row
            attr = curses.A_REVERSE if index == selected else 0
            req = "*" if field.required else " "
            label = clip(f"{req} {field.label}", 28)
            value = clip(field.value, max(10, width - 58))
            hint = clip(field_hint(field), 24)
            addstr_safe(stdscr, 4 + row, 2, f"{label:<28} {value:<{max(10, width - 58)}} {hint}", attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(fields)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(fields)
        elif key in (curses.KEY_ENTER, 10, 13):
            if fields[selected].choices:
                edited = prompt_choice(
                    stdscr,
                    f"Select {fields[selected].label}",
                    fields[selected].choices,
                    fields[selected].value,
                )
            else:
                edited = prompt_line(
                    stdscr,
                    f"{fields[selected].label}: ",
                    fields[selected].value,
                    path_browser=is_path_field(fields[selected]),
                )
            if edited is not None:
                fields[selected].value = edited
                if mode != "eval" and selected == 0:
                    specs, status = parse_script_args(edited)
                    fields = rebuild_fields(mode, specs, fields)
                    apply_model_selection(mode, fields, model_selection)
                    selected = 0
        elif key == ord("r"):
            if mode == "eval":
                status = "Eval uses only the selected storage directory."
                continue
            specs, status = parse_script_args(fields[0].value)
            fields = rebuild_fields(mode, specs, fields)
            apply_model_selection(mode, fields, model_selection)
            selected = min(selected, len(fields) - 1)
        elif key == ord("x"):
            command, error = build_command(mode, fields, specs)
            if error:
                status = error
            else:
                return command, fields[0].value


def truthy(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def build_command(mode: str, fields: list[FormField], specs: list[ArgSpec]) -> tuple[list[str], str | None]:
    if mode == "eval":
        storage_path = fields[0].value.strip()
        if not storage_path:
            return [], "storage path is required."
        storage = Path(storage_path).expanduser()
        if not storage.exists() or not storage.is_dir():
            return [], f"storage path does not exist: {storage}"
        if not EVAL_SCRIPT_PATH.is_file():
            return [], f"eval script does not exist: {EVAL_SCRIPT_PATH}"
        return [
            sys.executable,
            str(EVAL_SCRIPT_PATH),
            "--storage",
            str(storage.resolve()),
        ], None

    script_path = fields[0].value.strip()
    if not script_path:
        return [], "script path is required."
    script = Path(script_path).expanduser()
    if not script.exists() or not script.is_file():
        return [], f"script path does not exist: {script}"

    for field in fields:
        if field.required and not field.value.strip():
            return [], f"{field.label} is required."

    command = [sys.executable, str(script.resolve())]
    consumed_arg_dests: set[str] = set()

    if mode == "train":
        dataset = fields[1].value.strip()
        dataset_arg = infer_path_arg(specs, DATASET_ARG_CANDIDATES)
        if dataset and dataset_arg:
            command.extend([dataset_arg.primary_flag, dataset])
            consumed_arg_dests.add(dataset_arg.dest)
    else:
        weight = fields[1].value.strip()
        input_data = fields[2].value.strip()
        model_arg = infer_path_arg(specs, MODEL_ARG_CANDIDATES)
        data_arg = infer_path_arg(specs, DATASET_ARG_CANDIDATES)
        if weight and model_arg:
            command.extend([model_arg.primary_flag, weight])
            consumed_arg_dests.add(model_arg.dest)
        if input_data and data_arg and data_arg.dest not in consumed_arg_dests:
            command.extend([data_arg.primary_flag, input_data])
            consumed_arg_dests.add(data_arg.dest)

    for field in fields:
        if not field.arg or field.arg.dest in consumed_arg_dests:
            continue
        value = field.value.strip()
        if field.arg.is_bool_flag:
            enabled = truthy(value)
            if (field.arg.action == "store_true" and enabled) or (field.arg.action == "store_false" and not enabled):
                command.append(field.arg.primary_flag)
        elif value:
            command.extend([field.arg.primary_flag, value])

    extra = fields[-1].value.strip()
    if extra:
        try:
            command.extend(shlex.split(extra))
        except ValueError as exc:
            return [], f"Invalid extra args: {exc}"

    return command, None


def enqueue_output(proc: subprocess.Popen[str], output_queue: queue.Queue[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        output_queue.put(line.rstrip("\n"))
    proc.stdout.close()


def gpu_lines() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return ["GPU: nvidia-smi unavailable"]
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
            check=False,
        )
    except Exception as exc:
        return [f"GPU: unavailable ({exc})"]
    if proc.returncode != 0:
        return ["GPU: unavailable"]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(f"GPU {parts[0]} {parts[1]}: {parts[2]}% | {parts[3]}/{parts[4]} MiB")
    return rows or ["GPU: no devices"]


def resource_lines(proc: subprocess.Popen[str]) -> list[str]:
    lines = []
    if psutil is None:
        lines.append("psutil not installed: CPU/memory per-process unavailable")
    else:
        try:
            process = psutil.Process(proc.pid)
            cpu = process.cpu_percent(interval=None)
            mem = process.memory_info().rss / (1024 * 1024)
            children = process.children(recursive=True)
            child_cpu = sum(child.cpu_percent(interval=None) for child in children)
            child_mem = sum(child.memory_info().rss for child in children) / (1024 * 1024)
            vm = psutil.virtual_memory()
            lines.extend(
                [
                    f"PID: {proc.pid}",
                    f"Process CPU: {cpu + child_cpu:.1f}%",
                    f"Process RSS: {mem + child_mem:.1f} MiB",
                    f"System CPU: {psutil.cpu_percent(interval=None):.1f}%",
                    f"System Mem: {vm.percent:.1f}% ({vm.used // (1024 ** 2)} / {vm.total // (1024 ** 2)} MiB)",
                ]
            )
        except Exception as exc:
            lines.append(f"Resource read failed: {exc}")
    lines.extend(gpu_lines())
    return lines


def run_command_screen(stdscr: curses.window, title: str, command: list[str], cwd: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    output_queue: queue.Queue[str] = queue.Queue()
    logs = [f"$ {shlex.join(command)}"]

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd or None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception as exc:
        logs.append(f"Failed to start process: {exc}")
        draw_finished(stdscr, title, logs, -1)
        return

    reader = threading.Thread(target=enqueue_output, args=(proc, output_queue), daemon=True)
    reader.start()

    last_resource_at = 0.0
    resources = ["Collecting resource data..."]
    exit_code: int | None = None
    while True:
        while True:
            try:
                logs.append(output_queue.get_nowait())
            except queue.Empty:
                break
        if len(logs) > 2000:
            logs = logs[-1000:]

        now = time.time()
        if now - last_resource_at > 1.0:
            resources = resource_lines(proc)
            last_resource_at = now

        draw_run(stdscr, title, logs, resources, proc.poll())
        if proc.poll() is not None:
            exit_code = proc.returncode
            while True:
                try:
                    logs.append(output_queue.get_nowait())
                except queue.Empty:
                    break
            break
        time.sleep(0.1)

    draw_finished(stdscr, title, logs, exit_code if exit_code is not None else -1)


def draw_run(stdscr: curses.window, title: str, logs: list[str], resources: list[str], exit_code: int | None) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 12 or width < 50:
        addstr_safe(stdscr, 0, 0, "Terminal too small. Resize to at least 50x12.")
        stdscr.refresh()
        return

    top_h = max(6, int(height * 0.68))
    log_win = stdscr.derwin(top_h, width, 0, 0)
    res_win = stdscr.derwin(height - top_h, width, top_h, 0)
    draw_box(log_win, title)
    draw_box(res_win, "htop")

    log_height, log_width = log_win.getmaxyx()
    tail = logs[-(log_height - 2) :]
    for idx, line in enumerate(tail):
        addstr_safe(log_win, 1 + idx, 2, line, 0)

    state = "running" if exit_code is None else f"exit={exit_code}"
    addstr_safe(log_win, 0, max(2, log_width - len(state) - 4), f" {state} ")
    for idx, line in enumerate(resources[: max(1, height - top_h - 2)]):
        addstr_safe(res_win, 1 + idx, 2, line)
    stdscr.refresh()


def draw_finished(stdscr: curses.window, title: str, logs: list[str], exit_code: int) -> None:
    while True:
        result = "completed" if exit_code == 0 else "failed"
        draw_run(
            stdscr,
            title,
            logs,
            [f"Process {result} with exit code {exit_code}."],
            exit_code,
        )
        height, _ = stdscr.getmaxyx()
        addstr_safe(stdscr, height - 1, 2, "Press q to return to menu.")
        stdscr.refresh()
        stdscr.nodelay(False)
        if stdscr.getch() in (ord("q"), ord("Q"), 27):
            return


def draw_fatal_error(stdscr: curses.window, error: BaseException) -> None:
    logs = traceback.format_exception(type(error), error, error.__traceback__)
    flattened_logs = [
        line
        for block in logs
        for line in block.rstrip("\n").splitlines()
    ]

    while True:
        draw_run(
            stdscr,
            "TUI Error",
            flattened_logs,
            ["The TUI encountered an unexpected error."],
            -1,
        )
        height, _ = stdscr.getmaxyx()
        addstr_safe(stdscr, height - 1, 2, "Press q to close the TUI.")
        stdscr.refresh()
        stdscr.nodelay(False)
        if stdscr.getch() in (ord("q"), ord("Q"), 27):
            return


def app(stdscr: curses.window) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()

    model_selection: ModelConfigSelection | None = None
    try:
        while True:
            selection_status = (
                f" [{model_selection.name}:{model_selection.action}]"
                if model_selection
                else ""
            )
            choice = menu(
                stdscr,
                f"ML Project TUI{selection_status}",
                ["train", "eval", "print_inference", "model_config", "quit"],
            )
            if choice is None or choice == 4:
                return
            if choice == 3:
                selected_config = choose_model_configuration(stdscr)
                if selected_config is not None:
                    cleanup_selection(model_selection)
                    model_selection = selected_config
                continue

            mode = ["train", "eval", "print_inference"][choice]
            result = form_screen(stdscr, mode, model_selection)
            if result is None:
                continue
            command, script_path = result
            command_cwd = (
                str(EVAL_SCRIPT_PATH.parent)
                if mode == "eval"
                else str(Path(script_path).expanduser().parent)
            )
            run_command_screen(
                stdscr,
                mode.replace("_", " ").title(),
                command,
                cwd=command_cwd,
            )
    except Exception as error:
        draw_fatal_error(stdscr, error)
    finally:
        cleanup_selection(model_selection)


def main() -> None:
    curses.wrapper(app)


if __name__ == "__main__":
    main()
