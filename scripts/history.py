from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(
        self,
        json_path: str | Path,
        csv_path: str | Path,
        records: list[dict[str, Any]] | None = None,
    ):
        self.json_path = Path(json_path)
        self.csv_path = Path(csv_path)
        self.records = list(records or [])

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(dict(record))
        self.save()

    def save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json()
        self._write_csv()

    def _write_json(self) -> None:
        temporary_path = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=2, ensure_ascii=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, self.json_path)

    def _write_csv(self) -> None:
        temporary_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        fieldnames = self._fieldnames()

        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(self.records)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, self.csv_path)

    def _fieldnames(self) -> list[str]:
        fieldnames: list[str] = []
        for record in self.records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        return fieldnames
