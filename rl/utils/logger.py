"""Minimal dependency-free logger that writes metrics to CSV and prints progress."""

import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, **row: float) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
