import csv
from pathlib import Path
from typing import Any


class BufferedCSVCallback:
    def __init__(
        self,
        filepath: str,
        get_buffer_size=lambda: 100,
    ):

        self.filepath = Path(filepath)
        self.get_buffer_size = get_buffer_size

        self.fieldnames: list[str] = []
        self.buffer: list[dict[str, Any]] = []

        self._header_written = False

    def set_fieldnames(self, fieldnames: list[str]) -> None:
        self.fieldnames = fieldnames

    def add_row(self, row: dict[str, Any]) -> None:
        if not self.fieldnames:
            raise RuntimeError("CSV fieldnames have not been set")

        self.buffer.append(row)

        if len(self.buffer) >= self.get_buffer_size():
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        if not self.fieldnames:
            raise RuntimeError("CSV fieldnames have not been set")

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.filepath.exists()
        file_has_content = file_exists and self.filepath.stat().st_size > 0

        with self.filepath.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

            if not file_has_content:
                writer.writeheader()

            writer.writerows(self.buffer)
            f.flush()

        self.buffer.clear()
        self._header_written = True
