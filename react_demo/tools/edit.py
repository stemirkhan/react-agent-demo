"""Replace one unique text fragment in an explicitly allowed lesson file."""

import difflib
import hashlib
import os
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile
from typing import Any

from .read_file import ReadFileTool


class EditTool:
    name = "edit"

    def __init__(
        self, root: Path, allowed_files: tuple[str, ...], *, output_limit: int = 12000
    ) -> None:
        self.files = ReadFileTool(root, allowed_files, output_limit=output_limit)

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Заменить один точный фрагмент в разрешенном файле проекта. "
                    "old_text должен встречаться ровно один раз; новые файлы не создаются."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "enum": list(self.files.allowed_files),
                        },
                        "old_text": {"type": "string", "minLength": 1},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"path", "old_text", "new_text"}
            or not all(isinstance(value, str) for value in arguments.values())
        ):
            raise ValueError(
                "edit принимает только строковые path, old_text и new_text"
            )
        relative = arguments["path"]
        old_text, new_text = arguments["old_text"], arguments["new_text"]
        if not old_text or old_text == new_text:
            raise ValueError("old_text не должен быть пустым или совпадать с new_text")
        path = self.files.resolve_file(relative)
        original = path.read_bytes()
        before = original.decode("utf-8")
        start = before.find(old_text)
        if start < 0:
            raise ValueError("old_text не найден; файл не изменен")
        if before.find(old_text, start + 1) >= 0:
            raise ValueError("old_text встречается несколько раз; уточни фрагмент")
        after = before[:start] + new_text + before[start + len(old_text) :]
        updated = after.encode("utf-8")
        diff = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=relative,
                tofile=relative,
            )
        )

        # A failed write leaves the original intact. Recheck before replacing it.
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(updated)
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            if (
                self.files.resolve_file(relative) != path
                or path.read_bytes() != original
            ):
                raise ValueError(
                    "Файл изменился во время правки; повторно прочитай его"
                )
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {
            "ok": True,
            "path": relative,
            "changed": True,
            "diff": diff[: self.files.output_limit],
            "before_sha256": hashlib.sha256(original).hexdigest(),
            "after_sha256": hashlib.sha256(updated).hexdigest(),
            "truncated": len(diff) > self.files.output_limit,
        }
