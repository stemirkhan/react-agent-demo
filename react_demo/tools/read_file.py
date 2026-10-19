"""Read an explicitly allowed lesson file, never an arbitrary path."""

from pathlib import Path, PurePosixPath
from typing import Any


class ReadFileTool:
    name = "read_file"

    def __init__(
        self, root: Path, allowed_files: tuple[str, ...], *, output_limit: int = 12000
    ) -> None:
        if type(output_limit) is not int or output_limit < 100:
            raise ValueError("Лимит вывода должен быть целым числом не меньше 100")
        for relative in allowed_files:
            path = PurePosixPath(relative)
            if (
                not relative
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != relative
            ):
                raise ValueError("Разрешены только относительные пути без обхода папок")
        self.root = Path(root).resolve()
        self.allowed_files = allowed_files
        self.output_limit = output_limit

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Прочитать один файл учебного проекта. Путь от корня проекта.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "enum": list(self.allowed_files)}
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"path"}
            or not isinstance(arguments["path"], str)
        ):
            raise ValueError("read_file принимает только строковый path")
        text = self.resolve_file(arguments["path"]).read_text(encoding="utf-8")
        return {
            "ok": True,
            "path": arguments["path"],
            "content": text[: self.output_limit],
            "truncated": len(text) > self.output_limit,
        }

    def resolve_file(self, relative: str) -> Path:
        if relative not in self.allowed_files:
            raise ValueError("Файл не входит в список разрешенных")
        candidate = self.root
        for component in PurePosixPath(relative).parts:
            candidate /= component
            if candidate.is_symlink():
                raise ValueError("Символические ссылки не разрешены")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("Недопустимый файл")
        return resolved
