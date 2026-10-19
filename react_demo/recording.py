"""Append-only JSONL journal, redaction and validated replay."""

from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from types import TracebackType
from typing import Any
from uuid import uuid4

JsonObject = dict[str, Any]


class SecretRedactor:
    _SENSITIVE_KEYS = frozenset(
        {"authorization", "api_key", "apikey", "access_token", "deepseek_api_key"}
    )
    _KEY_PATTERN = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{16,}")

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, dict):
            return self._redact_mapping(value)
        if isinstance(value, (tuple, list)):
            return [self.redact(item) for item in value]
        return value

    def _redact_mapping(self, value: JsonObject) -> JsonObject:
        result: JsonObject = {}
        for key, item in value.items():
            redacted_key = self._redact_text(key) if isinstance(key, str) else key
            if str(key).lower() in self._SENSITIVE_KEYS:
                result[redacted_key] = "[REDACTED]"
            else:
                result[redacted_key] = self.redact(item)
        return result

    def _redact_text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")
        return self._KEY_PATTERN.sub("[REDACTED]", value)


class Recorder:
    @staticmethod
    def new_path(directory: str | Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(directory) / f"{timestamp}-{uuid4().hex[:10]}.jsonl"

    def __init__(
        self,
        filename: str | Path,
        *,
        secrets: Iterable[str] = (),
        on_event: Callable[[JsonObject], None] | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self._redactor = redactor if redactor is not None else SecretRedactor(secrets)
        self._on_event = on_event
        self._sequence = 0

    def emit(self, event: str, **data: Any) -> None:
        record = {
            "schema_version": 1,
            "sequence": self._sequence + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": self._redactor.redact(data),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        self._sequence = record["sequence"]
        if self._on_event is not None:
            self._on_event(record)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def read(filename: str | Path) -> Iterator[JsonObject]:
        with Path(filename).open(encoding="utf-8") as stream:
            previous = 0
            for line_number, line in enumerate(stream, 1):
                record = Recorder._parse_record(line, line_number, previous + 1)
                previous = record["sequence"]
                yield record

    @staticmethod
    def _parse_record(
        line: str, line_number: int, expected_sequence: int
    ) -> JsonObject:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Поврежден JSONL, строка {line_number}") from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("sequence") != expected_sequence
            or not isinstance(record.get("event"), str)
            or not isinstance(record.get("data"), dict)
        ):
            raise ValueError(f"Недопустимое событие, строка {line_number}")
        return record

    @staticmethod
    def show(record: dict[str, object], *, raw: bool = False) -> None:
        event, data = record["event"], record["data"]
        if not isinstance(data, dict):
            return
        if raw:
            text = json.dumps(record, ensure_ascii=False)
        elif event == "model.request":
            text = f"\nШаг {data['step']}: обращение к модели"
        elif event == "tool.call":
            text = f"Инструмент: {data['name']}({data['arguments']})"
        elif event == "tool.result":
            text = json.dumps(data["result"], ensure_ascii=False, indent=2)
        elif event == "agent.final":
            text = f"\nОтвет:\n{data['answer']}"
        elif event in ("run.failed", "agent.error"):
            text = f"Ошибка: {data.get('message', '')}"
        elif event == "log.path":
            text = f"Журнал: {data['path']}"
        else:
            return
        print("".join(char for char in text if char in "\n\t" or char.isprintable()))
