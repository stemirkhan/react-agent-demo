"""Direct DeepSeek API client; one HTTP request, no automatic retries."""

from copy import deepcopy
import json
import math
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .recording import Recorder

ENDPOINT = "https://api.deepseek.com/chat/completions"
RESPONSE_LIMIT = 2 * 1024 * 1024
DIAGNOSTIC_LIMIT = 8192


class ClientError(RuntimeError):
    """Safe error text; bounded response details stay in the local journal."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # Never forward the bearer key to another endpoint.
        return None


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        recorder: Recorder,
        *,
        max_tokens: int = 800,
        timeout: float = 30,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or "\n" in api_key
            or "\r" in api_key
        ):
            raise ValueError("Не задан корректный DEEPSEEK_API_KEY")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Нужно явно выбрать модель")
        if (
            type(max_tokens) is not int
            or max_tokens < 1
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Некорректные лимиты API")
        self.model = model.strip()
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key
        self._recorder = recorder
        self._opener = build_opener(NoRedirect())

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "thinking": {"type": "disabled"},
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        self._recorder.emit("api.request", endpoint=ENDPOINT, body=payload)
        request = Request(
            ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw = self._read(request)
        if len(raw) > RESPONSE_LIMIT:
            self._fail("Ответ API превысил лимит размера")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError):
            self._fail("API вернул не JSON", body=self._diagnostic(raw))
        self._recorder.emit("api.response", body=data)
        return self._message(data)

    def _read(self, request: Request) -> bytes:
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(RESPONSE_LIMIT + 1)
                if not isinstance(raw, bytes):
                    self._fail("HTTP-клиент вернул недопустимый тип ответа")
                return raw
        except HTTPError as exc:
            try:
                body = self._diagnostic(exc.read(DIAGNOSTIC_LIMIT))
            except OSError:
                body = "Не удалось прочитать тело ошибки HTTP"
            finally:
                exc.close()
            self._fail(f"DeepSeek вернул HTTP {exc.code}", status=exc.code, body=body)
        except (URLError, TimeoutError, OSError):
            self._fail("Ошибка соединения или таймаут DeepSeek; повтор не выполнялся")

    def _message(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or data.get("error"):
            self._fail("API вернул ошибку; подробности в журнале")
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            self._fail("API не вернул choices[0]")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            self._fail("Ответ модели обрезан лимитом токенов; повтор не выполнялся")
        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            self._fail("API прервал ответ; подробности в журнале")
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), (str, type(None)))
            or (
                message.get("tool_calls") is not None
                and not isinstance(message["tool_calls"], list)
            )
        ):
            self._fail("API не вернул корректное сообщение модели")
        return deepcopy(message)

    def _fail(self, message: str, **details: Any) -> NoReturn:
        self._recorder.emit("api.error", message=message, **details)
        raise ClientError(message) from None

    @staticmethod
    def _diagnostic(raw: bytes) -> str:
        return raw[:DIAGNOSTIC_LIMIT].decode("utf-8", errors="replace")
