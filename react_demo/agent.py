"""The lesson's central loop: model -> tool -> result -> model."""

from copy import deepcopy
import json
from typing import Any

from .client import DeepSeekClient
from .prompts import PromptTemplates
from .recording import Recorder
from .tools import BashTool, EditTool, ReadFileTool


class Agent:
    def __init__(
        self,
        client: DeepSeekClient,
        tools: list[ReadFileTool | BashTool | EditTool],
        recorder: Recorder,
        *,
        max_steps: int = 8,
        max_tool_calls: int = 16,
    ) -> None:
        if any(type(n) is not int or n < 1 for n in (max_steps, max_tool_calls)):
            raise ValueError("Лимиты должны быть положительными целыми числами")
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Имена инструментов не должны повторяться")
        self.recorder = recorder
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    def run(self, task: str, *, system: str = PromptTemplates.SYSTEM) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        definitions = [tool.definition() for tool in self.tools.values()]
        used_ids: set[str] = set()
        try:
            for step in range(1, self.max_steps + 1):
                self.recorder.emit(
                    "model.request", step=step, messages=messages, tools=definitions
                )
                message = self.client.complete(
                    deepcopy(messages), deepcopy(definitions)
                )
                self.recorder.emit("model.response", step=step, message=message)
                calls = self._validate(message, used_ids)
                messages.append(deepcopy(message))

                if not calls:
                    answer = message.get("content")
                    if not isinstance(answer, str) or not answer.strip():
                        raise RuntimeError("Модель не вернула ни вызова, ни ответа")
                    self.recorder.emit("agent.final", step=step, answer=answer)
                    return answer
                if len(used_ids) + len(calls) > self.max_tool_calls:
                    raise RuntimeError("Достигнут лимит вызовов инструментов")

                for call in calls:
                    result = self._execute(call, step)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    used_ids.add(call["id"])
            raise RuntimeError(
                "Достигнут лимит обращений к модели без финального ответа"
            )
        except Exception as exc:
            self.recorder.emit(
                "agent.error", error_type=type(exc).__name__, message=str(exc)
            )
            raise

    def _execute(self, call: dict[str, Any], step: int) -> dict[str, Any]:
        name, raw = call["function"]["name"], call["function"]["arguments"]
        self.recorder.emit(
            "tool.call", step=step, call_id=call["id"], name=name, arguments=raw
        )
        try:
            arguments = json.loads(raw)
            if not isinstance(arguments, dict):
                raise ValueError("Аргументы должны быть JSON-объектом")
            tool = self.tools.get(name)
            if tool is None:
                raise ValueError("Инструмент не разрешен")
            result = tool.execute(arguments)
        except (OSError, UnicodeError):
            result = {
                "ok": False,
                "error": {
                    "code": "unavailable_file",
                    "message": "Разрешенный файл недоступен",
                },
            }
        except ValueError as exc:
            result = {
                "ok": False,
                "error": {"code": "invalid_call", "message": str(exc)},
            }
        self.recorder.emit(
            "tool.result", step=step, call_id=call["id"], name=name, result=result
        )
        return result

    @staticmethod
    def _validate(message: dict[str, Any], used_ids: set[str]) -> list[dict[str, Any]]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise RuntimeError("Недопустимое сообщение модели")
        if not isinstance(message.get("content"), (str, type(None))):
            raise RuntimeError("Недопустимое содержимое сообщения")
        if not isinstance(message.get("reasoning_content"), (str, type(None))):
            raise RuntimeError("Недопустимый reasoning_content")
        calls = message.get("tool_calls")
        if calls is None:
            return []
        if not isinstance(calls, list):
            raise RuntimeError("Недопустимый список вызовов")
        ids = set(used_ids)
        for call in calls:
            if not isinstance(call, dict) or call.get("type") != "function":
                raise RuntimeError("Недопустимый вызов инструмента")
            call_id, function = call.get("id"), call.get("function")
            if not isinstance(call_id, str) or not call_id or call_id in ids:
                raise RuntimeError("ID вызова пустой или повторяется")
            if not isinstance(function, dict) or not all(
                isinstance(function.get(key), str) for key in ("name", "arguments")
            ):
                raise RuntimeError("Недопустимое имя или аргументы инструмента")
            ids.add(call_id)
        return calls
