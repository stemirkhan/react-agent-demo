"""Execute Bash in a mandatory Linux sandbox, never on the host."""

import math
import os
from pathlib import Path, PurePosixPath
import selectors
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


class BashTool:
    name = "bash"

    def __init__(
        self,
        root: Path,
        *,
        output_limit: int = 12000,
        timeout: float = 10,
        readonly_paths: tuple[str, ...] = (),
    ) -> None:
        if type(output_limit) is not int or output_limit < 100:
            raise ValueError("Лимит вывода должен быть целым числом не меньше 100")
        self._validate_timeout(timeout)
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Корень проекта должен быть папкой")
        self.output_limit = output_limit
        self.timeout = timeout
        self.readonly_paths = readonly_paths
        for relative in readonly_paths:
            self._readonly_path(relative)
        self._bubblewrap = shutil.which("bwrap")
        if sys.platform != "linux" or not self._bubblewrap:
            raise ValueError("Для bash нужен Linux с bubblewrap")
        if not Path("/usr/bin/bash").is_file():
            raise ValueError("Для песочницы нужен /usr/bin/bash")

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Выполнить Bash-команду в изолированной рабочей копии проекта. "
                    "Можно читать и менять рабочие файлы; настройки и тесты только "
                    "для чтения. Сеть и домашняя папка недоступны. Вернуть stdout, "
                    "stderr и код завершения; ненулевой код - результат команды."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "minLength": 1},
                        "timeout": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": self.timeout,
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }

    def preflight(self) -> None:
        """Prove namespaces work before spending a request to the model."""
        result = self.execute({"command": ":"})
        if not result["ok"] or result.get("exit_code") != 0:
            raise ValueError(
                "Не удалось запустить Bubblewrap; команды на хосте не выполняются"
            )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(arguments, dict)
            or "command" not in arguments
            or set(arguments) - {"command", "timeout"}
            or not isinstance(arguments["command"], str)
            or not arguments["command"].strip()
            or "\0" in arguments["command"]
        ):
            raise ValueError(
                "bash принимает строковый command и необязательный timeout"
            )
        timeout = arguments.get("timeout", self.timeout)
        self._validate_timeout(timeout)
        if timeout > self.timeout:
            raise ValueError("timeout не может превышать лимит инструмента")
        command = arguments["command"]
        try:
            result = self._run(self._sandbox_command(command), timeout)
        except OSError:
            result = {
                "ok": False,
                "error": {
                    "code": "sandbox_unavailable",
                    "message": "Не удалось запустить изолированную команду",
                },
                "stdout": "",
                "stderr": "",
                "truncated": False,
            }
        if not result.get("error") and result["stderr"].startswith("bwrap:"):
            result["ok"] = False
            result["error"] = {
                "code": "sandbox_unavailable",
                "message": "Не удалось запустить изолированную команду",
            }
        return {"command": command, "cwd": ".", **result}

    def _sandbox_command(self, command: str) -> list[str]:
        assert self._bubblewrap is not None
        argv = [
            self._bubblewrap,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--cap-drop",
            "ALL",
        ]
        for directory in ("/usr", "/lib", "/lib64", "/bin"):
            if Path(directory).exists():
                argv.extend(["--ro-bind", directory, directory])
        argv.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--bind",
                str(self.root),
                "/project",
            ]
        )
        for relative in self.readonly_paths:
            argv.extend(
                [
                    "--ro-bind",
                    str(self._readonly_path(relative)),
                    f"/project/{relative}",
                ]
            )
        environment = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/project",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "LANG": "C.UTF-8",
        }
        for key, value in environment.items():
            argv.extend(["--setenv", key, value])
        return [*argv, "--chdir", "/project", "/usr/bin/bash", "-c", command]

    def _run(self, argv: list[str], timeout: float) -> dict[str, Any]:
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = False
        timed_out = False
        deadline = time.monotonic() + timeout
        with (
            subprocess.Popen(
                argv,
                cwd=self.root,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                bufsize=0,
            ) as process,
            selectors.DefaultSelector() as selector,
        ):
            assert process.stdout is not None and process.stderr is not None
            for name, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            try:
                while selector.get_map() or process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if timed_out:
                            break
                        timed_out = True
                        self._kill(process)
                        deadline = time.monotonic() + 1
                    for key, _ in selector.select(timeout=min(0.1, max(0, remaining))):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        buffer = buffers[key.data]
                        available = self.output_limit - len(buffer)
                        buffer.extend(chunk[:available])
                        truncated |= len(chunk) > available
            finally:
                self._kill(process)
                process.wait()
            result: dict[str, Any] = {
                "ok": not timed_out,
                "exit_code": process.returncode,
                "truncated": truncated,
            }
        for name, buffer in buffers.items():
            result[name] = buffer.decode("utf-8", errors="replace").replace(
                "/project", "."
            )
        if timed_out:
            result["error"] = {
                "code": "timeout",
                "message": "Команда превысила лимит времени",
            }
        return result

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _readonly_path(self, relative: str) -> Path:
        if not isinstance(relative, str):
            raise ValueError("Путь только для чтения должен быть строкой")
        parts = PurePosixPath(relative)
        if (
            not relative
            or relative == "."
            or parts.is_absolute()
            or ".." in parts.parts
            or str(parts) != relative
        ):
            raise ValueError("Разрешены только относительные пути без обхода папок")
        candidate = self.root
        for component in parts.parts:
            candidate /= component
            if candidate.is_symlink():
                raise ValueError("Символические ссылки не разрешены")
        candidate = candidate.resolve(strict=True)
        if not candidate.is_relative_to(self.root) or not (
            candidate.is_file() or candidate.is_dir()
        ):
            raise ValueError("Недопустимый путь только для чтения")
        return candidate

    @staticmethod
    def _validate_timeout(timeout: Any) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "Таймаут инструмента должен быть положительным конечным числом"
            )
