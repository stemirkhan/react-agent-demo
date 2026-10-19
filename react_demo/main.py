"""Small entry point: local case check, live run, or saved-log replay."""

import argparse
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

from .agent import Agent
from .client import DeepSeekClient
from .prompts import PromptTemplates
from .recording import Recorder
from .tools import (
    CASE_FILES,
    CASE_ROOT,
    CASE_TEST_COMMAND,
    BashTool,
    EditTool,
    ReadFileTool,
)


class Lesson:
    def run(self, argv: Sequence[str] | None = None) -> int:
        args = self._parser().parse_args(argv)
        if args.command == "replay":
            try:
                for record in Recorder.read(args.path):
                    Recorder.show(record, raw=args.json)
                return 0
            except (OSError, ValueError) as exc:
                Recorder.show({"event": "run.failed", "data": {"message": str(exc)}})
                return 2

        key = os.environ.get("DEEPSEEK_API_KEY", "") if args.command == "run" else ""
        if args.command == "run" and (
            not args.live
            or not key.strip()
            or not args.model.strip()
            or "\n" in key
            or "\r" in key
        ):
            print(
                "Для запуска нужны --live, --model и корректный DEEPSEEK_API_KEY",
                file=sys.stderr,
            )
            return 2
        filename = Path(args.output) if args.output else Recorder.new_path("runs")
        try:
            with Recorder(filename, secrets=(key,), on_event=Recorder.show) as recorder:
                try:
                    reader = ReadFileTool(CASE_ROOT, CASE_FILES)
                    if args.command == "run":
                        workspace = filename.with_suffix(".project")
                        self._copy_case(reader, workspace)
                        recorder.emit("run.workspace", path=str(workspace))
                        reader = ReadFileTool(workspace, CASE_FILES)
                    recorder.emit(
                        "run.started",
                        mode=args.command,
                        python=sys.version.split()[0],
                        cwd=".",
                        file_hashes=self._hashes(reader),
                    )
                    if args.command == "case":
                        result = self._check_case(reader)
                        recorder.emit("tool.result", name="case", result=result)
                        expected = result.get(
                            "exit_code"
                        ) == 1 and "300 != 0" in result.get("stderr", "")
                        recorder.emit("run.completed", expected_failure=expected)
                        return 0 if expected else 1

                    shell = BashTool(
                        reader.root,
                        readonly_paths=("shop/settings.json", "tests"),
                    )
                    shell.preflight()
                    recorder.emit(
                        "run.settings",
                        provider="deepseek",
                        model=args.model,
                        thinking="disabled",
                        max_steps=args.max_steps,
                        max_tool_calls=args.max_tool_calls,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    client = DeepSeekClient(
                        key,
                        args.model,
                        recorder,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    agent = Agent(
                        client,
                        [reader, shell, EditTool(reader.root, ("shop/delivery.py",))],
                        recorder,
                        max_steps=args.max_steps,
                        max_tool_calls=args.max_tool_calls,
                    )
                    agent.run(PromptTemplates.DELIVERY_TASK)
                    recorder.emit("run.completed", file_hashes=self._hashes(reader))
                    return 0
                except Exception as exc:
                    recorder.emit(
                        "run.failed", error_type=type(exc).__name__, message=str(exc)
                    )
                    return 1
                except KeyboardInterrupt:
                    recorder.emit(
                        "run.failed",
                        error_type="KeyboardInterrupt",
                        message="Запуск прерван пользователем",
                    )
                    return 130
        except (OSError, ValueError) as exc:
            Recorder.show({"event": "run.failed", "data": {"message": str(exc)}})
            return 2
        finally:
            Recorder.show({"event": "log.path", "data": {"path": str(filename)}})

    @staticmethod
    def _copy_case(reader: ReadFileTool, target: Path) -> None:
        target.mkdir(mode=0o700)
        for name in reader.allowed_files:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reader.resolve_file(name), destination)

    @staticmethod
    def _hashes(reader: ReadFileTool) -> dict[str, str]:
        return {
            name: hashlib.sha256(reader.resolve_file(name).read_bytes()).hexdigest()
            for name in reader.allowed_files
        }

    @classmethod
    def _check_case(cls, reader: ReadFileTool) -> dict[str, Any]:
        with TemporaryDirectory(prefix="react-lesson-case-") as temporary:
            root = Path(temporary) / "project"
            cls._copy_case(reader, root)
            shell = BashTool(root, readonly_paths=("shop/settings.json", "tests"))
            shell.preflight()
            return shell.execute({"command": CASE_TEST_COMMAND})

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="ReAct: поиск причины и исправление падающего теста"
        )
        commands = parser.add_subparsers(dest="command", required=True)
        case = commands.add_parser("case", help="Проверить учебный дефект без API")
        case.add_argument("--output")
        live = commands.add_parser("run", help="Настоящий запуск через DeepSeek API")
        live.add_argument("--model", required=True)
        live.add_argument("--live", action="store_true")
        live.add_argument("--output")
        for flag, default, maximum in (
            ("max-steps", 8, 16),
            ("max-tool-calls", 16, 32),
            ("max-tokens", 800, 4096),
            ("timeout", 30, 120),
        ):
            live.add_argument(
                f"--{flag}",
                type=int,
                default=default,
                choices=range(1, maximum + 1),
                metavar="N",
            )
        replay = commands.add_parser(
            "replay", help="Показать сохраненный журнал без API"
        )
        replay.add_argument("path", type=Path)
        replay.add_argument("--json", action="store_true")
        return parser
