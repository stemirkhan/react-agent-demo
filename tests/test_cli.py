from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from react_demo.main import Lesson
from react_demo.recording import Recorder
from react_demo.tools import BashTool, CASE_FILES, CASE_ROOT, CASE_TEST_COMMAND


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.filename = Path(self.temporary.name) / "run.jsonl"
        self.workspace = self.filename.with_suffix(".project")
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client_patch = patch("react_demo.main.DeepSeekClient")
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        # A dummy key must never permit accidental HTTP in these tests.
        self.network_patch = patch(
            "urllib.request.OpenerDirector.open",
            side_effect=AssertionError("Network forbidden in unit tests"),
        )
        self.network = self.network_patch.start()
        self.addCleanup(self.network_patch.stop)
        # Ordinary CLI tests exercise orchestration, not OS namespace support.
        self.bash_init_patch = patch.object(
            BashTool, "__init__", autospec=True, side_effect=self.prepare_mocked_bash
        )
        self.bash_init_patch.start()
        self.addCleanup(self.bash_init_patch.stop)
        self.preflight_patch = patch.object(BashTool, "preflight")
        self.preflight = self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)
        self.bash_patch = patch.object(
            BashTool,
            "execute",
            return_value={
                "ok": True,
                "command": CASE_TEST_COMMAND,
                "cwd": ".",
                "exit_code": 1,
                "stdout": "",
                "stderr": "Offline fixture: 300 != 0\nRan 3 tests\nFAILED\n",
                "truncated": False,
            },
        )
        self.bash = self.bash_patch.start()
        self.addCleanup(self.bash_patch.stop)

    @staticmethod
    def prepare_mocked_bash(
        tool, root, *, output_limit=12000, timeout=10, readonly_paths=()
    ):
        tool.root = Path(root).resolve()
        tool.output_limit = output_limit
        tool.timeout = timeout
        tool.readonly_paths = readonly_paths

    def command(self, *arguments):
        with redirect_stderr(self.stderr), redirect_stdout(self.stdout):
            return Lesson().run(list(arguments))

    def records(self):
        return list(Recorder.read(self.filename))

    @staticmethod
    def hashes(root=CASE_ROOT):
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in CASE_FILES
        }

    def test_case_and_replay_make_no_api_requests(self):
        self.assertEqual(self.command("case", "--output", str(self.filename)), 0)
        self.assertEqual(self.command("replay", str(self.filename)), 0)
        self.client.assert_not_called()
        self.network.assert_not_called()
        records = self.records()
        self.assertEqual(records[-1]["data"], {"expected_failure": True})
        self.assertNotIn("api.request", [record["event"] for record in records])
        self.assertEqual(len(records[0]["data"]["file_hashes"]), 3)
        self.assertFalse(self.workspace.exists())
        self.preflight.assert_called_once()
        self.bash.assert_called_once_with({"command": CASE_TEST_COMMAND})

    def test_run_requires_explicit_live_and_environment_key(self):
        for args in (
            ("run", "--model", "deepseek-chat"),
            ("run", "--live", "--model", "deepseek-chat"),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.command(*args, "--output", str(self.filename)), 2)
        self.client.assert_not_called()
        self.assertFalse(self.filename.exists())
        self.network.assert_not_called()
        self.preflight.assert_not_called()

    def test_sandbox_preflight_failure_stops_before_api_and_keeps_attempt(self):
        original_hashes = self.hashes()
        self.preflight.side_effect = ValueError("Offline fixture: sandbox unavailable")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            status = self.command(
                "run",
                "--live",
                "--model",
                "offline-fixture",
                "--output",
                str(self.filename),
            )
        self.assertEqual(status, 1)
        self.preflight.assert_called_once()
        self.client.assert_not_called()
        self.network.assert_not_called()
        self.bash.assert_not_called()
        self.assertEqual(self.hashes(), original_hashes)
        self.assertEqual(self.hashes(self.workspace), original_hashes)
        self.assertEqual(self.records()[-1]["event"], "run.failed")
        self.assertIn("sandbox unavailable", self.records()[-1]["data"]["message"])
        self.assertNotIn("dummy-private-key", self.filename.read_text())

    def test_case_preflight_failure_has_no_host_shell_fallback(self):
        self.preflight.side_effect = ValueError("Offline fixture: sandbox unavailable")
        self.assertEqual(self.command("case", "--output", str(self.filename)), 1)
        self.preflight.assert_called_once()
        self.bash.assert_not_called()
        self.client.assert_not_called()
        self.network.assert_not_called()
        self.assertFalse(self.workspace.exists())
        self.assertEqual(self.records()[-1]["event"], "run.failed")

    def test_mocked_bash_command_is_forwarded_and_observation_reaches_model(self):
        command = "python3 -m unittest discover -s tests -v"
        self.client.return_value.complete.side_effect = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "offline-bash-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": command, "timeout": 3}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Офлайн-заглушка: наблюдение получено."},
        ]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            status = self.command(
                "run",
                "--live",
                "--model",
                "offline-fixture",
                "--output",
                str(self.filename),
            )
        self.assertEqual(status, 0)
        self.bash.assert_called_once_with({"command": command, "timeout": 3})
        messages = self.client.return_value.complete.call_args.args[0]
        self.assertEqual(messages[-1]["tool_call_id"], "offline-bash-1")
        self.assertEqual(json.loads(messages[-1]["content"])["exit_code"], 1)
        definitions = self.client.return_value.complete.call_args.args[1]
        self.assertEqual(
            {tool["function"]["name"] for tool in definitions},
            {"read_file", "bash", "edit"},
        )
        self.network.assert_not_called()

    def test_invalid_bash_arguments_do_not_execute_shell(self):
        self.client.return_value.complete.side_effect = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "offline-bash-invalid",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "[]"},
                    }
                ],
            },
            {"role": "assistant", "content": "Офлайн-заглушка: ошибка получена."},
        ]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            status = self.command(
                "run",
                "--live",
                "--model",
                "offline-fixture",
                "--output",
                str(self.filename),
            )
        self.assertEqual(status, 0)
        self.bash.assert_not_called()
        results = [
            record["data"]
            for record in self.records()
            if record["event"] == "tool.result"
        ]
        self.assertEqual(results[0]["result"]["error"]["code"], "invalid_call")
        self.network.assert_not_called()

    def test_mocked_live_attempt_failure_is_saved_and_not_retried(self):
        original_hashes = self.hashes()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            self.client.return_value.complete.side_effect = RuntimeError(
                "Client unavailable; dummy-private-key"
            )
            result = self.command(
                "run",
                "--live",
                "--model",
                "deepseek-chat",
                "--output",
                str(self.filename),
            )
        self.assertEqual(result, 1)
        self.assertEqual(self.client.return_value.complete.call_count, 1)
        records = self.records()
        started = next(
            record["data"] for record in records if record["event"] == "run.started"
        )
        self.assertEqual(started["mode"], "run")
        self.assertEqual(started["file_hashes"], original_hashes)
        self.assertEqual(records[-1]["event"], "run.failed")
        self.assertNotIn("dummy-private-key", self.filename.read_text())
        self.assertNotIn("dummy-private-key", self.stderr.getvalue())
        self.assertEqual(self.hashes(), original_hashes)
        self.assertEqual(self.hashes(self.workspace), original_hashes)
        workspace_event = next(
            record["data"] for record in records if record["event"] == "run.workspace"
        )
        self.assertEqual(workspace_event["path"], str(self.workspace))
        self.network.assert_not_called()

    def test_mocked_live_final_answer_and_limits_are_recorded(self):
        original_hashes = self.hashes()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            self.client.return_value.complete.return_value = {
                "role": "assistant",
                "content": "Нужна проверка файлов.",
            }
            result = self.command(
                "run",
                "--live",
                "--model",
                "deepseek-chat",
                "--max-steps",
                "2",
                "--max-tool-calls",
                "3",
                "--max-tokens",
                "100",
                "--timeout",
                "5",
                "--output",
                str(self.filename),
            )
        self.assertEqual(result, 0)
        self.client.assert_called_once()
        self.assertEqual(
            self.client.call_args.args[:2], ("dummy-private-key", "deepseek-chat")
        )
        self.assertEqual(
            self.client.call_args.kwargs, {"timeout": 5, "max_tokens": 100}
        )
        records = self.records()
        settings = next(
            record["data"] for record in records if record["event"] == "run.settings"
        )
        self.assertEqual(settings["max_steps"], 2)
        self.assertEqual(settings["max_tool_calls"], 3)
        self.assertEqual(records[-1]["event"], "run.completed")
        self.assertEqual(records[-1]["data"], {"file_hashes": original_hashes})
        self.assertEqual(self.hashes(), original_hashes)
        self.assertEqual(self.hashes(self.workspace), original_hashes)
        self.assertEqual(
            sorted(
                path.relative_to(self.workspace).as_posix()
                for path in self.workspace.rglob("*")
                if path.is_file()
            ),
            sorted(CASE_FILES),
        )
        self.assertEqual(self.workspace.stat().st_mode & 0o777, 0o700)
        self.assertIn("Нужна проверка файлов.", self.stdout.getvalue())

    def test_client_constructor_failure_is_recorded_and_redacted(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            self.client.side_effect = ValueError("Bad key: dummy-private-key")
            result = self.command(
                "run",
                "--live",
                "--model",
                "deepseek-chat",
                "--output",
                str(self.filename),
            )
        self.assertEqual(result, 1)
        self.assertEqual(self.records()[-1]["event"], "run.failed")
        self.assertNotIn("dummy-private-key", self.filename.read_text())
        self.assertNotIn("dummy-private-key", self.stderr.getvalue())
        self.assertTrue(self.workspace.is_dir())
        self.assertEqual(self.hashes(self.workspace), self.hashes())
        self.network.assert_not_called()

    def test_edited_working_copy_is_kept_when_later_model_request_fails(self):
        original_hashes = self.hashes()
        old_text = 'Path("settings.json")'
        new_text = 'Path(__file__).with_name("settings.json")'
        self.client.return_value.complete.side_effect = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "offline-edit-1",
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": json.dumps(
                                {
                                    "path": "shop/delivery.py",
                                    "old_text": old_text,
                                    "new_text": new_text,
                                }
                            ),
                        },
                    }
                ],
            },
            RuntimeError("Offline fixture: request failed after edit"),
        ]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            result = self.command(
                "run",
                "--live",
                "--model",
                "offline-fixture",
                "--output",
                str(self.filename),
            )
        self.assertEqual(result, 1)
        self.assertEqual(self.client.return_value.complete.call_count, 2)
        edited = (self.workspace / "shop/delivery.py").read_text(encoding="utf-8")
        self.assertIn(new_text, edited)
        self.assertNotIn(old_text, edited)
        self.assertEqual(self.hashes(), original_hashes)
        results = [
            record["data"]
            for record in self.records()
            if record["event"] == "tool.result"
        ]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["result"]["changed"])
        self.assertEqual(
            results[0]["result"]["after_sha256"],
            self.hashes(self.workspace)["shop/delivery.py"],
        )
        self.assertEqual(self.records()[-1]["event"], "run.failed")
        self.network.assert_not_called()

    def test_existing_workspace_is_not_overwritten_or_sent_to_model(self):
        self.workspace.mkdir()
        marker = self.workspace / "preserved.txt"
        marker.write_text("Earlier attempt\n", encoding="utf-8")
        original_hashes = self.hashes()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-private-key"}):
            result = self.command(
                "run",
                "--live",
                "--model",
                "deepseek-chat",
                "--output",
                str(self.filename),
            )
        self.assertNotEqual(result, 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "Earlier attempt\n")
        self.assertEqual(list(self.workspace.iterdir()), [marker])
        self.assertEqual(self.hashes(), original_hashes)
        self.client.assert_not_called()
        self.network.assert_not_called()

    def test_existing_log_is_not_overwritten(self):
        self.filename.write_text("existing\n")
        self.assertEqual(self.command("case", "--output", str(self.filename)), 2)
        self.assertEqual(self.filename.read_text(), "existing\n")

    def test_replay_json_includes_complete_records(self):
        with Recorder(self.filename) as recorder:
            recorder.emit("api.response", body={"usage": {"total_tokens": 12}})
        self.assertEqual(self.command("replay", str(self.filename), "--json"), 0)
        self.assertEqual(
            json.loads(self.stdout.getvalue())["data"]["body"]["usage"]["total_tokens"],
            12,
        )
        self.client.assert_not_called()

    def test_corrupt_or_missing_replay_returns_error(self):
        self.assertEqual(self.command("replay", str(self.filename)), 2)
        self.filename.write_text("not json\n")
        self.assertEqual(self.command("replay", str(self.filename)), 2)

    def test_terminal_controls_in_model_fields_are_removed(self):
        with redirect_stdout(self.stdout):
            Recorder.show(
                {
                    "event": "tool.call",
                    "data": {"name": "\x1b[31mread_file", "arguments": "\x1b[2J"},
                }
            )
            Recorder.show({"event": "agent.final", "data": {"answer": "\x1b[2JОтвет"}})
        self.assertNotIn("\x1b", self.stdout.getvalue())

    def test_cli_limits_rejected_before_request(self):
        limits = (
            ("--max-steps", "0"),
            ("--max-tool-calls", "33"),
            ("--max-tokens", "4097"),
            ("--timeout", "0"),
            ("--max-steps", "not-an-integer"),
        )
        for option, value in limits:
            with self.subTest(option=option), self.assertRaises(SystemExit) as raised:
                self.command(
                    "run",
                    "--live",
                    "--model",
                    "deepseek-chat",
                    option,
                    value,
                    "--output",
                    str(self.filename),
                )
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.filename.exists())
        self.client.assert_not_called()
        self.network.assert_not_called()

    def test_key_with_linebreak_and_empty_model_rejected_before_request(self):
        attempts = (("bad-key\n", "deepseek-chat"), ("valid-dummy-key", " "))
        for key, model in attempts:
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": key}):
                with self.subTest(model=model):
                    self.assertEqual(
                        self.command(
                            "run",
                            "--live",
                            "--model",
                            model,
                            "--output",
                            str(self.filename),
                        ),
                        2,
                    )
        self.assertFalse(self.filename.exists())
        self.client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
