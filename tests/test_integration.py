"""Офлайн-проверка обмена. Заданный маршрут - не настоящий запуск модели урока."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from react_demo.client import DeepSeekClient, ENDPOINT
from react_demo.main import Lesson
from react_demo.recording import Recorder
from react_demo.tools import (
    BashTool,
    CASE_FILES,
    CASE_ROOT,
    CASE_TEST_COMMAND,
    EditTool,
    ReadFileTool,
)


class HttpResponseFixture:
    def __init__(self, message):
        self.body = json.dumps(
            {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": (
                            "tool_calls" if message.get("tool_calls") else "stop"
                        ),
                    }
                ],
                "usage": {"total_tokens": 0},
            }
        ).encode("utf-8")

    def read(self, size):
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class OfflineClientFactory:
    """Настоящий клиент получает только заранее заданные HTTP-ответы."""

    def __init__(self):
        self.requests = []
        self.messages = iter(
            [
                self.assistant(
                    self.call("read-1", "read_file", '{"path":"shop/delivery.py"}')
                ),
                self.assistant(
                    self.call("read-2", "read_file", '{"path":"shop/settings.json"}'),
                    self.call(
                        "tests-1", "bash", json.dumps({"command": CASE_TEST_COMMAND})
                    ),
                ),
                self.assistant(
                    self.call(
                        "edit-1",
                        "edit",
                        json.dumps(
                            {
                                "path": "shop/delivery.py",
                                "old_text": 'Path("settings.json")',
                                "new_text": 'Path(__file__).with_name("settings.json")',
                            }
                        ),
                    )
                ),
                self.assistant(
                    self.call(
                        "tests-2", "bash", json.dumps({"command": CASE_TEST_COMMAND})
                    )
                ),
                {"role": "assistant", "content": "Офлайн-заглушка: обмен завершен."},
            ]
        )

    @staticmethod
    def call(call_id, name, arguments):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    @staticmethod
    def assistant(*calls):
        return {"role": "assistant", "content": None, "tool_calls": list(calls)}

    def __call__(self, api_key, model, recorder, **options):
        client = DeepSeekClient(api_key, model, recorder, **options)
        client._opener.open = Mock(side_effect=self.respond)
        return client

    def respond(self, request, **options):
        if request.full_url != ENDPOINT:
            raise AssertionError("Unexpected endpoint")
        self.requests.append(deepcopy(json.loads(request.data)))
        return HttpResponseFixture(next(self.messages))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        # This test deliberately uses real isolation; protocol-only tests do not.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in CASE_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(CASE_ROOT / name, target)
            try:
                BashTool(
                    root, readonly_paths=("shop/settings.json", "tests")
                ).preflight()
            except ValueError as exc:
                self.skipTest(f"Bubblewrap integration unavailable: {exc}")

    @staticmethod
    def hashes(root):
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in CASE_FILES
        }

    def test_lesson_client_agent_tools_recording_and_replay_without_network(self):
        original_hashes = self.hashes(CASE_ROOT)
        with (
            TemporaryDirectory() as temporary,
            patch.dict(
                "os.environ", {"DEEPSEEK_API_KEY": "dummy-integration-key"}, clear=True
            ),
            patch(
                "urllib.request.OpenerDirector.open",
                side_effect=AssertionError("Network forbidden"),
            ) as network,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            filename = Path(temporary) / "integration.jsonl"
            workspace = filename.with_suffix(".project")
            factory = OfflineClientFactory()
            with patch("react_demo.main.DeepSeekClient", side_effect=factory) as client:
                status = Lesson().run(
                    [
                        "run",
                        "--live",
                        "--model",
                        "offline-fixture",
                        "--output",
                        str(filename),
                    ]
                )
                self.assertEqual(status, 0)
                self.assertEqual(len(factory.requests), 5)
                definitions = {
                    tool["function"]["name"]: tool["function"]
                    for tool in factory.requests[0]["tools"]
                }
                self.assertEqual(set(definitions), {"read_file", "bash", "edit"})
                self.assertEqual(
                    definitions["bash"]["parameters"]["required"], ["command"]
                )
                self.assertEqual(
                    definitions["edit"]["parameters"]["properties"]["path"]["enum"],
                    ["shop/delivery.py"],
                )
                self.assertEqual(
                    [message["role"] for message in factory.requests[1]["messages"]],
                    ["system", "user", "assistant", "tool"],
                )
                records = list(Recorder.read(filename))
                results = [
                    record["data"]
                    for record in records
                    if record["event"] == "tool.result"
                ]
                self.assertEqual(
                    [result["call_id"] for result in results],
                    ["read-1", "read-2", "tests-1", "edit-1", "tests-2"],
                )
                self.assertEqual(results[2]["result"]["exit_code"], 1)
                self.assertEqual(results[2]["result"]["command"], CASE_TEST_COMMAND)
                self.assertIn("300 != 0", results[2]["result"]["stderr"])
                edit = results[3]["result"]
                self.assertTrue(edit["ok"])
                self.assertTrue(edit["changed"])
                self.assertEqual(edit["path"], "shop/delivery.py")
                self.assertIn('Path("settings.json")', edit["diff"])
                self.assertIn('Path(__file__).with_name("settings.json")', edit["diff"])
                self.assertEqual(
                    edit["before_sha256"], original_hashes["shop/delivery.py"]
                )
                self.assertEqual(results[4]["result"]["exit_code"], 0)
                self.assertIn("Ran 3 tests", results[4]["result"]["stderr"])
                self.assertIn("OK", results[4]["result"]["stderr"])
                last_tool_result = factory.requests[-1]["messages"][-1]
                self.assertEqual(last_tool_result["role"], "tool")
                self.assertEqual(last_tool_result["tool_call_id"], "tests-2")
                self.assertEqual(
                    json.loads(last_tool_result["content"])["exit_code"], 0
                )
                self.assertEqual(
                    sum(record["event"] == "api.response" for record in records), 5
                )
                self.assertEqual(records[-1]["event"], "run.completed")
                final_hashes = self.hashes(workspace)
                self.assertEqual(records[-1]["data"], {"file_hashes": final_hashes})
                self.assertEqual(edit["after_sha256"], final_hashes["shop/delivery.py"])
                self.assertNotEqual(
                    final_hashes["shop/delivery.py"],
                    original_hashes["shop/delivery.py"],
                )
                self.assertEqual(
                    final_hashes["shop/settings.json"],
                    original_hashes["shop/settings.json"],
                )
                self.assertEqual(
                    final_hashes["tests/test_delivery.py"],
                    original_hashes["tests/test_delivery.py"],
                )
                self.assertEqual(self.hashes(CASE_ROOT), original_hashes)
                workspace_event = next(
                    record["data"]
                    for record in records
                    if record["event"] == "run.workspace"
                )
                self.assertEqual(workspace_event["path"], str(workspace))
                self.assertNotIn("dummy-integration-key", filename.read_text())
                recorded_bytes = filename.read_bytes()
                with (
                    patch.object(
                        ReadFileTool,
                        "execute",
                        side_effect=AssertionError(
                            "Replay must not read project files"
                        ),
                    ),
                    patch.object(
                        EditTool,
                        "execute",
                        side_effect=AssertionError("Replay must not edit files"),
                    ),
                    patch.object(
                        BashTool,
                        "execute",
                        side_effect=AssertionError("Replay must not execute commands"),
                    ),
                    patch.object(
                        BashTool,
                        "preflight",
                        side_effect=AssertionError("Replay must not start a sandbox"),
                    ),
                ):
                    self.assertEqual(Lesson().run(["replay", str(filename)]), 0)
                self.assertEqual(len(factory.requests), 5)
                self.assertEqual(filename.read_bytes(), recorded_bytes)
                self.assertEqual(self.hashes(workspace), final_hashes)
                client.assert_called_once()
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
