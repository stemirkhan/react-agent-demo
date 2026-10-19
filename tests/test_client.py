"""DeepSeek HTTP tests use only prepared responses, never a live API."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from react_demo.client import (
    ClientError,
    DeepSeekClient,
    DIAGNOSTIC_LIMIT,
    ENDPOINT,
    NoRedirect,
    RESPONSE_LIMIT,
)
from react_demo.recording import Recorder


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_limit: int | None = None

    def read(self, limit: int) -> bytes:
        self.read_limit = limit
        return self.body[:limit]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.filename = Path(self.temporary.name) / "client.jsonl"
        self.recorder = Recorder(self.filename, secrets=("dummy-deepseek-key",))
        self.addCleanup(self.recorder.close)
        self.opener = Mock()
        opener_patch = patch("react_demo.client.build_opener", return_value=self.opener)
        self.build_opener = opener_patch.start()
        self.addCleanup(opener_patch.stop)
        self.client = DeepSeekClient(
            "dummy-deepseek-key", "deepseek-test-model", self.recorder
        )
        self.messages = [{"role": "user", "content": "Проверь проект"}]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

    def respond(
        self, message: dict[str, Any] | None = None, **choice_fields: Any
    ) -> dict[str, Any]:
        body = {
            "id": "offline-response",
            "usage": {"prompt_tokens": 42, "completion_tokens": 11},
            "choices": [
                {
                    "message": (
                        message
                        if message is not None
                        else {"role": "assistant", "content": "Ответ"}
                    ),
                    "finish_reason": "stop",
                    **choice_fields,
                }
            ],
        }
        self.raw_response(body)
        return body

    def raw_response(self, body: Any) -> FakeResponse:
        response = FakeResponse(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        self.opener.open.return_value = response
        return response

    def records(self) -> list[dict[str, Any]]:
        return list(Recorder.read(self.filename))

    def test_request_payload_and_no_key_in_journal(self) -> None:
        self.respond()
        self.assertEqual(
            self.client.complete(self.messages, self.tools)["content"], "Ответ"
        )
        request = self.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.get_header("Authorization"), "Bearer dummy-deepseek-key"
        )
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 30})
        self.assertEqual(payload["messages"], self.messages)
        self.assertEqual(payload["tools"], self.tools)
        self.assertEqual(payload["model"], "deepseek-test-model")
        self.assertEqual(payload["max_tokens"], 800)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["stream"])
        self.assertNotIn("parallel_tool_calls", payload)
        self.assertNotIn("headers", self.records()[0]["data"])
        self.assertNotIn("dummy-deepseek-key", self.filename.read_text())
        self.opener.open.assert_called_once()

    def test_usage_and_opaque_assistant_fields_are_preserved(self) -> None:
        message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Служебный текст",
            "opaque_field": {"signature": ["unchanged"]},
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"python3 -m unittest discover -s tests -v"}',
                    },
                }
            ],
        }
        body = self.respond(message, finish_reason="tool_calls")
        actual = self.client.complete(self.messages, self.tools)
        self.assertEqual(actual, message)
        self.assertEqual(self.records()[-1]["data"]["body"], body)
        actual["opaque_field"]["signature"].append("changed")
        self.assertEqual(message["opaque_field"]["signature"], ["unchanged"])

    def test_truncation_and_other_finish_reasons_fail_after_raw_response_saved(
        self,
    ) -> None:
        for reason in (
            "length",
            "content_filter",
            "aborted",
            "insufficient_system_resource",
            None,
        ):
            with self.subTest(reason=reason):
                body = self.respond(finish_reason=reason)
                with self.assertRaises(ClientError):
                    self.client.complete(self.messages, self.tools)
                self.assertEqual(self.records()[-2]["event"], "api.response")
                self.assertEqual(self.records()[-2]["data"]["body"], body)
                self.assertEqual(self.records()[-1]["event"], "api.error")
        self.assertEqual(self.opener.open.call_count, 5)

    def test_invalid_response_shapes_fail_with_raw_body_preserved(self) -> None:
        cases = [
            [],
            {},
            {"error": {"message": "bad"}},
            {"choices": []},
            {"choices": [None]},
            {"choices": [{"finish_reason": "stop", "message": None}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "user"}}]},
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": 3},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "tool_calls": {}},
                    }
                ]
            },
        ]
        for body in cases:
            with self.subTest(body=body):
                self.raw_response(body)
                with self.assertRaises(ClientError):
                    self.client.complete(self.messages, self.tools)
                self.assertEqual(self.records()[-2]["data"]["body"], body)

    def test_http_error_is_bounded_redacted_and_not_retried(self) -> None:
        error_body = b"dummy-deepseek-key " + b"x" * DIAGNOSTIC_LIMIT
        self.opener.open.side_effect = HTTPError(
            ENDPOINT, 401, "Unauthorized", {}, BytesIO(error_body)
        )
        with self.assertRaisesRegex(ClientError, "HTTP 401") as raised:
            self.client.complete(self.messages, self.tools)
        self.opener.open.assert_called_once()
        self.assertNotIn("dummy-deepseek-key", str(raised.exception))
        self.assertNotIn("dummy-deepseek-key", self.filename.read_text())
        error = self.records()[-1]["data"]
        self.assertEqual(error["status"], 401)
        self.assertIn("[REDACTED]", error["body"])
        self.assertLessEqual(len(error["body"]), DIAGNOSTIC_LIMIT)

    def test_network_failure_is_sanitized_and_not_retried(self) -> None:
        self.opener.open.side_effect = URLError("dummy-deepseek-key private diagnostic")
        with self.assertRaisesRegex(ClientError, "повтор не выполнялся"):
            self.client.complete(self.messages, self.tools)
        self.opener.open.assert_called_once()
        self.assertNotIn("private diagnostic", self.filename.read_text())
        self.assertNotIn("dummy-deepseek-key", self.filename.read_text())

    def test_response_size_limit_and_read_limit(self) -> None:
        response = FakeResponse(b"x" * (RESPONSE_LIMIT + 2))
        self.opener.open.return_value = response
        with self.assertRaisesRegex(ClientError, "лимит размера"):
            self.client.complete(self.messages, self.tools)
        self.assertEqual(response.read_limit, RESPONSE_LIMIT + 1)
        self.assertNotIn("body", self.records()[-1]["data"])

    def test_invalid_json_diagnostic_is_bounded(self) -> None:
        self.opener.open.return_value = FakeResponse(b"not-json " * DIAGNOSTIC_LIMIT)
        with self.assertRaisesRegex(ClientError, "не JSON"):
            self.client.complete(self.messages, self.tools)
        self.assertLessEqual(len(self.records()[-1]["data"]["body"]), DIAGNOSTIC_LIMIT)

    def test_echoed_key_in_successful_response_is_redacted_in_journal(self) -> None:
        self.respond({"role": "assistant", "content": "dummy-deepseek-key"})
        self.client.complete(self.messages, self.tools)
        self.assertNotIn("dummy-deepseek-key", self.filename.read_text())
        self.assertIn("[REDACTED]", self.filename.read_text())

    def test_non_binary_transport_response_fails(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = "not bytes"
        self.opener.open.return_value = response
        with self.assertRaisesRegex(ClientError, "тип ответа"):
            self.client.complete(self.messages, self.tools)

    def test_redirect_is_refused_without_another_request(self) -> None:
        handler = self.build_opener.call_args.args[0]
        self.assertIsInstance(handler, NoRedirect)
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://other.test")
        )
        self.opener.open.side_effect = HTTPError(
            ENDPOINT, 302, "Found", {}, BytesIO(b"redirect")
        )
        with self.assertRaisesRegex(ClientError, "HTTP 302"):
            self.client.complete(self.messages, self.tools)
        self.opener.open.assert_called_once()

    def test_settings_validation_happens_before_network(self) -> None:
        cases = [
            {"api_key": ""},
            {"api_key": "bad\nkey"},
            {"api_key": "bad\rkey"},
            {"model": " "},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"timeout": 0},
            {"timeout": True},
            {"timeout": float("nan")},
            {"timeout": float("inf")},
        ]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                DeepSeekClient(
                    **{
                        "api_key": "dummy-deepseek-key",
                        "model": "deepseek-test-model",
                        "recorder": self.recorder,
                        **updates,
                    }
                )
        self.opener.open.assert_not_called()

    def test_explicit_model_and_limits_are_public_and_used(self) -> None:
        client = DeepSeekClient(
            "dummy-deepseek-key",
            " chosen-model ",
            self.recorder,
            max_tokens=123,
            timeout=17,
        )
        self.respond()
        client.complete(self.messages, self.tools)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(
            (client.model, client.max_tokens, client.timeout), ("chosen-model", 123, 17)
        )
        self.assertEqual(json.loads(request.data)["model"], "chosen-model")
        self.assertEqual(json.loads(request.data)["max_tokens"], 123)
        self.assertEqual(self.opener.open.call_args.kwargs["timeout"], 17)
