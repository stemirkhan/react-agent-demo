"""Сценарные ответы здесь - заглушки протокола, не запуск настоящей модели."""

from copy import deepcopy
import json
import unittest

from react_demo.agent import Agent


class Payloads:
    @staticmethod
    def call(call_id="call-1", name="read_file", arguments=None):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments if arguments is not None else '{"path":"a.py"}',
            },
        }

    @staticmethod
    def answer(*calls, content=None):
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = list(calls)
        return message


class ScriptedClient:
    """Проверяет обмен сообщениями, а не качество модели."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class ReadFileFixture:
    name = "read_file"

    def __init__(self):
        self.executed = []

    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }

    def execute(self, arguments):
        if set(arguments) != {"path"} or not isinstance(arguments["path"], str):
            raise ValueError("Invalid arguments")
        self.executed.append(deepcopy(arguments))
        if arguments["path"] == "missing.py":
            raise OSError("Private host path must not be returned to model")
        return {"ok": True, "path": arguments["path"], "content": "fixture file"}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.tool = ReadFileFixture()

    def emit(self, event, **data):
        self.events.append((event, deepcopy(data)))

    def run_loop(self, client, **limits):
        return Agent(client, [self.tool], self, **limits).run("task", system="system")

    def test_result_goes_back_with_matching_id_and_tools_on_every_turn(self):
        client = ScriptedClient(
            Payloads.answer(Payloads.call()), Payloads.answer(content="Объяснение")
        )
        self.assertEqual(self.run_loop(client), "Объяснение")
        self.assertEqual(len(client.requests), 2)
        for request in client.requests:
            self.assertEqual(request["tools"], [self.tool.definition()])
        history = client.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in history],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(history[2]["tool_calls"][0]["id"], history[3]["tool_call_id"])
        self.assertEqual(json.loads(history[3]["content"])["content"], "fixture file")
        self.assertEqual(len(client.requests[0]["messages"]), 2)
        self.assertEqual(len(self.events[0][1]["messages"]), 2)

    def test_multiple_calls_get_one_result_each(self):
        client = ScriptedClient(
            Payloads.answer(Payloads.call("a"), Payloads.call("b")),
            Payloads.answer(content="Готово"),
        )
        self.run_loop(client)
        results = client.requests[1]["messages"][3:]
        self.assertEqual([message["tool_call_id"] for message in results], ["a", "b"])
        self.assertEqual(len(self.tool.executed), 2)

    def test_invalid_arguments_unknown_tool_and_file_errors_return_to_model(self):
        client = ScriptedClient(
            Payloads.answer(
                Payloads.call("json", arguments="not JSON"),
                Payloads.call("list", arguments="[]"),
                Payloads.call("unknown", name="shell", arguments="{}"),
                Payloads.call("missing", arguments='{"path":"missing.py"}'),
            ),
            Payloads.answer(content="Ошибки получены"),
        )
        self.run_loop(client)
        results = [
            json.loads(message["content"])
            for message in client.requests[1]["messages"][3:]
        ]
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["ok"] is False for result in results))
        self.assertTrue(all("error" in result for result in results))
        self.assertNotIn("Private host path", json.dumps(results))
        self.assertEqual(self.tool.executed, [{"path": "missing.py"}])

    def test_client_failure_is_saved_and_not_retried(self):
        client = ScriptedClient(
            RuntimeError("API unavailable"), Payloads.answer(content="Не вызывается")
        )
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            self.run_loop(client)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.events[-1][0], "agent.error")

    def test_step_limit_stops_without_new_request(self):
        client = ScriptedClient(
            Payloads.answer(Payloads.call()), Payloads.answer(content="Не вызывается")
        )
        with self.assertRaises(RuntimeError):
            self.run_loop(client, max_steps=1)
        self.assertEqual(len(client.requests), 1)

    def test_call_limit_stops_before_any_over_limit_execution(self):
        client = ScriptedClient(Payloads.answer(Payloads.call("a"), Payloads.call("b")))
        with self.assertRaises(RuntimeError):
            self.run_loop(client, max_tool_calls=1)
        self.assertEqual(self.tool.executed, [])
        self.assertNotIn("tool.call", [event for event, _ in self.events])

    def test_duplicate_id_across_turns_is_rejected(self):
        client = ScriptedClient(
            Payloads.answer(Payloads.call()), Payloads.answer(Payloads.call())
        )
        with self.assertRaises(RuntimeError):
            self.run_loop(client)
        self.assertEqual(len(self.tool.executed), 1)

    def test_empty_answer_is_not_success(self):
        client = ScriptedClient(Payloads.answer(content=" "))
        with self.assertRaises(RuntimeError):
            self.run_loop(client)
        self.assertNotIn("agent.final", [event for event, _ in self.events])

    def test_bad_message_is_saved_before_validation(self):
        client = ScriptedClient({"role": "user", "content": "wrong role"})
        with self.assertRaises(RuntimeError):
            self.run_loop(client)
        self.assertEqual(
            self.events[1],
            (
                "model.response",
                {"step": 1, "message": {"role": "user", "content": "wrong role"}},
            ),
        )

    def test_invalid_message_rejected_before_any_tool_execution(self):
        messages = [
            None,
            {},
            {"role": "assistant", "content": []},
            Payloads.answer(Payloads.call(), Payloads.call()),
            {"role": "assistant", "tool_calls": "wrong"},
            Payloads.answer({"type": "function", "function": {}}),
            Payloads.answer({"type": "other", "id": "a"}),
            Payloads.answer(Payloads.call(arguments={})),
            Payloads.answer(
                Payloads.call("valid"), {"type": "function", "id": "invalid"}
            ),
        ]
        for message in messages:
            with self.subTest(message=message), self.assertRaises(RuntimeError):
                self.run_loop(ScriptedClient(message))
        self.assertEqual(self.tool.executed, [])

    def test_invalid_limits_and_duplicate_tool_names(self):
        for limits in ({"max_steps": 0}, {"max_tool_calls": True}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                Agent(ScriptedClient(), [self.tool], self, **limits)
        with self.assertRaises(ValueError):
            Agent(ScriptedClient(), [self.tool, self.tool], self)

    def test_raw_assistant_fields_preserved_and_snapshots_are_independent(self):
        message = Payloads.answer(Payloads.call())
        message["reasoning_content"] = "opaque protocol field"
        message["tool_calls"][0]["extra"] = {"signature": "opaque"}
        client = ScriptedClient(message, Payloads.answer(content="Готово"))
        self.run_loop(client)
        sent = client.requests[1]["messages"][2]
        self.assertEqual(sent, message)
        sent["tool_calls"][0]["extra"]["signature"] = "mutated"
        self.assertEqual(message["tool_calls"][0]["extra"]["signature"], "opaque")

    def test_second_run_on_same_agent_starts_fresh_history(self):
        client = ScriptedClient(
            Payloads.answer(Payloads.call()),
            Payloads.answer(content="Первый"),
            Payloads.answer(Payloads.call()),
            Payloads.answer(content="Второй"),
        )
        agent = Agent(client, [self.tool], self)
        self.assertEqual(agent.run("task"), "Первый")
        self.assertEqual(agent.run("task"), "Второй")
        self.assertEqual(len(client.requests[2]["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
