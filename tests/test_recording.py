import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

from react_demo.recording import (
    Recorder,
    SecretRedactor,
)


class RecordingTests(unittest.TestCase):
    def test_redacts_exact_key_patterns_and_sensitive_fields(self) -> None:
        data = {
            "nested": ["prefix dummy-secret suffix", "sk-or-v1-dummy123"],
            "Authorization": "anything",
            "api_key": "anything",
            "ordinary": 4,
        }
        result = SecretRedactor(("dummy-secret",)).redact(data)
        self.assertEqual(result["nested"], ["prefix [REDACTED] suffix", "[REDACTED]"])
        self.assertEqual(result["Authorization"], "[REDACTED]")
        self.assertEqual(result["api_key"], "[REDACTED]")
        self.assertEqual(result["ordinary"], 4)
        self.assertEqual(data["Authorization"], "anything")

    def test_jsonl_round_trip_permissions_and_no_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "run.jsonl"
            with Recorder(filename, secrets=("dummy-secret",)) as recorder:
                recorder.emit("first", answer="По-русски", secret="dummy-secret")
                recorder.emit("second", value=3)
            records = list(Recorder.read(filename))
            self.assertEqual([r["sequence"] for r in records], [1, 2])
            self.assertEqual(records[0]["data"]["answer"], "По-русски")
            self.assertNotIn("dummy-secret", filename.read_text())
            if os.name == "posix":
                self.assertEqual(filename.stat().st_mode & 0o777, 0o600)
            before = filename.read_bytes()
            with self.assertRaises(FileExistsError):
                Recorder(filename)
            self.assertEqual(filename.read_bytes(), before)

    def test_event_flushed_before_terminal_callback(self) -> None:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "run.jsonl"

            def failed_display(record: dict[str, Any]) -> None:
                self.assertEqual(list(Recorder.read(filename))[0]["event"], "saved")
                raise RuntimeError("Display failed")

            with Recorder(filename, on_event=failed_display) as recorder:
                with self.assertRaises(RuntimeError):
                    recorder.emit("saved", value=1)
            self.assertEqual(len(list(Recorder.read(filename))), 1)

    def test_rejects_corrupted_json_and_sequence(self) -> None:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "run.jsonl"
            for content in (
                "not json\n",
                json.dumps(
                    {"schema_version": 1, "sequence": 2, "event": "test", "data": {}}
                )
                + "\n",
                "[]\n",
            ):
                with self.subTest(content=content):
                    filename.write_text(content)
                    with self.assertRaises(ValueError):
                        list(Recorder.read(filename))

    def test_log_paths_are_unique(self) -> None:
        self.assertNotEqual(Recorder.new_path("runs"), Recorder.new_path("runs"))

    def test_redactor_can_be_injected(self) -> None:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "run.jsonl"
            with Recorder(
                filename, redactor=SecretRedactor(("injected-secret",))
            ) as recorder:
                recorder.emit("saved", value="injected-secret")
            self.assertEqual(
                list(Recorder.read(filename))[0]["data"]["value"], "[REDACTED]"
            )

    def test_echoed_secret_is_redacted_in_mapping_keys_too(self) -> None:
        redactor = SecretRedactor(("dummy-secret",))
        self.assertEqual(
            redactor.redact({"dummy-secret": {"sk-or-v1-dummy123": "value"}}),
            {"[REDACTED]": {"[REDACTED]": "value"}},
        )

    def test_unserializable_event_does_not_consume_sequence(self) -> None:
        with TemporaryDirectory() as temporary:
            filename = Path(temporary) / "run.jsonl"
            with Recorder(filename) as recorder:
                with self.assertRaises(TypeError):
                    recorder.emit("not-saved", value=object())
                recorder.emit("saved", value=1)
            self.assertEqual(list(Recorder.read(filename))[0]["sequence"], 1)


if __name__ == "__main__":
    unittest.main()
