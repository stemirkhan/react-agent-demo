from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from react_demo.tools.bash import BashTool


class BashProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.platform = patch("react_demo.tools.bash.sys.platform", "linux")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.executable = patch(
            "react_demo.tools.bash.shutil.which", return_value="/usr/bin/bwrap"
        )
        self.executable.start()
        self.addCleanup(self.executable.stop)
        self.bash_exists = patch(
            "react_demo.tools.bash.Path.is_file", return_value=True
        )
        self.bash_exists.start()
        self.addCleanup(self.bash_exists.stop)

    def test_definition_has_generic_command_and_bounded_optional_timeout(self):
        tool = BashTool(self.root, timeout=7)
        parameters = tool.definition()["function"]["parameters"]
        self.assertEqual(tool.name, "bash")
        self.assertEqual(parameters["required"], ["command"])
        self.assertEqual(parameters["properties"]["timeout"]["maximum"], 7)
        parameters["properties"]["command"].clear()
        self.assertEqual(
            tool.definition()["function"]["parameters"]["properties"]["command"][
                "type"
            ],
            "string",
        )

    def test_command_completion_includes_nonzero_exit_as_observation(self):
        tool = BashTool(self.root)
        output = {
            "ok": True,
            "exit_code": 2,
            "stdout": "",
            "stderr": "bad",
            "truncated": False,
        }
        with patch.object(tool, "_run", return_value=output) as run:
            result = tool.execute({"command": "echo bad >&2; exit 2", "timeout": 2})
        self.assertTrue(result["ok"])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["command"], "echo bad >&2; exit 2")
        self.assertEqual(result["cwd"], ".")
        self.assertEqual(run.call_args.args[1], 2)

    def test_arguments_and_timeouts_are_validated_before_execution(self):
        tool = BashTool(self.root)
        for arguments in (
            None,
            [],
            {},
            {"command": 1},
            {"command": " "},
            {"command": "x\0"},
            {"command": "ls", "extra": True},
            {"command": "ls", "timeout": 11},
            *(
                {"command": "ls", "timeout": value}
                for value in (0, -1, True, "1", float("inf"), float("nan"))
            ),
        ):
            with self.subTest(arguments=arguments), patch.object(tool, "_run") as run:
                with self.assertRaises(ValueError):
                    tool.execute(arguments)
                run.assert_not_called()

    def test_configuration_validation(self):
        for value in (99, True, 100.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BashTool(self.root, output_limit=value)
        for value in (0, -1, True, "1", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BashTool(self.root, timeout=value)

    def test_missing_sandbox_and_non_linux_fail_closed(self):
        with (
            patch("react_demo.tools.bash.shutil.which", return_value=None),
            self.assertRaisesRegex(ValueError, "bubblewrap"),
        ):
            BashTool(self.root)
        with (
            patch("react_demo.tools.bash.sys.platform", "darwin"),
            self.assertRaisesRegex(ValueError, "Linux"),
        ):
            BashTool(self.root)
        with (
            patch("react_demo.tools.bash.Path.is_file", return_value=False),
            self.assertRaisesRegex(ValueError, "/usr/bin/bash"),
        ):
            BashTool(self.root)

    def test_readonly_paths_reject_traversal_and_symlinks(self):
        for relative in ("", ".", "../x", "/tmp/x", "shop/../x", "shop//x", 1):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                BashTool(self.root, readonly_paths=(relative,))
        (self.root / "link").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "Символические"):
            BashTool(self.root, readonly_paths=("link",))
        with self.assertRaises(OSError):
            BashTool(self.root, readonly_paths=("missing",))

    def test_preflight_raises_without_host_fallback(self):
        tool = BashTool(self.root)
        with patch.object(
            tool,
            "_run",
            return_value={
                "ok": True,
                "exit_code": 1,
                "stderr": "bwrap: denied",
                "stdout": "",
                "truncated": False,
            },
        ) as run:
            with self.assertRaisesRegex(ValueError, "Bubblewrap"):
                tool.preflight()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-3:], ["/usr/bin/bash", "-c", ":"])

    def test_startup_error_returns_sanitized_tool_error(self):
        tool = BashTool(self.root)
        with patch.object(tool, "_run", side_effect=OSError("host-only-secret")):
            result = tool.execute({"command": "ls"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "sandbox_unavailable")
        self.assertNotIn("host-only-secret", str(result))


@unittest.skipUnless(sys.platform == "linux", "POSIX stream runner")
class BoundedOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch("react_demo.tools.bash.shutil.which", return_value="/usr/bin/bwrap"):
            self.tool = BashTool(Path(self.directory.name), output_limit=100)

    def test_incremental_runner_discards_large_output_on_both_streams(self):
        # Exercise only the pipe reader with trusted test code, never model commands.
        result = self.tool._run(
            [
                sys.executable,
                "-c",
                "import os; os.write(1,b'x'*8388608); os.write(2,b'y'*8388608)",
            ],
            5,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "x" * 100)
        self.assertEqual(result["stderr"], "y" * 100)
        self.assertTrue(result["truncated"])

    def test_timeout_retains_partial_output_and_reaps_process(self):
        result = self.tool._run(
            [
                sys.executable,
                "-c",
                "import time; print('partial',flush=True); time.sleep(3)",
            ],
            0.15,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "timeout")
        self.assertIn("partial", result["stdout"])
        self.assertLess(result["exit_code"], 0)

    def test_invalid_utf8_is_replaced_not_crashed(self):
        result = self.tool._run(
            [sys.executable, "-c", "import os; os.write(1,b'\\xff')"], 1
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "\ufffd")


if __name__ == "__main__":
    unittest.main()
