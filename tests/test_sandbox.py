import hashlib
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from react_demo.tools import CASE_FILES, CASE_ROOT
from react_demo.tools.bash import BashTool


def copy_case(root):
    for relative in CASE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CASE_ROOT / relative, target)


class SandboxCommandTests(unittest.TestCase):
    def test_isolation_uses_namespaces_clean_environment_and_readonly_fixtures(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            copy_case(root)
            with (
                patch(
                    "react_demo.tools.bash.shutil.which", return_value="/usr/bin/bwrap"
                ),
                patch("react_demo.tools.bash.sys.platform", "linux"),
                patch("react_demo.tools.bash.Path.is_file", return_value=True),
            ):
                tool = BashTool(root, readonly_paths=("shop/settings.json", "tests"))
                command = tool._sandbox_command(
                    "python3 -m unittest discover -s tests -v"
                )
        for flag in (
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ):
            self.assertIn(flag, command)
        self.assertIn(["--cap-drop", "ALL"], self._pieces(command, 2))
        self.assertIn(["--bind", str(root), "/project"], self._pieces(command, 3))
        self.assertIn(
            [
                "--ro-bind",
                str(root / "shop/settings.json"),
                "/project/shop/settings.json",
            ],
            self._pieces(command, 3),
        )
        self.assertIn(
            ["--ro-bind", str(root / "tests"), "/project/tests"],
            self._pieces(command, 3),
        )
        self.assertIn(["--chdir", "/project"], self._pieces(command, 2))
        self.assertNotIn(str(CASE_ROOT), command)
        self.assertNotIn(str(Path.home()), command)
        self.assertNotIn("DEEPSEEK_API_KEY", command)
        self.assertNotIn("HOME", command)
        self.assertEqual(
            command[-3:],
            ["/usr/bin/bash", "-c", "python3 -m unittest discover -s tests -v"],
        )

    @staticmethod
    def _pieces(command, length):
        return [
            command[index : index + length]
            for index in range(len(command) - length + 1)
        ]


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("bwrap"), "Нужен Linux с Bubblewrap"
)
class SandboxExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            copy_case(root)
            try:
                BashTool(
                    root, readonly_paths=("shop/settings.json", "tests")
                ).preflight()
            except (OSError, ValueError, RuntimeError) as exc:
                raise unittest.SkipTest(str(exc)) from exc

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "project"
        copy_case(self.root)
        self.tool = BashTool(self.root, readonly_paths=("shop/settings.json", "tests"))
        self.original_hashes = self.hashes(CASE_ROOT)
        self.addCleanup(
            lambda: self.assertEqual(self.hashes(CASE_ROOT), self.original_hashes)
        )

    def command(self, command, **arguments):
        return self.tool.execute({"command": command, **arguments})

    def test_original_failure_and_known_fix_only_in_disposable_copy(self):
        result = self.command("python3 -m unittest discover -s tests -v")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("300 != 0", result["stderr"])
        self.assertNotIn("/project", result["stderr"])
        self.assertNotIn(str(self.root), result["stderr"])
        source = self.root / "shop/delivery.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                'Path("settings.json")', 'Path(__file__).with_name("settings.json")'
            ),
            encoding="utf-8",
        )
        result = self.command("python3 -m unittest discover -s tests -v")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("OK", result["stderr"])
        self.assertEqual(self.hashes(CASE_ROOT), self.original_hashes)

    def test_generic_shell_pipeline_redirects_and_persistent_file_changes(self):
        result = self.command(
            "printf 'alpha\\nbeta\\n' | grep beta > note.txt && cat note.txt"
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "beta\n")
        self.assertEqual((self.root / "note.txt").read_text(), "beta\n")
        result = self.command("printf gamma >> note.txt; cat note.txt")
        self.assertEqual(result["stdout"], "beta\ngamma")

    def test_settings_tests_and_system_mounts_are_readonly(self):
        before = self.hashes(self.root)
        for command in (
            "printf changed > shop/settings.json",
            "printf changed > tests/test_delivery.py",
            "printf changed > tests/new_test.py",
            "rm shop/settings.json",
            "rm -rf tests",
            "printf changed > /usr/react-lesson-test",
        ):
            with self.subTest(command=command):
                result = self.command(command)
                self.assertTrue(result["ok"], result)
                self.assertNotEqual(result["exit_code"], 0, result)
        self.assertEqual(self.hashes(self.root), before)
        self.assertFalse((self.root / "tests/new_test.py").exists())

    def test_escaping_workspace_symlink_does_not_expose_host_file(self):
        sentinel = Path(self.directory.name) / "host-only.txt"
        sentinel.write_text("host secret", encoding="utf-8")
        (self.root / "outside").symlink_to(sentinel)
        result = self.command("cat outside")
        self.assertTrue(result["ok"], result)
        self.assertNotEqual(result["exit_code"], 0)
        result = self.command("printf changed > outside")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertEqual(sentinel.read_text(), "host secret")

    def test_host_files_keys_and_network_are_not_available(self):
        sentinel = Path(self.directory.name) / "host-only.txt"
        sentinel.write_text("host secret", encoding="utf-8")
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            script = f"""
import os
from pathlib import Path
import socket
assert not Path({str(sentinel)!r}).exists()
assert not Path({str(Path.home())!r}).exists()
assert "DEEPSEEK_API_KEY" not in os.environ
assert "HOME" not in os.environ
try:
    socket.create_connection(("127.0.0.1", {port}), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError("host network available")
print("isolated")
"""
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "host-sentinel-secret"}):
                result = self.command("python3 -c " + shlex.quote(script))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["exit_code"], 0, result["stderr"])
        self.assertEqual(result["stdout"], "isolated\n")
        self.assertEqual(sentinel.read_text(), "host secret")

    def test_timeout_kills_background_children_and_retains_partial_output(self):
        start = time.monotonic()
        result = self.command(
            "echo partial; (sleep 0.6; echo survived > child-marker) & sleep 3",
            timeout=0.15,
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["error"]["code"], "timeout")
        self.assertIn("partial", result["stdout"])
        self.assertLess(time.monotonic() - start, 1.5)
        time.sleep(0.7)
        self.assertFalse((self.root / "child-marker").exists())

    def test_detached_child_does_not_survive_completed_command(self):
        result = self.command(
            "setsid /usr/bin/bash -c 'sleep 0.6; echo survived > detached-marker' >/dev/null 2>&1 < /dev/null &"
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["exit_code"], 0)
        time.sleep(0.7)
        self.assertFalse((self.root / "detached-marker").exists())

    def test_large_output_is_truncated_without_deadlock(self):
        tool = BashTool(
            self.root, output_limit=100, readonly_paths=("shop/settings.json", "tests")
        )
        result = tool.execute(
            {
                "command": "python3 -c "
                + shlex.quote(
                    "import os; os.write(1,b'x'*8388608); os.write(2,b'y'*8388608)"
                )
            }
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stdout"], "x" * 100)
        self.assertEqual(result["stderr"], "y" * 100)
        self.assertTrue(result["truncated"])

    @staticmethod
    def hashes(root):
        return {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in CASE_FILES
        }


if __name__ == "__main__":
    unittest.main()
