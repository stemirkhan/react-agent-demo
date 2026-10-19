import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from react_demo.tools.edit import EditTool


class EditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "shop").mkdir()
        self.path = self.root / "shop/delivery.py"
        self.path.write_bytes(b"value = 300\n")
        self.editor = EditTool(self.root, ("shop/delivery.py",))

    def arguments(self, old_text="300", new_text="0", path="shop/delivery.py"):
        return {"path": path, "old_text": old_text, "new_text": new_text}

    def test_unique_replacement_returns_diff_and_full_hashes(self):
        before = self.path.read_bytes()
        result = self.editor.execute(self.arguments())
        after = self.path.read_bytes()
        self.assertEqual(after, b"value = 0\n")
        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertEqual(result["path"], "shop/delivery.py")
        self.assertIn("--- shop/delivery.py\n", result["diff"])
        self.assertIn("-value = 300\n+value = 0\n", result["diff"])
        self.assertNotIn(str(self.root), result["diff"])
        self.assertEqual(result["before_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result["after_sha256"], hashlib.sha256(after).hexdigest())
        self.assertFalse(result["truncated"])

    def test_ambiguous_including_overlapping_matches_do_not_write(self):
        for text, old_text in ((b"300 300", "300"), (b"aaa", "aa")):
            with self.subTest(text=text):
                self.path.write_bytes(text)
                with self.assertRaises(ValueError):
                    self.editor.execute(self.arguments(old_text=old_text))
                self.assertEqual(self.path.read_bytes(), text)

    def test_missing_empty_and_noop_fragment_do_not_write(self):
        original = self.path.read_bytes()
        for old_text, new_text in (("missing", "0"), ("", "0"), ("300", "300")):
            with self.subTest(old_text=old_text), self.assertRaises(ValueError):
                self.editor.execute(self.arguments(old_text, new_text))
            self.assertEqual(self.path.read_bytes(), original)

    def test_deletion_is_allowed(self):
        self.editor.execute(self.arguments(" = 300", ""))
        self.assertEqual(self.path.read_bytes(), b"value\n")

    def test_invalid_arguments_are_rejected(self):
        for arguments in (
            None,
            [],
            {},
            {"path": "shop/delivery.py", "old_text": "300"},
            {"path": 2, "old_text": "300", "new_text": "0"},
            {"path": "shop/delivery.py", "old_text": False, "new_text": "0"},
            {"path": "shop/delivery.py", "old_text": "300", "new_text": None},
            {**self.arguments(), "extra": True},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.editor.execute(arguments)
        self.assertEqual(self.path.read_bytes(), b"value = 300\n")

    def test_paths_and_uneditable_settings_or_tests_are_rejected(self):
        (self.root / "tests").mkdir()
        (self.root / "shop/settings.json").write_text("300", encoding="utf-8")
        (self.root / "tests/test_delivery.py").write_text("300", encoding="utf-8")
        for relative in (
            "../delivery.py",
            "/etc/passwd",
            "shop/../shop/delivery.py",
            "shop/settings.json",
            "tests/test_delivery.py",
        ):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                self.editor.execute(self.arguments(path=relative))
        self.assertEqual((self.root / "shop/settings.json").read_text(), "300")
        self.assertEqual((self.root / "tests/test_delivery.py").read_text(), "300")

    def test_missing_file_is_not_created(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.editor.execute(self.arguments())
        self.assertFalse(self.path.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symbolic links unavailable")
    def test_symlink_file_and_directory_are_rejected(self):
        actual = self.root / "actual.py"
        actual.write_bytes(b"300")
        self.path.unlink()
        self.path.symlink_to(actual)
        with self.assertRaises(ValueError):
            self.editor.execute(self.arguments())
        self.assertEqual(actual.read_bytes(), b"300")
        self.path.unlink()
        (self.root / "shop").rmdir()
        actual_directory = self.root / "actual"
        actual_directory.mkdir()
        (actual_directory / "delivery.py").write_bytes(b"300")
        (self.root / "shop").symlink_to(actual_directory, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.editor.execute(self.arguments())
        self.assertEqual((actual_directory / "delivery.py").read_bytes(), b"300")

    def test_unicode_and_crlf_are_preserved(self):
        self.path.write_bytes("цена = 300\r\n# доставка\r\n".encode("utf-8"))
        self.editor.execute(self.arguments())
        self.assertEqual(
            self.path.read_bytes(), "цена = 0\r\n# доставка\r\n".encode("utf-8")
        )

    def test_diff_handles_no_final_newline(self):
        self.path.write_bytes(b"300")
        result = self.editor.execute(self.arguments())
        self.assertEqual(self.path.read_bytes(), b"0")
        self.assertIn("-300\n\\ No newline at end of file\n", result["diff"])
        self.assertIn("+0\n\\ No newline at end of file\n", result["diff"])

    def test_limited_diff_keeps_full_hashes(self):
        self.path.write_bytes(b"x" * 200 + b"300\n")
        result = EditTool(self.root, ("shop/delivery.py",), output_limit=100).execute(
            self.arguments()
        )
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["diff"]), 100)
        self.assertEqual(
            result["after_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest()
        )

    def test_failed_atomic_replace_preserves_original_and_cleans_temp_file(self):
        with patch("react_demo.tools.edit.os.replace", side_effect=OSError("failure")):
            with self.assertRaises(OSError):
                self.editor.execute(self.arguments())
        self.assertEqual(self.path.read_bytes(), b"value = 300\n")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_detects_concurrent_change_without_overwriting_it(self):
        def change_file(*args):
            self.path.write_bytes(b"external change\n")

        with patch("react_demo.tools.edit.os.chmod", side_effect=change_file):
            with self.assertRaises(ValueError):
                self.editor.execute(self.arguments())
        self.assertEqual(self.path.read_bytes(), b"external change\n")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    @unittest.skipUnless(os.name == "posix", "Unix permissions required")
    def test_file_permissions_are_preserved(self):
        self.path.chmod(0o640)
        self.editor.execute(self.arguments())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)

    def test_definition_is_independent_and_describes_exact_arguments(self):
        definition = self.editor.definition()
        function = definition["function"]
        self.assertEqual(function["name"], "edit")
        parameters = function["parameters"]
        self.assertEqual(parameters["required"], ["path", "old_text", "new_text"])
        self.assertEqual(parameters["properties"]["path"]["enum"], ["shop/delivery.py"])
        self.assertFalse(parameters["additionalProperties"])
        parameters["properties"]["path"]["enum"].append("shop/settings.json")
        self.assertEqual(
            self.editor.definition()["function"]["parameters"]["properties"]["path"][
                "enum"
            ],
            ["shop/delivery.py"],
        )

    def test_invalid_configuration_is_rejected(self):
        for limit in (False, 99, 100.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                EditTool(self.root, ("shop/delivery.py",), output_limit=limit)
        with self.assertRaises(ValueError):
            EditTool(self.root, ("../delivery.py",))


if __name__ == "__main__":
    unittest.main()
