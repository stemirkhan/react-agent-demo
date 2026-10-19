from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from react_demo.tools import CASE_FILES, CASE_ROOT, ReadFileTool


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.reader = ReadFileTool(CASE_ROOT, CASE_FILES)

    def test_reads_only_exact_allowed_paths(self):
        for path in CASE_FILES:
            with self.subTest(path=path):
                result = self.reader.execute({"path": path})
                self.assertTrue(result["ok"])
                self.assertEqual(
                    result["content"], (CASE_ROOT / path).read_text(encoding="utf-8")
                )
        for path in (
            "../settings.json",
            "/etc/passwd",
            "shop/../shop/settings.json",
            "react_demo/prompts.py",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.reader.execute({"path": path})

    def test_rejects_invalid_arguments(self):
        for arguments in (
            [],
            None,
            {},
            {"path": 2},
            {"path": "shop/delivery.py", "extra": True},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.reader.execute(arguments)

    def test_symlink_file_and_directory_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shop").mkdir()
            (root / "shop/settings.json").symlink_to(CASE_ROOT / "shop/settings.json")
            with self.assertRaises(ValueError):
                ReadFileTool(root, CASE_FILES).execute({"path": "shop/settings.json"})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shop").symlink_to(CASE_ROOT / "shop", target_is_directory=True)
            with self.assertRaises(ValueError):
                ReadFileTool(root, CASE_FILES).execute({"path": "shop/delivery.py"})

    def test_missing_file_raises_io_error(self):
        with TemporaryDirectory() as temporary, self.assertRaises(OSError):
            ReadFileTool(Path(temporary), CASE_FILES).execute(
                {"path": "shop/delivery.py"}
            )

    def test_read_truncation_and_definitions_are_independent(self):
        self.assertTrue(
            ReadFileTool(CASE_ROOT, CASE_FILES, output_limit=100).execute(
                {"path": "shop/delivery.py"}
            )["truncated"]
        )
        definition = self.reader.definition()
        definition["function"]["parameters"]["properties"]["path"]["enum"].clear()
        self.assertEqual(
            self.reader.definition()["function"]["parameters"]["properties"]["path"][
                "enum"
            ],
            list(CASE_FILES),
        )

    def test_invalid_tool_config_rejected(self):
        for value in (99, 0, True, 100.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReadFileTool(CASE_ROOT, CASE_FILES, output_limit=value)
        for path in (
            "/etc/passwd",
            "../secrets",
            "shop/../settings.json",
            "shop//settings.json",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ReadFileTool(CASE_ROOT, (path,))


if __name__ == "__main__":
    unittest.main()
