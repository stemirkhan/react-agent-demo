import re
import unittest

from react_demo.prompts import PromptTemplates
from react_demo.tools import CASE_FILES


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.system = PromptTemplates.SYSTEM
        self.task = PromptTemplates.DELIVERY_TASK

    def test_system_does_not_include_delivery_case(self):
        for detail in (
            *CASE_FILES,
            "2000",
            "2500",
            "3000",
            "доставк",
            "падающ",
        ):
            with self.subTest(detail=detail):
                self.assertNotIn(detail, self.system.lower())

    def test_task_lists_available_files_without_revealing_cause(self):
        paths = re.findall(r"^- (.+)$", self.task, flags=re.MULTILINE)
        self.assertEqual(paths, list(CASE_FILES))
        self.assertIn("Изменять можно только shop/delivery.py", self.task)
        self.assertIn("Настройки и тесты не меняй", self.task)
        self.assertIn("проверь результат тестами", self.task)
        self.assertNotIn("3000", self.task)
        self.assertNotIn("FileNotFoundError", self.task)


if __name__ == "__main__":
    unittest.main()
