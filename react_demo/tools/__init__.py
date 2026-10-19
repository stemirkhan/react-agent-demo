"""Three tools available in the lesson's working copy."""

from pathlib import Path
from .bash import BashTool
from .edit import EditTool

from .read_file import ReadFileTool

CASE_ROOT = Path(__file__).resolve().parents[2] / "case_project"
CASE_FILES = (
    "shop/delivery.py",
    "shop/settings.json",
    "tests/test_delivery.py",
)
CASE_TEST_COMMAND = "python3 -m unittest discover -s tests -v"

__all__ = [
    "CASE_ROOT",
    "CASE_FILES",
    "CASE_TEST_COMMAND",
    "ReadFileTool",
    "BashTool",
    "EditTool",
]
