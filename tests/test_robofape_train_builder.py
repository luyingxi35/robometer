import ast
import unittest
from pathlib import Path


class TestRoboFAPETrainBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "mani_envs/data_collection/build_setting_a_twelve_task.py"
        cls.tree = ast.parse(path.read_text())

    def test_canonical_rows_only_publish_target_progress(self):
        string_constants = {
            node.value for node in ast.walk(self.tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("target_progress", string_constants)
        self.assertIn("canonical rows must expose only target_progress", string_constants)

    def test_failure_builder_reads_instant_not_max_progress(self):
        attributes = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        get_keys = {
            node.args[0].value
            for node in attributes
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        self.assertIn("instant_progress", get_keys)
        self.assertNotIn("max_progress", get_keys)


if __name__ == "__main__":
    unittest.main()
