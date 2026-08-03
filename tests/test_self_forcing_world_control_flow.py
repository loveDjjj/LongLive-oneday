import ast
import unittest
from pathlib import Path


class SelfForcingWorldControlFlowTest(unittest.TestCase):
    def test_rollout_exit_steps_are_broadcast_over_world(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "pipeline"
            / "self_forcing_training.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "generate_and_sync_list"
        )
        broadcasts = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "broadcast"
        ]

        self.assertEqual(len(broadcasts), 1)
        keywords = {keyword.arg: keyword.value for keyword in broadcasts[0].keywords}
        self.assertNotIn("group", keywords)


if __name__ == "__main__":
    unittest.main()
