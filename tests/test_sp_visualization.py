import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_method(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


class SequenceParallelVisualizationTest(unittest.TestCase):
    def test_inference_kv_cache_resolves_local_attention_heads(self):
        method = find_method(
            ROOT / "pipeline" / "causal_diffusion_inference.py",
            "_initialize_kv_cache",
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_kv_cache_heads"
        ]

        self.assertEqual(len(calls), 1)

    def test_validation_sampler_uses_data_parallel_layout(self):
        method = find_method(
            ROOT / "trainer" / "distillation.py",
            "__init__",
        )
        sampler_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "DistributedSampler"
        ]
        matching_calls = []
        for call in sampler_calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            num_replicas = keywords.get("num_replicas")
            rank = keywords.get("rank")
            if (
                isinstance(num_replicas, ast.Attribute)
                and num_replicas.attr == "data_parallel_size"
                and isinstance(rank, ast.Attribute)
                and rank.attr == "data_parallel_rank"
            ):
                matching_calls.append(call)

        self.assertEqual(len(matching_calls), 1)


if __name__ == "__main__":
    unittest.main()
