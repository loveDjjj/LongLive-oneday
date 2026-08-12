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

    def test_dedicated_vae_path_is_a_chunked_background_pipeline(self):
        method = find_method(
            ROOT / "pipeline" / "causal_diffusion_inference.py",
            "_inference_inner",
        )
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

        has_queue = any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "queue"
            and call.func.attr == "Queue"
            for call in calls
        )
        has_thread = any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "threading"
            and call.func.attr == "Thread"
            for call in calls
        )
        queued_names = {
            argument.id
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "vae_work_queue"
            and call.func.attr == "put"
            and call.args
            and isinstance((argument := call.args[0]), ast.Name)
        }
        waits_for_worker = any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in {"vae_all_done", "vae_bg_thread"}
            and call.func.attr in {"wait", "join"}
            for call in calls
        )

        self.assertTrue(has_queue)
        self.assertTrue(has_thread)
        self.assertIn("latent_on_vae", queued_names)
        self.assertTrue(waits_for_worker)

    def test_pipeline_exports_overlap_telemetry(self):
        method = find_method(
            ROOT / "pipeline" / "causal_diffusion_inference.py",
            "_inference_inner",
        )
        assignments = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "last_inference_metrics"
                for target in node.targets
            )
        ]
        exported_keys = {
            key.value
            for assignment in assignments
            if isinstance(assignment.value, ast.Dict)
            for key in assignment.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertTrue(
            {
                "ar_loop_seconds",
                "vae_decode_seconds",
                "vae_enqueue_seconds",
                "vae_drain_seconds",
                "vae_overlap_seconds",
                "vae_chunks",
                "vae_queue_peak",
            }
            <= exported_keys
        )


if __name__ == "__main__":
    unittest.main()
