import types
import unittest
from unittest import mock

import torch

from utils.device import create_event, create_stream, stream_context, supports_streams


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeAccelerator:
    class Stream:
        def __init__(self, device=None):
            self.device = device

    class Event:
        def __init__(self, enable_timing=False):
            self.enable_timing = enable_timing

    @staticmethod
    def stream(stream):
        return _FakeContext()


class DeviceStreamTest(unittest.TestCase):
    def test_npu_runtime_uses_npu_stream_and_event_apis(self):
        device = types.SimpleNamespace(type="npu")
        with mock.patch.object(torch, "npu", _FakeAccelerator(), create=True):
            self.assertTrue(supports_streams(device))
            stream = create_stream(device)
            event = create_event(device, enable_timing=True)
            self.assertIs(stream.device, device)
            self.assertTrue(event.enable_timing)
            with stream_context(stream, device):
                pass

    def test_cpu_does_not_report_stream_support(self):
        self.assertFalse(supports_streams(torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
