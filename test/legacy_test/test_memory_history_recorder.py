#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import platform
import tempfile
import unittest

import paddle

_VMM_FLAG = 'FLAGS_use_virtual_memory_auto_growth'

# The GPU allocator is built lazily on the first allocation, so the VMM V1 stack
# has to be selected before anything touches device memory -- hence module
# scope rather than setUp().
if paddle.is_compiled_with_cuda():
    paddle.set_flags({_VMM_FLAG: 1})


def _skip() -> bool:
    return (
        (not paddle.is_compiled_with_cuda())
        or paddle.is_compiled_with_rocm()
        or platform.system() == "Windows"
    )


@unittest.skipIf(
    _skip(), "memory history recording requires the CUDA VMM V1 allocator"
)
class TestMemoryHistoryRecorder(unittest.TestCase):
    def tearDown(self):
        # Never leave recording on for the next test: it is process-global.
        paddle.device.cuda._record_memory_history(enabled=None)

    def _traces(self):
        snapshot = paddle.device.cuda._snapshot()
        self.assertIn("segments", snapshot)
        self.assertIn("device_traces", snapshot)
        self.assertGreater(len(snapshot["device_traces"]), 0)
        return snapshot["device_traces"][0]

    def test_requires_vmm_allocator_flag(self):
        # Enabling without the VMM V1 allocator must fail loudly rather than
        # arm a recorder that cannot produce allocation events.
        paddle.set_flags({_VMM_FLAG: 0})
        try:
            with self.assertRaises(RuntimeError):
                paddle.device.cuda._record_memory_history(enabled="all")
        finally:
            paddle.set_flags({_VMM_FLAG: 1})

    def test_disable_is_allowed_without_the_flag(self):
        # Turning recording off must never raise, whatever the flag says.
        paddle.set_flags({_VMM_FLAG: 0})
        try:
            paddle.device.cuda._record_memory_history(enabled=None)
        finally:
            paddle.set_flags({_VMM_FLAG: 1})

    def test_alloc_and_free_events_are_recorded(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=100000
        )
        tensor = paddle.zeros(shape=[1024], dtype="float32")
        del tensor

        traces = self._traces()
        actions = [e["action"] for e in traces]
        self.assertIn("alloc", actions)
        self.assertIn("free_completed", actions)

        for event in traces:
            self.assertIn("addr", event)
            self.assertIn("size", event)
            self.assertIn("stream", event)
            self.assertIn("time_us", event)
            self.assertIn("op_name", event)
            self.assertIn("frames", event)

    def test_alloc_and_free_report_the_same_size(self):
        # The alloc hook records the actual block size, so an alloc/free pair
        # for one address must agree (a raw request size would not).
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=100000
        )
        tensor = paddle.zeros(shape=[7], dtype="float32")
        del tensor

        sizes = {}
        for event in self._traces():
            if event["action"] == "alloc":
                sizes[event["addr"]] = event["size"]
            elif event["action"] == "free_completed":
                if event["addr"] in sizes:
                    self.assertEqual(sizes[event["addr"]], event["size"])
        self.assertGreater(len(sizes), 0)

    def test_annotation_marker(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=100000
        )
        # A marker is only appended to rings that already exist.
        tensor = paddle.zeros(shape=[64], dtype="float32")
        paddle.device.cuda._annotate_memory_history("gstep 3 begin")
        del tensor

        markers = [e for e in self._traces() if e["action"] == "annotation"]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["op_name"], "gstep 3 begin")
        self.assertEqual(markers[0]["size"], 0)

    def test_annotation_without_recording_is_a_no_op(self):
        paddle.device.cuda._record_memory_history(enabled=None)
        paddle.device.cuda._annotate_memory_history("ignored")
        self.assertEqual(len(self._traces()), 0)

    def test_disable_clears_recorded_events(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=100000
        )
        tensor = paddle.zeros(shape=[64], dtype="float32")
        del tensor
        self.assertGreater(len(self._traces()), 0)

        # Disabling drops the rings, so a snapshot has to be taken first.
        paddle.device.cuda._record_memory_history(enabled=None)
        self.assertEqual(len(self._traces()), 0)

    def test_max_entries_bounds_the_ring(self):
        max_entries = 8
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=max_entries
        )
        for _ in range(64):
            tensor = paddle.zeros(shape=[32], dtype="float32")
            del tensor
        self.assertLessEqual(len(self._traces()), max_entries)

    def test_python_stack_is_captured(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", stacks="all", max_entries=100000
        )
        tensor = paddle.zeros(shape=[128], dtype="float32")
        del tensor

        allocs = [e for e in self._traces() if e["action"] == "alloc"]
        self.assertGreater(len(allocs), 0)
        # A resolved Python stack has several frames; the op_name fallback
        # produces exactly one.
        deepest = max(len(e["frames"]) for e in allocs)
        self.assertGreater(deepest, 1)
        for event in allocs:
            for frame in event["frames"]:
                self.assertIn("filename", frame)
                self.assertIn("name", frame)
                self.assertIn("line", frame)

    def test_stacks_disabled_falls_back_to_op_name(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", stacks="none", max_entries=100000
        )
        tensor = paddle.zeros(shape=[128], dtype="float32")
        del tensor

        allocs = [e for e in self._traces() if e["action"] == "alloc"]
        self.assertGreater(len(allocs), 0)
        for event in allocs:
            self.assertLessEqual(len(event["frames"]), 1)

    def test_stacks_min_size_skips_small_allocations(self):
        paddle.device.cuda._record_memory_history(
            enabled="all",
            stacks="all",
            max_entries=100000,
            stacks_min_size=1 << 30,
        )
        tensor = paddle.zeros(shape=[128], dtype="float32")
        del tensor

        allocs = [e for e in self._traces() if e["action"] == "alloc"]
        self.assertGreater(len(allocs), 0)
        # Every allocation is far below the threshold, so no stack is attached.
        for event in allocs:
            self.assertLessEqual(len(event["frames"]), 1)

    def test_dump_snapshot_roundtrip(self):
        paddle.device.cuda._record_memory_history(
            enabled="all", max_entries=100000
        )
        tensor = paddle.zeros(shape=[256], dtype="float32")
        del tensor

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "snapshot.pickle")
            paddle.device.cuda._dump_snapshot(path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                loaded = pickle.load(f)

        self.assertIn("segments", loaded)
        self.assertIn("device_traces", loaded)
        actions = [e["action"] for e in loaded["device_traces"][0]]
        self.assertIn("alloc", actions)


if __name__ == "__main__":
    unittest.main()
