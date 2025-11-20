# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.base import core


def is_custom_device():
    custom_dev_types = paddle.device.get_all_custom_device_type()
    if custom_dev_types and paddle.device.is_compiled_with_custom_device(
        custom_dev_types[0]
    ):
        return True
    return False


class TestEventStreamAPIs(unittest.TestCase):
    """Test paddle.device Event and Stream APIs across different hardware types."""

    def setUp(self):
        """Set up test environment."""
        if not (
            core.is_compiled_with_cuda()
            or core.is_compiled_with_xpu()
            or is_custom_device()
        ):
            self.skipTest("CUDA, XPU or Custom Device not available")

        self.cuda_available = core.is_compiled_with_cuda()
        self.xpu_available = core.is_compiled_with_xpu()
        self.custom_device_available = is_custom_device()

        # Get available custom device types
        if self.custom_device_available:
            self.custom_device_types = core.get_all_custom_device_type()
            self.default_custom_device = self.custom_device_types[0]
        else:
            self.custom_device_types = []
            self.default_custom_device = None

        self._original_device = paddle.device.get_device()
        self._original_stream = paddle.device.current_stream()

    def tearDown(self):
        """Clean up after timing functionality test."""
        paddle.device.synchronize()
        paddle.device.set_device(self._original_device)
        try:
            paddle.device.set_stream(self._original_stream)
        except Exception:
            pass

    def test_event_stream_apis_cuda(self):
        """Test Event and Stream APIs with CUDA."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        self._test_event_stream_apis_impl('gpu:0')

    def test_event_stream_apis_customdevice(self):
        """Test Event and Stream APIs with custom device."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        self._test_event_stream_apis_impl(f'{self.default_custom_device}:0')

    def test_event_stream_apis_xpu(self):
        """Test Event and Stream APIs with XPU."""
        if not core.is_compiled_with_xpu():
            self.skipTest("XPU not available")
        self._test_event_stream_apis_impl('xpu:0')

    def _test_event_stream_apis_impl(self, device_str):
        """Test Event and Stream APIs implementation."""
        # Set device
        paddle.device.set_device(device_str)

        # Test Event creation with different parameters
        event1 = paddle.device.Event()
        self.assertIsInstance(event1, paddle.device.Event)

        event2 = paddle.device.Event(device=device_str, enable_timing=True)
        self.assertIsInstance(event2, paddle.device.Event)

        event3 = paddle.device.Event(
            device=device_str, enable_timing=True, blocking=True
        )
        self.assertIsInstance(event3, paddle.device.Event)

        # Test Stream creation with different parameters
        stream1 = paddle.device.Stream()
        self.assertIsInstance(stream1, paddle.device.Stream)

        stream2 = paddle.device.Stream(device=device_str)
        self.assertIsInstance(stream2, paddle.device.Stream)

        stream3 = paddle.device.Stream(device=device_str, priority=1)
        self.assertIsInstance(stream3, paddle.device.Stream)

        # Test current_stream
        current_stream = paddle.device.current_stream()
        self.assertIsInstance(current_stream, paddle.device.Stream)

        # Test set_stream
        prev_stream = paddle.device.set_stream(stream1)
        self.assertIsInstance(prev_stream, paddle.device.Stream)

        prev_stream = paddle.cuda.set_stream(stream1)
        self.assertIsInstance(prev_stream, paddle.cuda.Stream)

        # Test Event.record() with default stream
        event1.record()
        # Query result may be True immediately for some devices
        try:
            self.assertFalse(event1.query())
        except AssertionError:
            pass  # Some devices may complete immediately

        # Test Event.record() with specific stream
        self.assertTrue(event2.query())

        # Test Event.synchronize()
        event1.synchronize()  # Wait for event to complete
        self.assertTrue(event1.query())  # Should be completed now

        # Test Stream.query()
        if not core.is_compiled_with_xpu():
            self.assertTrue(
                stream1.query()
            )  # Should be completed (no work submitted)

        # Test Stream.synchronize()
        stream1.synchronize()  # Should not raise exception

        # Test Stream.wait_event()
        stream2.wait_event(event1)

        # Test Stream.wait_stream()
        stream2.wait_stream(stream1)

        # Test Stream.record_event()
        event4 = stream1.record_event()
        self.assertIsInstance(event4, paddle.device.Event)

        # Test record_event with existing event
        stream1.record_event(event3)

        # Test Event.elapsed_time()
        if hasattr(event1, 'event_base') and hasattr(event2, 'event_base'):
            # Create events with timing enabled
            start_event = paddle.device.Event(
                device=device_str, enable_timing=True
            )
            end_event = paddle.device.Event(
                device=device_str, enable_timing=True
            )

            # Record start event
            start_event.record()

            # Submit some work to the stream
            with paddle.device.stream_guard(stream1):
                # Create a tensor to ensure some work is done
                tensor = paddle.randn([100, 100], dtype='float32')
                result = tensor * 2

            # Record end event
            end_event.record()

            # Synchronize to ensure events are recorded
            end_event.synchronize()

            # Measure elapsed time
            if not core.is_compiled_with_xpu():
                elapsed_time = start_event.elapsed_time(end_event)
                self.assertIsInstance(elapsed_time, (int, float))
                self.assertGreaterEqual(elapsed_time, 0)

        # Test stream_guard context manager
        with paddle.device.stream_guard(stream1):
            # Inside the context, current stream should be stream1
            guarded_stream = paddle.device.current_stream()
            self.assertEqual(guarded_stream.device, stream1.device)

            # Test operations within stream guard
            tensor1 = paddle.ones([10, 10])
            tensor2 = paddle.ones([10, 10])
            result = tensor1 + tensor2

        # After exiting context, stream should be restored
        restored_stream = paddle.device.current_stream()
        self.assertEqual(restored_stream.device, prev_stream.device)

        # Test Stream properties and methods
        self.assertTrue(hasattr(stream1, 'stream_base'))
        self.assertTrue(hasattr(stream1, 'device'))
        if not core.is_compiled_with_xpu():
            self.assertTrue(callable(stream1.query))
        self.assertTrue(callable(stream1.synchronize))
        self.assertTrue(callable(stream1.wait_event))
        self.assertTrue(callable(stream1.wait_stream))
        self.assertTrue(callable(stream1.record_event))

        # Test Event properties and methods
        self.assertTrue(hasattr(event1, 'event_base'))
        self.assertTrue(hasattr(event1, 'device'))
        self.assertTrue(callable(event1.record))
        self.assertTrue(callable(event1.query))
        if not core.is_compiled_with_xpu():
            self.assertTrue(callable(event1.elapsed_time))
        self.assertTrue(callable(event1.synchronize))

        # Test Stream equality and hash
        stream_copy = paddle.device.Stream(device=device_str)
        self.assertNotEqual(stream1, stream_copy)  # Different stream objects
        self.assertEqual(
            hash(stream1), hash(stream1)
        )  # Same hash for same object

        # Test Stream representation
        stream_repr = repr(stream1)
        self.assertIn('paddle.device.Stream', stream_repr)
        self.assertIn(str(stream1.device), stream_repr)

        # Test Event representation
        event_repr = repr(event1)
        self.assertIsNotNone(event_repr)

        # Clean up
        paddle.device.synchronize()

    def test_event_stream_error_handling(self):
        """Test Event and Stream error handling."""
        # Test with invalid device types
        with self.assertRaises(ValueError):
            paddle.device.Event(device='invalid_device:0')

        with self.assertRaises(ValueError):
            paddle.device.Stream(device='invalid_device:0')

        # Test Event.elapsed_time with incompatible events
        if core.is_compiled_with_cuda() or is_custom_device():
            device_str = (
                'gpu:0'
                if core.is_compiled_with_cuda()
                else f'{self.default_custom_device}:0'
            )
            paddle.device.set_device(device_str)

            event1 = paddle.device.Event(device=device_str)
            event2 = paddle.device.Event(device=device_str)

            # Should not raise exception even if events are not recorded
            try:
                elapsed = event1.elapsed_time(event2)
                self.assertIsInstance(elapsed, (int, float))
            except Exception:
                # Some implementations might raise exception, which is also acceptable
                pass


class TestEventStreamTimingFunctionality(unittest.TestCase):
    """Test Event timing functionality with actual work in isolated environment."""

    def setUp(self):
        """Set up test environment for timing functionality."""
        if not (
            core.is_compiled_with_cuda()
            or core.is_compiled_with_xpu()
            or is_custom_device()
        ):
            self.skipTest("CUDA, XPU or Custom Device not available")

        self.cuda_available = core.is_compiled_with_cuda()
        self.custom_device_available = is_custom_device()

        # Get available custom device types
        if self.custom_device_available:
            self.custom_device_types = core.get_all_custom_device_type()
            self.default_custom_device = self.custom_device_types[0]
        else:
            self.custom_device_types = []
            self.default_custom_device = None

        self._original_device = paddle.device.get_device()
        self._original_stream = paddle.device.current_stream()

    def tearDown(self):
        """Clean up after timing functionality test."""
        paddle.device.synchronize()
        paddle.device.set_device(self._original_device)
        try:
            paddle.device.set_stream(self._original_stream)
        except Exception:
            pass

    def test_event_stream_timing_functionality(self):
        """Test Event timing functionality with actual work."""
        if not (self.cuda_available or self.custom_device_available):
            self.skipTest(
                "Timing functionality test requires CUDA or custom device"
            )

        device_str = (
            'gpu:0'
            if self.cuda_available
            else f'{self.default_custom_device}:0'
        )
        paddle.device.set_device(device_str)

        # Create events with timing enabled
        start_event = paddle.device.Event(device=device_str, enable_timing=True)
        end_event = paddle.device.Event(device=device_str, enable_timing=True)

        # Create a stream for work execution
        stream = paddle.device.Stream(device=device_str)

        # Record start event
        start_event.record(stream)

        # Perform some work on the stream
        with paddle.device.stream_guard(stream):
            # Create and perform operations on tensors
            x = paddle.randn([1000, 1000], dtype='float32')
            y = paddle.randn([1000, 1000], dtype='float32')
            # Matrix multiplication - computationally intensive
            z = paddle.matmul(x, y)
            # Ensure the operation is executed
            z_mean = z.mean()

        # Record end event
        end_event.record(stream)

        # Wait for the end event to complete
        end_event.synchronize()
        if not core.is_compiled_with_xpu():
            # Calculate elapsed time
            elapsed_time = start_event.elapsed_time(end_event)

            # Verify the timing result
            self.assertIsInstance(elapsed_time, (int, float))
            self.assertGreater(elapsed_time, 0)  # Should take some time


class TestEventAPIs(unittest.TestCase):
    """Unified test for paddle.Event, paddle.device.Event, and paddle.cuda.Event."""

    def setUp(self):
        if not paddle.device.is_compiled_with_cuda():
            self.skipTest("This test requires CUDA.")
        self.device = "gpu:0"
        paddle.device.set_device(self.device)

        self.event_classes = [
            ("paddle.Event", paddle.Event),
            ("paddle.cuda.Event", paddle.cuda.Event),
        ]

    def test_event_timing_consistency(self):
        """Check timing consistency across different Event APIs."""
        for name, EventCls in self.event_classes:
            with self.subTest(api=name):
                start = EventCls(enable_timing=True)
                end = EventCls(enable_timing=True)

                start.record()

                x = paddle.randn([2048, 2048], dtype="float32")
                y = paddle.randn([2048, 2048], dtype="float32")
                z = paddle.matmul(x, y)
                _ = z.mean()

                end.record()
                end.synchronize()

                elapsed = start.elapsed_time(end)
                self.assertIsInstance(elapsed, (int, float))
                self.assertGreater(
                    elapsed,
                    0.0,
                    f"{name} should measure positive elapsed time.",
                )

    def test_event_methods_available(self):
        """Ensure all Event variants expose expected methods."""
        for name, EventCls in self.event_classes:
            with self.subTest(api=name):
                e = EventCls(enable_timing=True)
                self.assertTrue(hasattr(e, "record"))
                self.assertTrue(hasattr(e, "synchronize"))
                self.assertTrue(hasattr(e, "elapsed_time"))


if __name__ == '__main__':
    unittest.main()
