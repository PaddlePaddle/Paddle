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


class TestDeviceAPIs(unittest.TestCase):
    """Test paddle.device APIs across different hardware types."""

    def setUp(self):
        """Set up test environment."""
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

    def test_device_count_cuda(self):
        """Test device_count with CUDA."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_device_count_xpu(self):
        """Test device_count with XPU."""
        if not core.is_compiled_with_xpu():
            self.skipTest("XPU not available")
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_device_count_customdevice(self):
        """Test device_count with custom device."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

        # Test with specific device type
        count_custom = paddle.device.device_count(self.default_custom_device)
        self.assertIsInstance(count_custom, int)
        self.assertGreaterEqual(count_custom, 0)

    def test_get_device_properties_cuda(self):
        """Test get_device_properties with CUDA."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        # Test with default device
        props = paddle.device.get_device_properties()
        self.assertIsNotNone(props)

        # Test with string input
        props_str = paddle.device.get_device_properties('gpu:0')
        self.assertIsNotNone(props_str)

        props_str = paddle.device.get_device_properties('cuda:0')
        self.assertIsNotNone(props_str)

        # Test with integer input
        props_int = paddle.device.get_device_properties(0)
        self.assertIsNotNone(props_int)

        # Test with CUDAPlace input
        props_int = paddle.device.get_device_properties(paddle.CUDAPlace(0))
        self.assertIsNotNone(props_int)

    def test_get_device_properties_customdevice(self):
        """Test get_device_properties with custom device."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        # Test with default device
        props = paddle.device.get_device_properties()
        self.assertIsNotNone(props)

        # Test with string input (device only)
        props_device = paddle.device.get_device_properties(
            self.default_custom_device
        )
        self.assertIsNotNone(props_device)

        # Test with string input (device:id)
        props_str = paddle.device.get_device_properties(
            f'{self.default_custom_device}:0'
        )
        self.assertIsNotNone(props_str)

        # Test with integer input
        props_int = paddle.device.get_device_properties(0)
        self.assertIsNotNone(props_int)

        # Test with CustomPlace input
        props_custom = paddle.device.get_device_properties(
            paddle.CustomPlace(self.default_custom_device, 0)
        )
        self.assertIsNotNone(props_custom)

    def test_empty_cache_cuda(self):
        """Test empty_cache with CUDA."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        # Should not raise any exception
        paddle.device.empty_cache()

    def test_empty_cache_customdevice(self):
        """Test empty_cache with custom device."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        # Should not raise any exception
        paddle.device.empty_cache()

    def test_memory_apis_cuda(self):
        """Test memory management APIs with CUDA with actual tensor allocation."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        # Set device to GPU
        paddle.device.set_device('gpu')

        # Test max_memory_allocated with different input types
        mem1 = paddle.device.max_memory_allocated()
        self.assertIsInstance(mem1, int)
        self.assertGreaterEqual(mem1, 0)

        mem2 = paddle.device.max_memory_allocated('gpu:0')
        self.assertIsInstance(mem2, int)
        self.assertGreaterEqual(mem2, 0)

        mem3 = paddle.device.max_memory_allocated(0)
        self.assertIsInstance(mem3, int)
        self.assertGreaterEqual(mem3, 0)

        mem7 = paddle.device.max_memory_allocated(paddle.CUDAPlace(0))
        self.assertIsInstance(mem7, int)
        self.assertGreaterEqual(mem7, 0)

        # Test max_memory_reserved with different input types
        mem4 = paddle.device.max_memory_reserved()
        self.assertIsInstance(mem4, int)
        self.assertGreaterEqual(mem4, 0)

        mem8 = paddle.device.max_memory_reserved('gpu:0')
        self.assertIsInstance(mem8, int)
        self.assertGreaterEqual(mem8, 0)

        mem9 = paddle.device.max_memory_reserved(0)
        self.assertIsInstance(mem9, int)
        self.assertGreaterEqual(mem9, 0)

        mem10 = paddle.device.max_memory_reserved(paddle.CUDAPlace(0))
        self.assertIsInstance(mem10, int)
        self.assertGreaterEqual(mem10, 0)

        # Test memory_allocated with different input types
        mem5 = paddle.device.memory_allocated()
        self.assertIsInstance(mem5, int)
        self.assertGreaterEqual(mem5, 0)

        mem11 = paddle.device.memory_allocated('gpu:0')
        self.assertIsInstance(mem11, int)
        self.assertGreaterEqual(mem11, 0)

        mem12 = paddle.device.memory_allocated(0)
        self.assertIsInstance(mem12, int)
        self.assertGreaterEqual(mem12, 0)

        mem13 = paddle.device.memory_allocated(paddle.CUDAPlace(0))
        self.assertIsInstance(mem13, int)
        self.assertGreaterEqual(mem13, 0)

        # Test memory_reserved with different input types
        mem6 = paddle.device.memory_reserved()
        self.assertIsInstance(mem6, int)
        self.assertGreaterEqual(mem6, 0)

        mem14 = paddle.device.memory_reserved('gpu:0')
        self.assertIsInstance(mem14, int)
        self.assertGreaterEqual(mem14, 0)

        mem15 = paddle.device.memory_reserved(0)
        self.assertIsInstance(mem15, int)
        self.assertGreaterEqual(mem15, 0)

        mem16 = paddle.device.memory_reserved(paddle.CUDAPlace(0))
        self.assertIsInstance(mem16, int)
        self.assertGreaterEqual(mem16, 0)

        # Now test actual memory allocation and tracking
        initial_allocated = paddle.device.memory_allocated()
        initial_max_allocated = paddle.device.max_memory_allocated()
        initial_reserved = paddle.device.memory_reserved()
        initial_max_reserved = paddle.device.max_memory_reserved()

        # Allocate first tensor (10MB)
        tensor1 = paddle.randn([256, 256, 256], dtype='float32')  # ~67MB

        # Check memory after first allocation
        allocated_after_first = paddle.device.memory_allocated()
        max_allocated_after_first = paddle.device.max_memory_allocated()
        reserved_after_first = paddle.device.memory_reserved()
        max_reserved_after_first = paddle.device.max_memory_reserved()

        self.assertGreater(allocated_after_first, initial_allocated)
        self.assertGreater(max_allocated_after_first, initial_max_allocated)
        self.assertGreaterEqual(reserved_after_first, initial_reserved)
        self.assertGreaterEqual(max_reserved_after_first, initial_max_reserved)

        # Allocate second tensor (5MB)
        tensor2 = paddle.randn([128, 128, 128], dtype='float32')  # ~8MB

        # Check memory after second allocation
        allocated_after_second = paddle.device.memory_allocated()
        max_allocated_after_second = paddle.device.max_memory_allocated()
        reserved_after_second = paddle.device.memory_reserved()
        max_reserved_after_second = paddle.device.max_memory_reserved()

        # Memory should have increased further
        self.assertGreater(allocated_after_second, allocated_after_first)
        self.assertGreater(
            max_allocated_after_second, max_allocated_after_first
        )
        self.assertGreaterEqual(reserved_after_second, reserved_after_first)
        self.assertGreaterEqual(
            max_reserved_after_second, max_reserved_after_first
        )

        # Release first tensor
        del tensor1

        # Check memory after releasing first tensor
        allocated_after_release = paddle.device.memory_allocated()
        max_allocated_after_release = paddle.device.max_memory_allocated()
        reserved_after_release = paddle.device.memory_reserved()
        max_reserved_after_release = paddle.device.max_memory_reserved()

        # Current allocated should decrease, but max should stay the same
        self.assertLess(allocated_after_release, allocated_after_second)
        self.assertEqual(
            max_allocated_after_release, max_allocated_after_second
        )
        self.assertLessEqual(reserved_after_release, reserved_after_second)
        self.assertEqual(max_reserved_after_release, max_reserved_after_second)

        # Test reset functions
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_reserved()
        paddle.device.synchronize()

        # Check memory after reset
        allocated_after_reset = paddle.device.memory_allocated()
        max_allocated_after_reset = paddle.device.max_memory_allocated()
        reserved_after_reset = paddle.device.memory_reserved()
        max_reserved_after_reset = paddle.device.max_memory_reserved()

        # Current allocated should remain the same, but max should be reset to current level
        self.assertEqual(allocated_after_reset, allocated_after_release)
        self.assertLessEqual(
            max_allocated_after_reset, max_allocated_after_release
        )
        self.assertEqual(reserved_after_reset, reserved_after_release)
        self.assertLessEqual(
            max_reserved_after_reset, max_reserved_after_release
        )

        # Clean up
        del tensor2
        paddle.device.empty_cache()

    def test_memory_apis_customdevice(self):
        """Test memory management APIs with custom device with actual tensor allocation."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        # Set device to custom device
        paddle.device.set_device(self.default_custom_device)

        # Test max_memory_allocated with different input types
        mem1 = paddle.device.max_memory_allocated()
        self.assertIsInstance(mem1, int)
        self.assertGreaterEqual(mem1, 0)

        mem2 = paddle.device.max_memory_allocated(self.default_custom_device)
        self.assertIsInstance(mem2, int)
        self.assertGreaterEqual(mem2, 0)

        mem3 = paddle.device.max_memory_allocated(
            f'{self.default_custom_device}:0'
        )
        self.assertIsInstance(mem3, int)
        self.assertGreaterEqual(mem3, 0)

        mem4 = paddle.device.max_memory_allocated(0)
        self.assertIsInstance(mem4, int)
        self.assertGreaterEqual(mem4, 0)

        # Test with CustomPlace
        custom_place = core.CustomPlace(self.default_custom_device, 0)
        mem5 = paddle.device.max_memory_allocated(custom_place)
        self.assertIsInstance(mem5, int)
        self.assertGreaterEqual(mem5, 0)

        # Test max_memory_reserved with different input types
        mem6 = paddle.device.max_memory_reserved()
        self.assertIsInstance(mem6, int)
        self.assertGreaterEqual(mem6, 0)

        mem7 = paddle.device.max_memory_reserved(self.default_custom_device)
        self.assertIsInstance(mem7, int)
        self.assertGreaterEqual(mem7, 0)

        mem8 = paddle.device.max_memory_reserved(
            f'{self.default_custom_device}:0'
        )
        self.assertIsInstance(mem8, int)
        self.assertGreaterEqual(mem8, 0)

        mem9 = paddle.device.max_memory_reserved(0)
        self.assertIsInstance(mem9, int)
        self.assertGreaterEqual(mem9, 0)

        # Test with CustomPlace
        custom_place = core.CustomPlace(self.default_custom_device, 0)
        mem10 = paddle.device.max_memory_reserved(custom_place)
        self.assertIsInstance(mem10, int)
        self.assertGreaterEqual(mem10, 0)

        # Test memory_allocated with different input types
        mem11 = paddle.device.memory_allocated()
        self.assertIsInstance(mem11, int)
        self.assertGreaterEqual(mem11, 0)

        mem12 = paddle.device.memory_allocated(self.default_custom_device)
        self.assertIsInstance(mem12, int)
        self.assertGreaterEqual(mem12, 0)

        mem13 = paddle.device.memory_allocated(
            f'{self.default_custom_device}:0'
        )
        self.assertIsInstance(mem13, int)
        self.assertGreaterEqual(mem13, 0)

        mem14 = paddle.device.memory_allocated(0)
        self.assertIsInstance(mem14, int)
        self.assertGreaterEqual(mem14, 0)

        # Test with CustomPlace
        custom_place = core.CustomPlace(self.default_custom_device, 0)
        mem15 = paddle.device.memory_allocated(custom_place)
        self.assertIsInstance(mem15, int)
        self.assertGreaterEqual(mem15, 0)

        # Test memory_reserved with different input types
        mem16 = paddle.device.memory_reserved()
        self.assertIsInstance(mem16, int)
        self.assertGreaterEqual(mem16, 0)

        mem17 = paddle.device.memory_reserved(self.default_custom_device)
        self.assertIsInstance(mem17, int)
        self.assertGreaterEqual(mem17, 0)

        mem18 = paddle.device.memory_reserved(f'{self.default_custom_device}:0')
        self.assertIsInstance(mem18, int)
        self.assertGreaterEqual(mem18, 0)

        mem19 = paddle.device.memory_reserved(0)
        self.assertIsInstance(mem19, int)
        self.assertGreaterEqual(mem19, 0)

        # Test with CustomPlace
        custom_place = core.CustomPlace(self.default_custom_device, 0)
        mem20 = paddle.device.memory_reserved(custom_place)
        self.assertIsInstance(mem20, int)
        self.assertGreaterEqual(mem20, 0)

        # Now test actual memory allocation and tracking
        initial_allocated = paddle.device.memory_allocated()
        initial_max_allocated = paddle.device.max_memory_allocated()
        initial_reserved = paddle.device.memory_reserved()
        initial_max_reserved = paddle.device.max_memory_reserved()

        # Allocate first tensor
        tensor1 = paddle.randn([128, 128, 128], dtype='float32')  # ~8MB

        # Check memory after first allocation
        allocated_after_first = paddle.device.memory_allocated()
        max_allocated_after_first = paddle.device.max_memory_allocated()
        reserved_after_first = paddle.device.memory_reserved()
        max_reserved_after_first = paddle.device.max_memory_reserved()

        # Memory should have increased
        self.assertGreater(allocated_after_first, initial_allocated)
        self.assertGreater(max_allocated_after_first, initial_max_allocated)
        self.assertGreaterEqual(reserved_after_first, initial_reserved)
        self.assertGreaterEqual(max_reserved_after_first, initial_max_reserved)

        # Allocate second tensor
        tensor2 = paddle.randn([64, 64, 64], dtype='float32')  # ~2MB

        # Check memory after second allocation
        allocated_after_second = paddle.device.memory_allocated()
        max_allocated_after_second = paddle.device.max_memory_allocated()
        reserved_after_second = paddle.device.memory_reserved()
        max_reserved_after_second = paddle.device.max_memory_reserved()

        # Memory should have increased further
        self.assertGreater(allocated_after_second, allocated_after_first)
        self.assertGreater(
            max_allocated_after_second, max_allocated_after_first
        )
        self.assertGreaterEqual(reserved_after_second, reserved_after_first)
        self.assertGreaterEqual(
            max_reserved_after_second, max_reserved_after_first
        )

        # Release first tensor
        del tensor1

        # Check memory after releasing first tensor
        allocated_after_release = paddle.device.memory_allocated()
        max_allocated_after_release = paddle.device.max_memory_allocated()
        reserved_after_release = paddle.device.memory_reserved()
        max_reserved_after_release = paddle.device.max_memory_reserved()

        # Current allocated should decrease, but max should stay the same
        self.assertLess(allocated_after_release, allocated_after_second)
        self.assertEqual(
            max_allocated_after_release, max_allocated_after_second
        )
        self.assertLessEqual(reserved_after_release, reserved_after_second)
        self.assertEqual(max_reserved_after_release, max_reserved_after_second)

        # Test reset functions
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_reserved()

        # Check memory after reset
        allocated_after_reset = paddle.device.memory_allocated()
        max_allocated_after_reset = paddle.device.max_memory_allocated()
        reserved_after_reset = paddle.device.memory_reserved()
        max_reserved_after_reset = paddle.device.max_memory_reserved()

        # Current allocated should remain the same, but max should be reset to current level
        self.assertEqual(allocated_after_reset, allocated_after_release)
        self.assertLessEqual(
            max_allocated_after_reset, max_allocated_after_release
        )
        self.assertEqual(reserved_after_reset, reserved_after_release)
        self.assertLessEqual(
            max_reserved_after_reset, max_reserved_after_release
        )

        # Clean up
        del tensor2
        paddle.device.empty_cache()

    def test_reset_memory_apis_cuda(self):
        """Test reset memory APIs with CUDA with actual tensor allocation."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        # Set device to GPU
        paddle.device.set_device('gpu')

        # Get initial memory values
        initial_max_allocated = paddle.device.max_memory_allocated()
        initial_max_reserved = paddle.device.max_memory_reserved()

        # Allocate tensor to increase memory usage
        tensor = paddle.randn([256, 256, 256], dtype='float32')  # ~67MB

        # Check that max memory has increased
        max_allocated_after_alloc = paddle.device.max_memory_allocated()
        max_reserved_after_alloc = paddle.device.max_memory_reserved()
        self.assertGreater(max_allocated_after_alloc, initial_max_allocated)
        self.assertGreaterEqual(max_reserved_after_alloc, initial_max_reserved)

        # Test reset functions with different input types
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_allocated('gpu:0')
        paddle.device.reset_max_memory_allocated(0)
        paddle.device.reset_max_memory_allocated(paddle.CUDAPlace(0))

        paddle.device.reset_max_memory_reserved()
        paddle.device.reset_max_memory_reserved('gpu:0')
        paddle.device.reset_max_memory_reserved(0)
        paddle.device.reset_max_memory_reserved(paddle.CUDAPlace(0))

        # Check that max memory has been reset
        max_allocated_after_reset = paddle.device.max_memory_allocated()
        max_reserved_after_reset = paddle.device.max_memory_reserved()

        # Max memory should be reset to current level (which should be lower than after allocation)
        self.assertLessEqual(
            max_allocated_after_reset, max_allocated_after_alloc
        )
        self.assertLessEqual(max_reserved_after_reset, max_reserved_after_alloc)

        # Clean up
        del tensor
        paddle.device.empty_cache()

    def test_reset_memory_apis_customdevice(self):
        """Test reset memory APIs with custom device with actual tensor allocation."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        # Set device to custom device
        paddle.device.set_device(self.default_custom_device)

        # Get initial memory values
        initial_max_allocated = paddle.device.max_memory_allocated()
        initial_max_reserved = paddle.device.max_memory_reserved()

        # Allocate tensor to increase memory usage
        tensor = paddle.randn([128, 128, 128], dtype='float32')  # ~8MB

        # Check that max memory has increased
        max_allocated_after_alloc = paddle.device.max_memory_allocated()
        max_reserved_after_alloc = paddle.device.max_memory_reserved()
        self.assertGreater(max_allocated_after_alloc, initial_max_allocated)
        self.assertGreaterEqual(max_reserved_after_alloc, initial_max_reserved)

        # Test reset functions with different input types
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_allocated(self.default_custom_device)
        paddle.device.reset_max_memory_allocated(
            f'{self.default_custom_device}:0'
        )
        paddle.device.reset_max_memory_allocated(0)

        custom_place = core.CustomPlace(self.default_custom_device, 0)
        paddle.device.reset_max_memory_allocated(custom_place)

        paddle.device.reset_max_memory_reserved()
        paddle.device.reset_max_memory_reserved(self.default_custom_device)
        paddle.device.reset_max_memory_reserved(
            f'{self.default_custom_device}:0'
        )
        paddle.device.reset_max_memory_reserved(0)

        custom_place = core.CustomPlace(self.default_custom_device, 0)
        paddle.device.reset_max_memory_reserved(custom_place)

        # Check that max memory has been reset
        max_allocated_after_reset = paddle.device.max_memory_allocated()
        max_reserved_after_reset = paddle.device.max_memory_reserved()

        # Max memory should be reset to current level (which should be lower than after allocation)
        self.assertLessEqual(
            max_allocated_after_reset, max_allocated_after_alloc
        )
        self.assertLessEqual(max_reserved_after_reset, max_reserved_after_alloc)

        # Clean up
        del tensor
        paddle.device.empty_cache()

    def test_stream_apis_cuda(self):
        """Test stream APIs with CUDA."""
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        # Test current_stream with different input types
        stream1 = paddle.device.current_stream()
        self.assertIsNotNone(stream1)

        stream2 = paddle.device.current_stream(paddle.CUDAPlace(0))
        self.assertIsNotNone(stream2)

        # stream3 = paddle.device.current_stream(0)
        # self.assertIsNotNone(stream3)

        # Test synchronize
        paddle.device.synchronize()
        paddle.device.synchronize(paddle.CUDAPlace(0))
        # paddle.device.synchronize(0)

    def test_stream_apis_customdevice(self):
        """Test stream APIs with custom device."""
        if not is_custom_device():
            self.skipTest("Custom device not available")
        # Test current_stream with different input types
        stream1 = paddle.device.current_stream()
        self.assertIsNotNone(stream1)

        stream2 = paddle.device.current_stream(self.default_custom_device)
        self.assertIsNotNone(stream2)

        stream3 = paddle.device.current_stream(
            f'{self.default_custom_device}:0'
        )
        self.assertIsNotNone(stream3)

        # stream4 = paddle.device.current_stream(0)
        # self.assertIsNotNone(stream4)

        # Test synchronize
        paddle.device.synchronize()
        paddle.device.synchronize(self.default_custom_device)
        paddle.device.synchronize(f'{self.default_custom_device}:0')
        # paddle.device.synchronize(0)

    def test_stream_apis_xpu(self):
        """Test stream APIs with XPU."""
        if not core.is_compiled_with_xpu():
            self.skipTest("XPU not available")
        # Test current_stream with different input types
        stream1 = paddle.device.current_stream()
        self.assertIsNotNone(stream1)

        stream2 = paddle.device.current_stream(core.XPUPlace(0))
        self.assertIsNotNone(stream2)

        # stream3 = paddle.device.current_stream(0)
        # self.assertIsNotNone(stream3)

        # Test synchronize
        paddle.device.synchronize()
        paddle.device.synchronize('xpu:0')
        # paddle.device.synchronize(0)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        if not (
            core.is_compiled_with_xpu()
            or core.is_compiled_with_cuda()
            or is_custom_device()
        ):
            self.skipTest("CUDA, XPU and Custom device not available")
        # Test invalid device ID format
        with self.assertRaises(ValueError):
            paddle.device.max_memory_allocated('gpu:invalid')

        # Test invalid input type
        with self.assertRaises(ValueError):
            paddle.device.max_memory_allocated([1, 2, 3])


if __name__ == '__main__':
    unittest.main()
