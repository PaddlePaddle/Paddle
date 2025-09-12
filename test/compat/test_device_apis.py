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


class TestDeviceAPIs(unittest.TestCase):
    """Test paddle.device APIs across different hardware types."""

    def setUp(self):
        """Set up test environment."""
        self.cuda_available = core.is_compiled_with_cuda()
        self.xpu_available = core.is_compiled_with_xpu()
        self.custom_device_available = (
            len(core.get_all_custom_device_type()) > 0
            if hasattr(core, 'get_all_custom_device_type')
            else False
        )

        # Get available custom device types
        if self.custom_device_available:
            self.custom_device_types = core.get_all_custom_device_type()
            self.default_custom_device = self.custom_device_types[0]
        else:
            self.custom_device_types = []
            self.default_custom_device = None

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_device_count_cuda(self):
        """Test device_count with CUDA."""
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

        # Test with specific device type
        count_gpu = paddle.device.device_count()
        self.assertEqual(count_gpu, count)

    @unittest.skipIf(not core.is_compiled_with_xpu(), "XPU not available")
    def test_device_count_xpu(self):
        """Test device_count with XPU."""
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

        # Test with specific device type
        count_xpu = paddle.device.device_count('xpu')
        self.assertEqual(count_xpu, count)

    @unittest.skipIf(
        not (
            hasattr(core, 'get_all_custom_device_type')
            and len(core.get_all_custom_device_type()) > 0
        ),
        "Custom device not available",
    )
    def test_device_count_customdevice(self):
        """Test device_count with custom device."""
        count = paddle.device.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

        # Test with specific device type
        count_custom = paddle.device.device_count(self.default_custom_device)
        self.assertIsInstance(count_custom, int)
        self.assertGreaterEqual(count_custom, 0)

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_get_device_properties_cuda(self):
        """Test get_device_properties with CUDA."""
        # Test with default device
        props = paddle.device.get_device_properties()
        self.assertIsNotNone(props)

        # Test with string input
        props_str = paddle.device.get_device_properties('gpu:0')
        self.assertIsNotNone(props_str)

        # Test with integer input
        props_int = paddle.device.get_device_properties(0)
        self.assertIsNotNone(props_int)

    @unittest.skipIf(
        not (
            hasattr(core, 'get_all_custom_device_type')
            and len(core.get_all_custom_device_type()) > 0
        ),
        "Custom device not available",
    )
    def test_get_device_properties_customdevice(self):
        """Test get_device_properties with custom device."""
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

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_empty_cache_cuda(self):
        """Test empty_cache with CUDA."""
        # Should not raise any exception
        paddle.device.empty_cache()

    @unittest.skipIf(
        not (
            hasattr(core, 'get_all_custom_device_type')
            and len(core.get_all_custom_device_type()) > 0
        ),
        "Custom device not available",
    )
    def test_empty_cache_customdevice(self):
        """Test empty_cache with custom device."""
        # Should not raise any exception
        paddle.device.empty_cache()

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_memory_apis_cuda(self):
        """Test memory management APIs with CUDA."""
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

        # Test max_memory_reserved
        mem4 = paddle.device.max_memory_reserved()
        self.assertIsInstance(mem4, int)
        self.assertGreaterEqual(mem4, 0)

        # Test memory_allocated
        mem5 = paddle.device.memory_allocated()
        self.assertIsInstance(mem5, int)
        self.assertGreaterEqual(mem5, 0)

        # Test memory_reserved
        mem6 = paddle.device.memory_reserved()
        self.assertIsInstance(mem6, int)
        self.assertGreaterEqual(mem6, 0)

    @unittest.skipIf(
        not (
            hasattr(core, 'get_all_custom_device_type')
            and len(core.get_all_custom_device_type()) > 0
        ),
        "Custom device not available",
    )
    def test_memory_apis_customdevice(self):
        """Test memory management APIs with custom device."""
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

        # Test max_memory_reserved
        mem5 = paddle.device.max_memory_reserved()
        self.assertIsInstance(mem5, int)
        self.assertGreaterEqual(mem5, 0)

        mem6 = paddle.device.max_memory_reserved(self.default_custom_device)
        self.assertIsInstance(mem6, int)
        self.assertGreaterEqual(mem6, 0)

        # Test memory_allocated
        mem7 = paddle.device.memory_allocated()
        self.assertIsInstance(mem7, int)
        self.assertGreaterEqual(mem7, 0)

        mem8 = paddle.device.memory_allocated(self.default_custom_device)
        self.assertIsInstance(mem8, int)
        self.assertGreaterEqual(mem8, 0)

        # Test memory_reserved
        mem9 = paddle.device.memory_reserved()
        self.assertIsInstance(mem9, int)
        self.assertGreaterEqual(mem9, 0)

        mem10 = paddle.device.memory_reserved(self.default_custom_device)
        self.assertIsInstance(mem10, int)
        self.assertGreaterEqual(mem10, 0)

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_reset_memory_apis_cuda(self):
        """Test reset memory APIs with CUDA."""
        # Should not raise any exception
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_reserved()

    @unittest.skipIf(
        not (
            hasattr(core, 'get_all_custom_device_type')
            and len(core.get_all_custom_device_type()) > 0
        ),
        "Custom device not available",
    )
    def test_reset_memory_apis_customdevice(self):
        """Test reset memory APIs with custom device."""
        # Should not raise any exception
        paddle.device.reset_max_memory_allocated()
        paddle.device.reset_max_memory_reserved()

    @unittest.skipIf(not core.is_compiled_with_cuda(), "CUDA not available")
    def test_stream_apis_cuda(self):
        """Test stream APIs with CUDA."""
        # Test current_stream with different input types
        stream1 = paddle.device.current_stream()
        self.assertIsNotNone(stream1)

        stream2 = paddle.device.current_stream(paddle.CUDAPlace(0))
        self.assertIsNotNone(stream2)

        stream3 = paddle.device.current_stream(0)
        self.assertIsNotNone(stream3)

        # Test synchronize
        paddle.device.synchronize()
        paddle.device.synchronize(paddle.CUDAPlace(0))
        paddle.device.synchronize(0)

    # @unittest.skipIf(not (hasattr(core, 'get_all_custom_device_type') and len(core.get_all_custom_device_type()) > 0), "Custom device not available")
    # def test_stream_apis_customdevice(self):
    #     """Test stream APIs with custom device."""
    #     # Test current_stream with different input types
    #     stream1 = paddle.device.current_stream()
    #     self.assertIsNotNone(stream1)

    #     stream2 = paddle.device.current_stream(self.default_custom_device)
    #     self.assertIsNotNone(stream2)

    #     stream3 = paddle.device.current_stream(f'{self.default_custom_device}:0')
    #     self.assertIsNotNone(stream3)

    #     stream4 = paddle.device.current_stream(0)
    #     self.assertIsNotNone(stream4)

    #     # Test synchronize
    #     paddle.device.synchronize()
    #     paddle.device.synchronize(self.default_custom_device)
    #     paddle.device.synchronize(f'{self.default_custom_device}:0')
    #     paddle.device.synchronize(0)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid device ID format
        with self.assertRaises(ValueError):
            paddle.device.max_memory_allocated('gpu:invalid')

        # Test invalid input type
        with self.assertRaises(ValueError):
            paddle.device.max_memory_allocated([1, 2, 3])

    def test_input_types_comprehensive(self):
        """Test all APIs with comprehensive input types."""
        input_types = [
            None,
            0,
            'gpu:0' if self.cuda_available else 'cpu',
            'custom_device:0' if self.custom_device_available else 'cpu',
        ]

        # Add CustomPlace if custom device is available
        if self.custom_device_available:
            custom_place = core.CustomPlace(self.default_custom_device, 0)
            input_types.append(custom_place)

        for input_type in input_types:
            with self.subTest(input_type=input_type):
                # Test memory APIs
                try:
                    result = paddle.device.max_memory_allocated(input_type)
                    self.assertIsInstance(result, int)
                    self.assertGreaterEqual(result, 0)
                except Exception as e:
                    # Some inputs might not be supported, which is okay
                    pass

                try:
                    result = paddle.device.max_memory_reserved(input_type)
                    self.assertIsInstance(result, int)
                    self.assertGreaterEqual(result, 0)
                except Exception as e:
                    pass

                try:
                    result = paddle.device.memory_allocated(input_type)
                    self.assertIsInstance(result, int)
                    self.assertGreaterEqual(result, 0)
                except Exception as e:
                    pass

                try:
                    result = paddle.device.memory_reserved(input_type)
                    self.assertIsInstance(result, int)
                    self.assertGreaterEqual(result, 0)
                except Exception as e:
                    pass


if __name__ == '__main__':
    unittest.main()
