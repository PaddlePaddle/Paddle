# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from unittest import TestCase

import paddle


def should_skip_tests():
    """
    Check if tests should be skipped based on device availability.
    Skip if neither CUDA, XPU, nor any custom device is available.
    """
    # Check CUDA availability
    cuda_available = paddle.is_compiled_with_cuda()

    # Check XPU availability
    xpu_available = paddle.is_compiled_with_xpu()

    # Check custom device availability
    custom_available = False
    try:
        custom_devices = paddle.device.get_all_custom_device_type()
        if custom_devices:
            for device_type in custom_devices:
                if paddle.device.is_compiled_with_custom_device(device_type):
                    custom_available = True
                    break
    except Exception:
        custom_available = False

    # Skip tests if no supported devices are available
    return not (cuda_available or xpu_available or custom_available)


# Check if we should skip all tests
if should_skip_tests():
    print(
        "Skipping paddle.cuda API tests: No CUDA, XPU, or custom devices available"
    )
    sys.exit(0)


class TestCurrentDevice(TestCase):
    def test_current_device_return_type(self):
        """Test that current_device returns an integer."""
        device_id = paddle.cuda.current_device()
        self.assertIsInstance(
            device_id, int, "current_device should return an integer"
        )

    def test_current_device_non_negative(self):
        """Test that current_device returns a non-negative integer."""
        device_id = paddle.cuda.current_device()
        self.assertGreaterEqual(
            device_id, 0, "current_device should return a non-negative integer"
        )

    def test_current_device_with_device_set(self):
        """Test current_device after setting device."""
        if paddle.device.cuda.device_count() > 0:
            # Test with CUDA device
            original_device = paddle.device.get_device()

            # Set to device 0 if available
            paddle.device.set_device('gpu:0')
            device_id = paddle.cuda.current_device()
            self.assertEqual(
                device_id, 0, "current_device should return 0 when gpu:0 is set"
            )

            # Restore original device
            paddle.device.set_device(original_device)


class TestDeviceCount(TestCase):
    def test_device_count_return_type(self):
        """Test that device_count returns an integer."""
        count = paddle.cuda.device_count()
        self.assertIsInstance(
            count, int, "device_count should return an integer"
        )

    def test_device_count_non_negative(self):
        """Test that device_count returns a non-negative integer."""
        count = paddle.cuda.device_count()
        self.assertGreaterEqual(
            count, 0, "device_count should return a non-negative integer"
        )


class TestEmptyCache(TestCase):
    def test_empty_cache_return_type(self):
        """Test that empty_cache returns None."""
        result = paddle.cuda.empty_cache()
        self.assertIsNone(result, "empty_cache should return None")

    def test_empty_cache_no_exception(self):
        """Test that empty_cache does not raise any exceptions."""
        try:
            paddle.cuda.empty_cache()
        except Exception as e:
            self.fail(f"empty_cache raised an exception: {e}")

    def test_empty_cache_with_memory_allocation(self):
        """Test that empty_cache works after memory allocation."""
        if paddle.cuda.device_count() > 0:
            # Get initial memory state
            initial_memory = paddle.cuda.memory_allocated()

            # Allocate some memory
            tensor = paddle.randn([1000, 1000])
            allocated_memory = paddle.cuda.memory_allocated()

            # Verify that memory was actually allocated
            self.assertGreater(
                allocated_memory,
                initial_memory,
                "Memory should increase after tensor allocation",
            )

            # Delete tensor and empty cache
            del tensor
            paddle.cuda.empty_cache()

            # Check memory after empty_cache
            final_memory = paddle.cuda.memory_allocated()

            # Memory should be reduced after empty_cache
            # Note: We allow some tolerance as memory management may not free everything immediately
            self.assertLessEqual(
                final_memory,
                allocated_memory,
                "Memory should be reduced after empty_cache",
            )


class TestIsInitialized(TestCase):
    def test_is_initialized_return_type(self):
        """Test that is_initialized returns a boolean."""
        result = paddle.cuda.is_initialized()
        self.assertIsInstance(
            result, bool, "is_initialized should return a boolean"
        )

    def test_is_initialized_no_exception(self):
        """Test that is_initialized does not raise any exceptions."""
        try:
            paddle.cuda.is_initialized()
        except Exception as e:
            self.fail(f"is_initialized raised an exception: {e}")

    def test_is_initialized_with_device_availability(self):
        """Test that is_initialized returns True when devices are available."""
        # This test checks if is_initialized correctly detects device compilation
        # The result should be consistent with device availability checks
        initialized = paddle.cuda.is_initialized()

        # If any device is available, is_initialized should return True
        cuda_available = paddle.is_compiled_with_cuda()
        xpu_available = paddle.is_compiled_with_xpu()

        # Check custom devices
        custom_available = False
        try:
            custom_devices = paddle.device.get_all_custom_device_type()
            if custom_devices:
                for device_type in custom_devices:
                    if paddle.device.is_compiled_with_custom_device(
                        device_type
                    ):
                        custom_available = True
                        break
        except Exception:
            custom_available = False

        # is_initialized should return True if any device type is compiled
        expected = cuda_available or xpu_available or custom_available
        self.assertEqual(
            initialized,
            expected,
            f"is_initialized should return {expected} when cuda={cuda_available}, xpu={xpu_available}, custom={custom_available}",
        )


class TestMemoryAllocated(TestCase):
    def test_memory_allocated_return_type(self):
        """Test that memory_allocated returns an integer."""
        result = paddle.cuda.memory_allocated()
        self.assertIsInstance(
            result, int, "memory_allocated should return an integer"
        )

    def test_memory_allocated_non_negative(self):
        """Test that memory_allocated returns a non-negative integer."""
        result = paddle.cuda.memory_allocated()
        self.assertGreaterEqual(
            result, 0, "memory_allocated should return a non-negative integer"
        )

    def test_memory_allocated_consistency(self):
        """Test that memory_allocated returns consistent results when called multiple times."""
        result1 = paddle.cuda.memory_allocated()
        result2 = paddle.cuda.memory_allocated()
        # Memory should be the same or increase (but not decrease without explicit free)
        self.assertGreaterEqual(
            result2, result1 - 1024, "memory_allocated should be consistent"
        )

    def test_memory_allocated_with_device_param(self):
        """Test that memory_allocated works with device parameter."""
        if paddle.cuda.device_count() > 0:
            # Test with device index
            result_index = paddle.cuda.memory_allocated(0)
            self.assertIsInstance(
                result_index,
                int,
                "memory_allocated should return an integer with device index",
            )
            self.assertGreaterEqual(
                result_index,
                0,
                "memory_allocated should return non-negative with device index",
            )

    def test_memory_allocated_no_exception(self):
        """Test that memory_allocated does not raise any exceptions."""
        try:
            paddle.cuda.memory_allocated()
        except Exception as e:
            self.fail(f"memory_allocated raised an exception: {e}")


class TestMemoryReserved(TestCase):
    def test_memory_reserved_return_type(self):
        """Test that memory_reserved returns an integer."""
        result = paddle.cuda.memory_reserved()
        self.assertIsInstance(
            result, int, "memory_reserved should return an integer"
        )

    def test_memory_reserved_non_negative(self):
        """Test that memory_reserved returns a non-negative integer."""
        result = paddle.cuda.memory_reserved()
        self.assertGreaterEqual(
            result, 0, "memory_reserved should return a non-negative integer"
        )

    def test_memory_reserved_consistency(self):
        """Test that memory_reserved returns consistent results when called multiple times."""
        result1 = paddle.cuda.memory_reserved()
        result2 = paddle.cuda.memory_reserved()
        # Reserved memory should be the same or increase (but not decrease without explicit free)
        self.assertGreaterEqual(
            result2, result1 - 1024, "memory_reserved should be consistent"
        )

    def test_memory_reserved_with_device_param(self):
        """Test that memory_reserved works with device parameter."""
        if paddle.cuda.device_count() > 0:
            # Test with device index
            result_index = paddle.cuda.memory_reserved(0)
            self.assertIsInstance(
                result_index,
                int,
                "memory_reserved should return an integer with device index",
            )
            self.assertGreaterEqual(
                result_index,
                0,
                "memory_reserved should return non-negative with device index",
            )

    def test_memory_reserved_no_exception(self):
        """Test that memory_reserved does not raise any exceptions."""
        try:
            paddle.cuda.memory_reserved()
        except Exception as e:
            self.fail(f"memory_reserved raised an exception: {e}")

    def test_memory_reserved_vs_allocated(self):
        """Test that memory_reserved is greater than or equal to memory_allocated."""
        if paddle.cuda.is_initialized():
            reserved = paddle.cuda.memory_reserved()
            allocated = paddle.cuda.memory_allocated()
            self.assertGreaterEqual(
                reserved,
                allocated,
                "memory_reserved should be >= memory_allocated",
            )


class TestSetDevice(TestCase):
    def test_set_device_return_type(self):
        """Test that set_device returns None."""
        if paddle.cuda.device_count() > 0:
            result = paddle.cuda.set_device(0)
            self.assertIsNone(result, "set_device should return None")

    def test_set_device_no_exception(self):
        """Test that set_device does not raise any exceptions."""
        if paddle.cuda.device_count() > 0:
            try:
                paddle.cuda.set_device(0)
            except Exception as e:
                self.fail(f"set_device raised an exception: {e}")

    def test_set_device_with_int_param(self):
        """Test that set_device works with integer parameter."""
        if paddle.cuda.device_count() > 0:
            try:
                # Test with device index 0
                paddle.cuda.set_device(0)
                # Verify device was set correctly
                current_device = paddle.cuda.current_device()
                self.assertEqual(
                    current_device, 0, "set_device should set device to 0"
                )
            except Exception as e:
                self.fail(
                    f"set_device with int parameter raised an exception: {e}"
                )

    def test_set_device_with_str_param(self):
        """Test that set_device works with string parameter."""
        if paddle.is_compiled_with_cuda():
            try:
                # Test with device string
                paddle.cuda.set_device('gpu:0')
                # Verify device was set correctly
                current_device = paddle.cuda.current_device()
                self.assertEqual(
                    current_device,
                    0,
                    "set_device should set device to 0 with 'gpu:0'",
                )
            except Exception as e:
                self.fail(
                    f"set_device with string parameter raised an exception: {e}"
                )

    def test_set_device_with_cuda_place_param(self):
        """Test that set_device works with CUDAPlace parameter."""
        if paddle.is_compiled_with_cuda():
            try:
                # Test with CUDAPlace
                place = paddle.CUDAPlace(0)
                paddle.cuda.set_device(place)
                # Verify device was set correctly
                current_device = paddle.cuda.current_device()
                self.assertEqual(
                    current_device,
                    0,
                    "set_device should set device to 0 with CUDAPlace",
                )
            except Exception as e:
                self.fail(
                    f"set_device with CUDAPlace parameter raised an exception: {e}"
                )

    def test_set_device_with_xpu_place_param(self):
        """Test that set_device works with XPUPlace parameter."""
        if paddle.is_compiled_with_xpu():
            try:
                # Test with XPUPlace
                place = paddle.XPUPlace(0)
                paddle.cuda.set_device(place)
                # Verify device was set correctly
                current_device = paddle.cuda.current_device()
                # For XPU, we check if the device string contains 'xpu:0'
                device_str = paddle.device.get_device()
                self.assertEqual(
                    device_str,
                    'xpu:0',
                    "set_device should set device to xpu:0 with XPUPlace",
                )
            except Exception as e:
                self.fail(
                    f"set_device with XPUPlace parameter raised an exception: {e}"
                )

    def test_set_device_with_xpu_str_param(self):
        """Test that set_device works with XPU string parameter."""
        if paddle.is_compiled_with_xpu():
            try:
                # Test with XPU device string
                paddle.cuda.set_device('xpu:0')
                # Verify device was set correctly
                device_str = paddle.device.get_device()
                self.assertEqual(
                    device_str,
                    'xpu:0',
                    "set_device should set device to xpu:0 with 'xpu:0'",
                )
            except Exception as e:
                self.fail(
                    f"set_device with XPU string parameter raised an exception: {e}"
                )

    def test_set_device_with_custom_place_param(self):
        """Test that set_device works with CustomPlace parameter."""
        custom_devices = paddle.device.get_all_custom_device_type()
        if custom_devices:
            try:
                # Test with CustomPlace
                device_type = custom_devices[0]
                place = paddle.CustomPlace(device_type, 0)
                paddle.cuda.set_device(place)
                # Verify device was set correctly
                device_str = paddle.device.get_device()
                expected_str = f'{device_type}:0'
                self.assertEqual(
                    device_str,
                    expected_str,
                    f"set_device should set device to {expected_str} with CustomPlace",
                )
            except Exception as e:
                self.fail(
                    f"set_device with CustomPlace parameter raised an exception: {e}"
                )

    def test_set_device_with_custom_str_param(self):
        """Test that set_device works with Custom device string parameter."""
        custom_devices = paddle.device.get_all_custom_device_type()
        if custom_devices:
            try:
                # Test with Custom device string
                device_type = custom_devices[0]
                paddle.cuda.set_device(f'{device_type}:0')
                # Verify device was set correctly
                device_str = paddle.device.get_device()
                expected_str = f'{device_type}:0'
                self.assertEqual(
                    device_str,
                    expected_str,
                    f"set_device should set device to {expected_str} with custom device string",
                )
            except Exception as e:
                self.fail(
                    f"set_device with custom device string parameter raised an exception: {e}"
                )

    def test_set_device_invalid_param(self):
        """Test that set_device raises ValueError for invalid parameter types."""
        with self.assertRaises(ValueError) as context:
            paddle.cuda.set_device(3.14)  # Invalid float parameter
        self.assertIn("Unsupported device type", str(context.exception))

        with self.assertRaises(ValueError) as context:
            paddle.cuda.set_device([0])  # Invalid list parameter
        self.assertIn("Unsupported device type", str(context.exception))


if __name__ == '__main__':
    unittest.main()
