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

# test_async_offload_reload_debug.py (XPU version, with debug and performance tests)
# (c) 2023-2025 PaddlePaddle Authors

import time
import unittest

import numpy as np

import paddle
from paddle.incubate.tensor.manipulation import (
    async_offload,
    async_offload_with_offset,
    async_reload,
    create_xpu_async_load,
)


def print_debug_info(tensor, name):
    """Prints the device placement of a tensor."""
    print(f"{name} is on device: {tensor.place}")


class TestSaveLoadLargeParameters(unittest.TestCase):
    def offload_and_reload(self, data0):
        # Print initial device info for the input tensor.
        print_debug_info(data0, "data0 (original)")

        loader = create_xpu_async_load()
        data1 = paddle.randn([10, 10])
        print_debug_info(data1, "data1 (for compute)")

        # Offload data0 -> pinned memory (usually on CPU)
        cpu_data, task = async_offload(data0, loader)
        print_debug_info(cpu_data, "cpu_data (after offload)")

        # Do some random compute on the XPU.
        res = paddle.matmul(data1, data1)
        print_debug_info(res, "res (after first compute)")

        # Wait on the CPU side for the offload task to complete.
        task.cpu_wait()

        # Reload from pinned (CPU) memory -> back to XPU.
        xpu_data, task = async_reload(cpu_data, loader)
        print_debug_info(xpu_data, "xpu_data (after reload)")

        # Another compute on the XPU.
        res = paddle.matmul(data1, data1)
        print_debug_info(res, "res (after second compute)")

        # Wait on both the device (XPU) and CPU sides.
        task.xpu_wait()
        task.cpu_wait()

        # Check correctness.
        np.testing.assert_array_equal(data0.numpy(), cpu_data.numpy())
        np.testing.assert_array_equal(data0.numpy(), xpu_data.numpy())

    def test_large_parameters_paddle_save_tensor(self):
        # Create XPU data.
        data0 = paddle.randn([10, 5])
        print_debug_info(
            data0, "data0 in test_large_parameters_paddle_save_tensor"
        )
        # NOTE: If you need to explicitly move data0 to XPU (depending on your PaddlePaddle version),
        # you might do: data0 = data0.xpu()
        self.offload_and_reload(data0)

    def test_large_parameters_paddle_save_model_weight(self):
        model = paddle.nn.Linear(10, 5)
        data0 = model.weight
        print_debug_info(
            data0,
            "model.weight in test_large_parameters_paddle_save_model_weight",
        )
        self.offload_and_reload(data0)

    def test_offload_with_offset(self):
        loader = create_xpu_async_load()
        data1 = paddle.randn([100])
        print_debug_info(data1, "data1 in test_offload_with_offset")
        data2 = paddle.randn(
            [100]
        ).cpu()  # Ensure data2 is on CPU (pinned memory)
        print_debug_info(data2, "data2 in test_offload_with_offset (CPU)")

        # Partial offload in two segments.
        task1 = async_offload_with_offset(
            src_tensor=data1,
            dst_tensor=data2,
            src_offset=0,
            dst_offset=0,
            offload_size=50,
            async_loader=loader,
        )
        task2 = async_offload_with_offset(
            src_tensor=data1,
            dst_tensor=data2,
            src_offset=50,
            dst_offset=50,
            offload_size=50,
            async_loader=loader,
        )

        # Wait for tasks to complete.
        task1.xpu_wait()
        task2.cpu_wait()

        print_debug_info(data1, "data1 after offload_with_offset")
        print_debug_info(data2, "data2 after offload_with_offset")

        # Check that the data matches.
        np.testing.assert_array_equal(data1.numpy(), data2.numpy())

    def test_xpu_offload_performance(self):
        """
        Performance test: Measure the time cost of offloading 100MB of data.

        All extra print/debug calls have been removed to minimize overhead.
        The offload is run for multiple iterations (with a warm-up) and averaged
        for a more accurate measurement.
        """
        loader = create_xpu_async_load()
        # For float32, each element takes 4 bytes. For 100MB, we need 25M elements.
        num_elements = 25_000_000
        data = paddle.randn([num_elements])
        # Warm-up run to stabilize performance.
        _ = paddle.randn([10, 10])

        iterations = 10
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            cpu_data, task = async_offload(data, loader)
            task.cpu_wait()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / iterations

        # Check data integrity (this is not measured)
        np.testing.assert_array_equal(data.numpy(), cpu_data.numpy())

        # Output the average time (printing outside the measured loop is acceptable)
        print(
            f"Average time taken for offloading 100MB data over {iterations} iterations: {avg_time:.4f} seconds"
        )


if __name__ == '__main__':
    # Removing the default device print to avoid any extra overhead during performance tests.
    # print("Default Paddle device:", paddle.get_device())
    unittest.main()
