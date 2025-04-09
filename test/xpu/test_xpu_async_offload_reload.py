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

import unittest

import numpy as np

import paddle

# Disable static mode so that .numpy() can be called.
paddle.disable_static()

from paddle.incubate.tensor.manipulation import (
    async_offload,
    async_offload_with_offset,
    async_reload,
    create_xpu_async_load,
)


def print_debug_info(tensor, name):
    """Prints debug information for a tensor."""
    # print(f"{name} is on device: {tensor.place}")
    # print(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}")
    try:
        arr = tensor.numpy()
        flat = arr.flatten()
        # print(f"{name} first 5 elements: {flat[:5]}")
        # print(f"{name} full array:\n{arr}")
    except Exception as e:
        # print(f"{name} cannot be converted to numpy array: {e}")
        raise


class TestSaveLoadLargeParameters(unittest.TestCase):
    def offload_and_reload(self, data0):
        print_debug_info(data0, "data0 (original)")

        # Create a fixed compute tensor for matmul
        data1 = paddle.arange(0, 100, dtype="float32").reshape([10, 10])
        print_debug_info(data1, "data1 (for compute)")

        loader = create_xpu_async_load()

        # Offload data0 -> pinned memory
        cpu_data, task = async_offload(data0, loader)
        print_debug_info(cpu_data, "cpu_data (after offload)")

        # Do a compute on XPU
        res = paddle.matmul(data1, data1)
        print_debug_info(res, "res (after first compute)")

        # Wait for the offload task to complete (CPU side).
        task.cpu_wait()

        # Reload from pinned memory back to XPU.
        xpu_data, task = async_reload(cpu_data, loader)
        print_debug_info(xpu_data, "xpu_data (after reload)")

        # Do another compute on XPU
        res = paddle.matmul(data1, data1)
        print_debug_info(res, "res (after second compute)")

        # Wait on both the device (XPU) and CPU sides.
        task.xpu_wait()
        task.cpu_wait()

        # Extract numpy arrays and print max differences.
        a = data0.numpy()
        b = cpu_data.numpy()
        c = xpu_data.numpy()
        # print("Max diff (data0 - cpu_data):", np.max(np.abs(a - b)))
        # print("Max diff (data0 - xpu_data):", np.max(np.abs(a - c)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)

    def test_large_parameters_paddle_save_tensor(self):
        # Create a fixed tensor with known values using linspace.
        arr = np.linspace(0, 1, 50).reshape([10, 5]).astype("float32")
        data0 = paddle.to_tensor(arr, place=paddle.XPUPlace(0))
        print_debug_info(
            data0, "data0 in test_large_parameters_paddle_save_tensor"
        )
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
        # Create a fixed source tensor with all elements equal to 3.14.
        # Since paddle.full() does not accept a place argument in dynamic mode,
        # we first create a NumPy array and then convert it to a tensor on XPU.
        src_arr = np.full([100], 3.14, dtype="float32")
        data1 = paddle.to_tensor(src_arr, place=paddle.XPUPlace(0))
        print_debug_info(data1, "data1 in test_offload_with_offset")
        # Create a destination tensor on CPU (pinned memory) initialized to zeros.
        dst_arr = np.zeros([100], dtype="float32")
        data2 = paddle.to_tensor(dst_arr, place=paddle.XPUPinnedPlace())
        print_debug_info(data2, "data2 in test_offload_with_offset (CPU)")

        # Offload in two segments.
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

        # Wait for both tasks.
        task1.xpu_wait()
        task2.cpu_wait()

        print_debug_info(data1, "data1 after offload_with_offset")
        print_debug_info(data2, "data2 after offload_with_offset")
        diff = np.max(np.abs(data1.numpy() - data2.numpy()))
        # print("Max diff (data1 - data2):", diff)
        np.testing.assert_array_equal(data1.numpy(), data2.numpy())


if __name__ == '__main__':
    # print("Default Paddle device:", paddle.get_device())
    unittest.main()
