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
#
# (c) 2023-2025 PaddlePaddle Authors

import unittest

import numpy as np

import paddle

if not hasattr(paddle, "XPUPinnedPlace"):
    from paddle.base.core import XPUPinnedPlace as _XPUPinnedPlace

    paddle.XPUPinnedPlace = lambda: _XPUPinnedPlace(0)


def print_debug_info(tensor, name):
    """Prints the device placement of a tensor."""
    # print(f"{name} is on device: {tensor.place}")
    pass


class TestXPUPinnedToCpuCopy(unittest.TestCase):
    def test_copy_from_xpu_pinned_to_cpu(self):
        # Create a sample numpy array.
        arr = np.random.rand(10, 10).astype('float32')

        # Create an XPU pinned memory place using the same interface as GPU pinned memory.
        xpu_pinned_place = paddle.XPUPinnedPlace()

        # Create a tensor in XPU pinned memory.
        tensor_pinned = paddle.to_tensor(arr, place=paddle.XPUPinnedPlace())
        # print_debug_info(tensor_pinned, "tensor_pinned (XPU pinned)")

        # Since tensor.copy_to() is not available, copy the tensor by converting to NumPy and back.
        tensor_cpu = paddle.to_tensor(
            tensor_pinned.numpy(), place=paddle.CPUPlace()
        )
        # print_debug_info(tensor_cpu, "tensor_cpu (after copy to CPU)")

        # Verify that the destination tensor is on CPU.
        self.assertIn("cpu", str(tensor_cpu.place))

        # Check correctness: ensure the data remains unchanged after the copy.
        np.testing.assert_array_equal(tensor_cpu.numpy(), arr)

    def test_copy_from_xpu_to_xpu_pinned(self):
        # Create a sample numpy array.
        arr = np.random.rand(10, 10).astype('float32')

        # Create a tensor on an XPU device.
        tensor_xpu = paddle.to_tensor(arr, place=paddle.XPUPlace(0))
        # print_debug_info(tensor_xpu, "tensor_xpu (XPU)")

        # Copy the tensor from XPU to XPU pinned memory by converting to NumPy and back.
        tensor_xpu_pinned = paddle.to_tensor(
            tensor_xpu.numpy(), place=paddle.XPUPinnedPlace()
        )
        # print_debug_info(tensor_xpu_pinned, "tensor_xpu_pinned (after copy to XPU pinned)")

        # Verify that the destination tensor is on XPU pinned memory.
        self.assertIn("pinned", str(tensor_xpu_pinned.place).lower())

        # Check correctness: ensure the data remains unchanged after the copy.
        np.testing.assert_array_equal(tensor_xpu_pinned.numpy(), arr)


if __name__ == '__main__':
    # print("Default Paddle device:", paddle.get_device())
    unittest.main()
