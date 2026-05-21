# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Test HIP BF16 convolution kernel registration.

This test keeps coverage focused on minimal convolution forward passes so the
suite validates kernel registration without pulling in unrelated BF16 operator
chains.
"""

import unittest

import numpy as np

import paddle
from paddle.base import core


@unittest.skipIf(not core.is_compiled_with_rocm(), "HIP/ROCm is not available")
class TestHIPBF16Conv2dKernel(unittest.TestCase):
    """Test that conv2d kernel is registered for BF16 on HIP."""

    def test_conv2d_bf16_forward(self):
        """Test conv2d BF16 forward pass on HIP."""
        paddle.set_device("gpu")

        # Create BF16 input tensor
        input_np = np.random.randn(1, 3, 64, 64).astype(np.float32)
        filter_np = np.random.randn(8, 3, 3, 3).astype(np.float32)

        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")
        filter_tensor = paddle.to_tensor(filter_np).astype("bfloat16")

        # This should not raise "kernel not registered" error
        output = paddle.nn.functional.conv2d(input_tensor, filter_tensor)

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [1, 8, 62, 62])
        # Verify output is not NaN
        self.assertFalse(paddle.isnan(output).any())

    def test_conv2d_bf16_with_groups(self):
        """Test conv2d BF16 with groups (depthwise-like) on HIP."""
        paddle.set_device("gpu")

        input_np = np.random.randn(1, 8, 16, 16).astype(np.float32)
        filter_np = np.random.randn(8, 1, 3, 3).astype(np.float32)

        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")
        filter_tensor = paddle.to_tensor(filter_np).astype("bfloat16")

        output = paddle.nn.functional.conv2d(
            input_tensor, filter_tensor, groups=8
        )

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [1, 8, 14, 14])


if __name__ == "__main__":
    unittest.main()
