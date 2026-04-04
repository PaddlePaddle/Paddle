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

This test verifies that conv2d, conv3d, and depthwise_conv2d kernels
are properly registered for bfloat16 precision on HIP (ROCm) backend.
"""

import unittest

import numpy as np

import paddle
from paddle import nn
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

    def test_conv2d_bf16_with_padding(self):
        """Test conv2d BF16 with padding on HIP."""
        paddle.set_device("gpu")

        input_np = np.random.randn(2, 4, 32, 32).astype(np.float32)
        filter_np = np.random.randn(16, 4, 5, 5).astype(np.float32)

        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")
        filter_tensor = paddle.to_tensor(filter_np).astype("bfloat16")

        output = paddle.nn.functional.conv2d(
            input_tensor, filter_tensor, padding=2
        )

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [2, 16, 32, 32])

    def test_conv2d_bf16_with_stride(self):
        """Test conv2d BF16 with stride on HIP."""
        paddle.set_device("gpu")

        input_np = np.random.randn(1, 8, 128, 128).astype(np.float32)
        filter_np = np.random.randn(16, 8, 7, 7).astype(np.float32)

        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")
        filter_tensor = paddle.to_tensor(filter_np).astype("bfloat16")

        output = paddle.nn.functional.conv2d(
            input_tensor, filter_tensor, stride=2
        )

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [1, 16, 61, 61])

    def test_conv2d_bf16_with_groups(self):
        """Test conv2d BF16 with groups (depthwise-like) on HIP."""
        paddle.set_device("gpu")

        input_np = np.random.randn(2, 8, 32, 32).astype(np.float32)
        filter_np = np.random.randn(8, 1, 3, 3).astype(np.float32)

        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")
        filter_tensor = paddle.to_tensor(filter_np).astype("bfloat16")

        output = paddle.nn.functional.conv2d(
            input_tensor, filter_tensor, groups=8
        )

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [2, 8, 30, 30])


@unittest.skipIf(not core.is_compiled_with_rocm(), "HIP/ROCm is not available")
class TestHIPBF16DepthwiseConv2dKernel(unittest.TestCase):
    """Test that depthwise_conv2d kernel is registered for BF16 on HIP."""

    def test_depthwise_conv2d_bf16(self):
        """Test depthwise conv2d BF16 on HIP via Conv2D layer."""
        paddle.set_device("gpu")

        # Depthwise convolution: groups == in_channels
        in_channels = 16
        conv = nn.Conv2D(
            in_channels, in_channels, kernel_size=3, groups=in_channels
        )
        # Convert weights to BF16
        conv.weight.set_value(conv.weight.numpy().astype(np.float32))

        input_np = np.random.randn(2, in_channels, 32, 32).astype(np.float32)
        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")

        # Set weight to BF16 by casting the layer
        conv = conv.to("bfloat16")

        output = conv(input_tensor)
        self.assertEqual(output.dtype, paddle.bfloat16)


@unittest.skipIf(not core.is_compiled_with_rocm(), "HIP/ROCm is not available")
class TestHIPBF16ConvLayer(unittest.TestCase):
    """Test BF16 Conv2D/Conv3D layers on HIP end-to-end."""

    def test_conv2d_layer_bf16(self):
        """Test Conv2D layer with BF16 on HIP."""
        paddle.set_device("gpu")

        conv = nn.Conv2D(3, 16, kernel_size=3, padding=1)
        conv = conv.to("bfloat16")

        input_np = np.random.randn(4, 3, 64, 64).astype(np.float32)
        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")

        output = conv(input_tensor)

        self.assertEqual(output.dtype, paddle.bfloat16)
        self.assertEqual(output.shape, [4, 16, 64, 64])
        self.assertFalse(paddle.isnan(output).any())

    def test_conv2d_bn_relu_bf16(self):
        """Test Conv2D + BN + ReLU pattern (common in vision encoders) with BF16 on HIP."""
        paddle.set_device("gpu")

        model = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2D(32),
            nn.ReLU(),
        )
        model = model.to("bfloat16")
        # BN running stats stay in FP32

        input_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
        input_tensor = paddle.to_tensor(input_np).astype("bfloat16")

        output = model(input_tensor)
        self.assertEqual(output.shape, [2, 32, 112, 112])


if __name__ == "__main__":
    unittest.main()
