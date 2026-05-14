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

# [AUTO-GENERATED] Tests for phi/kernels/cpu/box_coder_kernel.cc
# box_coder_kernel.cc: CPU box coder kernel (encode/decode center-size box format)
# Used in object detection for converting between box representations.
# Encode mode: target_box is 2D [N, 4]
# Decode mode: target_box is 3D [N, M, 4]

import unittest

import numpy as np

import paddle


class TestBoxCoderKernel(unittest.TestCase):
    """Test suite for paddle.vision.ops.box_coder CPU kernel.

    测试 paddle.vision.ops.box_coder 的 CPU 内核，涵盖编码、解码、不同方差模式等场景。
    BoxCoder 在目标检测中用于在锚框 (prior_box) 与目标框 (target_box) 之间转换。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_encode_center_size_basic(self):
        """Test encode_center_size with basic inputs (2D target_box).

        测试基本的 encode_center_size 编码操作（2D target_box）。
        prior_box: [xmin, ymin, xmax, ymax]
        encode: (cx_diff/width, cy_diff/height, log(w_ratio), log(h_ratio)) / variance
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        target_box = paddle.to_tensor([[0.25, 0.25, 0.75, 0.75]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box,
            code_type='encode_center_size',
        )
        # prior: center=(0.5,0.5), w=1, h=1
        # target: center=(0.5,0.5), w=0.5, h=0.5
        # encode: (0/1, 0/1, log(0.5/1), log(0.5/1)) = (0, 0, -0.6931, -0.6931)
        np.testing.assert_allclose(result.numpy()[0, 0, :2], 0.0, atol=1e-4)
        self.assertAlmostEqual(result.numpy()[0, 0, 2], np.log(0.5), places=3)
        self.assertAlmostEqual(result.numpy()[0, 0, 3], np.log(0.5), places=3)

    def test_decode_center_size_identity(self):
        """Test decode_center_size with zero deltas (should return prior_box).

        测试零偏移量的 decode_center_size（应返回 prior_box）。
        """
        prior_box = paddle.to_tensor(
            [[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5]]
        )
        zero_deltas = paddle.zeros([1, 2, 4], dtype='float32')
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=zero_deltas,
            code_type='decode_center_size',
        )
        # With zero deltas, decoded boxes should match prior boxes
        np.testing.assert_allclose(
            result.numpy()[0, 0], prior_box.numpy()[0], atol=1e-4
        )
        np.testing.assert_allclose(
            result.numpy()[0, 1], prior_box.numpy()[1], atol=1e-4
        )

    def test_decode_center_size_with_tensor_var(self):
        """Test decode_center_size with prior_box_var as tensor.

        测试 prior_box_var 为张量的 decode_center_size。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        prior_var = paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]])
        zero_deltas = paddle.zeros([1, 1, 4], dtype='float32')
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=prior_var,
            target_box=zero_deltas,
            code_type='decode_center_size',
        )
        np.testing.assert_allclose(
            result.numpy()[0, 0], [0.0, 0.0, 1.0, 1.0], atol=1e-4
        )

    def test_encode_decode_roundtrip(self):
        """Test that encoding then decoding approximately recovers original box.

        测试编码后解码可以近似恢复原始框。
        Encode uses 2D target_box, decode uses 3D.
        """
        prior_box = paddle.to_tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.5, 0.5]]
        )
        target_box_encode = paddle.to_tensor(
            [[0.25, 0.25, 0.75, 0.75], [0.2, 0.2, 0.4, 0.4]]
        )

        # Encode (2D target_box for encode mode)
        encoded = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box_encode,
            code_type='encode_center_size',
        )
        # encoded shape: [2, 2, 4]

        # Decode (3D target_box for decode mode)
        decoded = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=encoded,
            code_type='decode_center_size',
        )

        # Decoded should approximately match original target_box
        np.testing.assert_allclose(
            decoded.numpy()[0, 0], target_box_encode.numpy()[0], atol=1e-3
        )
        np.testing.assert_allclose(
            decoded.numpy()[1, 1], target_box_encode.numpy()[1], atol=1e-3
        )

    def test_encode_with_variance(self):
        """Test encode_center_size with variance scaling (2D target_box).

        测试带方差缩放的 encode_center_size（2D target_box）。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        target_box = paddle.to_tensor([[0.25, 0.25, 0.75, 0.75]])
        variance = [0.1, 0.1, 0.2, 0.2]
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=variance,
            target_box=target_box,
            code_type='encode_center_size',
        )
        # With variance, encode values are divided by variance
        np.testing.assert_allclose(result.numpy()[0, 0, :2], 0.0, atol=1e-4)

    def test_decode_axis1(self):
        """Test decode_center_size with axis=1.

        测试 axis=1 的 decode_center_size。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        zero_deltas = paddle.zeros([1, 1, 4], dtype='float32')
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=zero_deltas,
            code_type='decode_center_size',
            axis=1,
        )
        np.testing.assert_allclose(
            result.numpy()[0, 0], [0.0, 0.0, 1.0, 1.0], atol=1e-4
        )

    def test_box_normalized_true(self):
        """Test box_coder with box_normalized=True (default).

        测试 box_normalized=True（默认值）的框编解码。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        zero_deltas = paddle.zeros([1, 1, 4], dtype='float32')
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=zero_deltas,
            code_type='decode_center_size',
            box_normalized=True,
        )
        np.testing.assert_allclose(
            result.numpy()[0, 0], [0.0, 0.0, 1.0, 1.0], atol=1e-4
        )

    def test_box_normalized_false(self):
        """Test box_coder with box_normalized=False.

        测试 box_normalized=False（非归一化坐标）的框编解码。
        When not normalized, width/height computation adds 1.
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 99.0, 99.0]])
        zero_deltas = paddle.zeros([1, 1, 4], dtype='float32')
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=zero_deltas,
            code_type='decode_center_size',
            box_normalized=False,
        )
        # With box_normalized=False: width = xmax - xmin + 1
        # prior_box width = 99 - 0 + 1 = 100
        # center_x = 0 + 100/2 = 50, center_y = 50
        # decoded: xmin = 50 - 100/2 = 0, ymin = 0
        #          xmax = 50 + 100/2 - 1 = 99, ymax = 99
        np.testing.assert_allclose(
            result.numpy()[0, 0], [0.0, 0.0, 99.0, 99.0], atol=1e-4
        )

    def test_encode_multiple_priors(self):
        """Test encoding with multiple prior boxes (2D target_box).

        测试多个锚框的编码操作（2D target_box）。
        """
        prior_box = paddle.to_tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 1.0, 1.0],
            ]
        )
        target_box = paddle.to_tensor([[0.25, 0.25, 0.75, 0.75]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box,
            code_type='encode_center_size',
        )
        self.assertEqual(result.shape, (1, 3, 4))

    def test_decode_nonzero_deltas(self):
        """Test decode with non-zero deltas.

        测试非零偏移量的解码操作。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        # deltas: move center by 0.1 in x, scale width by e^0.1
        deltas = paddle.to_tensor([[[0.1, 0.0, 0.0, 0.0]]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=deltas,
            code_type='decode_center_size',
        )
        # center_x = 0.1 * 1.0 * 1.0 + 0.5 = 0.6
        # decoded_xmin = 0.6 - 1.0/2 = 0.1
        # decoded_xmax = 0.6 + 1.0/2 = 1.1
        np.testing.assert_allclose(result.numpy()[0, 0, 0], 0.1, atol=1e-4)
        np.testing.assert_allclose(result.numpy()[0, 0, 2], 1.1, atol=1e-4)

    def test_encode_float64(self):
        """Test encode_center_size with float64 dtype (2D target_box).

        测试 float64 数据类型的编码操作（2D target_box）。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]], dtype='float64')
        target_box = paddle.to_tensor(
            [[0.25, 0.25, 0.75, 0.75]], dtype='float64'
        )
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box,
            code_type='encode_center_size',
        )
        self.assertEqual(result.dtype, paddle.float64)

    def test_encode_default_variance(self):
        """Test encode_center_size with default variance [1,1,1,1].

        测试默认方差 [1,1,1,1] 的 encode_center_size 编码操作。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 1.0, 1.0]])
        target_box = paddle.to_tensor([[0.25, 0.25, 0.75, 0.75]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box,
            code_type='encode_center_size',
        )
        # encode: (0, 0, log(0.5), log(0.5))
        np.testing.assert_allclose(result.numpy()[0, 0, :2], 0.0, atol=1e-4)
        self.assertAlmostEqual(result.numpy()[0, 0, 2], np.log(0.5), places=3)

    def test_decode_scale_deltas(self):
        """Test decode with scaling deltas (log width/height).

        测试缩放偏移量（对数宽高）的解码操作。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 2.0, 2.0]])
        # log(2) for width and height scaling
        deltas = paddle.to_tensor([[[0.0, 0.0, np.log(2.0), np.log(2.0)]]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=deltas,
            code_type='decode_center_size',
        )
        # prior center=(1,1), w=2, h=2
        # width_out = exp(log(2)) * 2 = 4, height_out = 4
        # decoded: xmin = 1-2 = -1, ymin = -1, xmax = 1+2 = 3, ymax = 3
        np.testing.assert_allclose(
            result.numpy()[0, 0], [-1.0, -1.0, 3.0, 3.0], atol=1e-4
        )

    def test_encode_off_center(self):
        """Test encode with off-center target box (2D target_box).

        测试目标框偏离锚框中心的编码操作。
        """
        prior_box = paddle.to_tensor([[0.0, 0.0, 2.0, 2.0]])
        target_box = paddle.to_tensor([[1.0, 1.0, 3.0, 3.0]])
        result = paddle.vision.ops.box_coder(
            prior_box,
            prior_box_var=[1.0, 1.0, 1.0, 1.0],
            target_box=target_box,
            code_type='encode_center_size',
        )
        # prior: center=(1,1), w=2, h=2
        # target: center=(2,2), w=2, h=2
        # encode: ((2-1)/2, (2-1)/2, log(2/2), log(2/2)) = (0.5, 0.5, 0, 0)
        np.testing.assert_allclose(
            result.numpy()[0, 0], [0.5, 0.5, 0.0, 0.0], atol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
