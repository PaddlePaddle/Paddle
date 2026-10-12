# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Unit test for paddle.nn.functional.norm
# Target: cover normalize, batch_norm, layer_norm code paths

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestNormalizeErrorPaths(unittest.TestCase):
    """Test normalize() error paths and edge cases."""

    def setUp(self):
        paddle.disable_static()

    def test_normalize_basic_float32(self):
        """Basic normalize with float32 input."""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'
        )
        out = F.normalize(x, p=2.0, axis=1)
        norms = paddle.norm(out, p=2.0, axis=1)
        np.testing.assert_allclose(
            norms.numpy(), np.array([1.0, 1.0]), rtol=1e-5
        )

    def test_normalize_p1(self):
        """Normalize with p=1 (L1 normalization)."""
        x = paddle.to_tensor([[3.0, 0.0, 4.0]], dtype='float32')
        out = F.normalize(x, p=1.0, axis=1)
        norms = paddle.norm(out, p=1.0, axis=1)
        np.testing.assert_allclose(norms.numpy(), np.array([1.0]), rtol=1e-5)

    def test_normalize_inf(self):
        """Normalize with p=float('inf') (max normalization)."""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float32')
        out = F.normalize(x, p=float('inf'), axis=1)
        max_vals = paddle.max(paddle.abs(out), axis=1)
        np.testing.assert_allclose(max_vals.numpy(), np.array([1.0]), rtol=1e-5)

    def test_normalize_axis_0(self):
        """Normalize along axis=0."""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        out = F.normalize(x, p=2.0, axis=0)
        norms = paddle.norm(out, p=2.0, axis=0)
        np.testing.assert_allclose(
            norms.numpy(), np.array([1.0, 1.0]), rtol=1e-5
        )

    def test_normalize_negative_axis(self):
        """Normalize with negative axis."""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float32')
        out = F.normalize(x, p=2.0, axis=-1)
        norms = paddle.norm(out, p=2.0, axis=-1)
        np.testing.assert_allclose(norms.numpy(), np.array([1.0]), rtol=1e-5)

    def test_normalize_epsilon(self):
        """Normalize with custom epsilon."""
        x = paddle.to_tensor([[0.0, 0.0, 0.0]], dtype='float32')
        out = F.normalize(x, p=2.0, axis=1, epsilon=1e-12)
        np.testing.assert_allclose(out.numpy(), np.zeros((1, 3)), atol=1e-6)

    def test_normalize_float64(self):
        """Normalize with float64 input."""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float64')
        out = F.normalize(x, p=2.0, axis=1)
        norms = paddle.norm(out, p=2.0, axis=1)
        np.testing.assert_allclose(norms.numpy(), np.array([1.0]), rtol=1e-5)


class TestBatchNormErrorPaths(unittest.TestCase):
    """Test batch_norm() error paths."""

    def setUp(self):
        paddle.disable_static()

    def test_batch_norm_invalid_data_format(self):
        """Invalid data_format should raise ValueError."""
        x = paddle.randn([2, 3, 4, 4])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        with self.assertRaises(ValueError):
            F.batch_norm(x, mean, var, data_format='INVALID')

    def test_batch_norm_nchw(self):
        """batch_norm with NCHW format should work."""
        x = paddle.randn([2, 3, 4, 4])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        out = F.batch_norm(x, mean, var, data_format='NCHW')
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_nhwc(self):
        """batch_norm with NHWC format should work."""
        x = paddle.randn([2, 4, 4, 3])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        out = F.batch_norm(x, mean, var, data_format='NHWC')
        self.assertEqual(out.shape, [2, 4, 4, 3])

    def test_batch_norm_1d_ncl(self):
        """batch_norm with 1D NCL format."""
        x = paddle.randn([2, 3, 10])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        out = F.batch_norm(x, mean, var, data_format='NCL')
        self.assertEqual(out.shape, [2, 3, 10])

    def test_batch_norm_3d_ncdhw(self):
        """batch_norm with 3D NCDHW format."""
        x = paddle.randn([2, 3, 4, 4, 4])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        out = F.batch_norm(x, mean, var, data_format='NCDHW')
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_batch_norm_with_training_false(self):
        """batch_norm with training=False should use global stats."""
        x = paddle.randn([2, 3, 4, 4])
        mean = paddle.zeros([3])
        var = paddle.ones([3])
        out = F.batch_norm(x, mean, var, training=False)
        self.assertEqual(out.shape, [2, 3, 4, 4])


class TestLayerNormErrorPaths(unittest.TestCase):
    """Test layer_norm() error paths."""

    def setUp(self):
        paddle.disable_static()

    def test_layer_norm_shape_mismatch(self):
        """Shape mismatch between normalized_shape and input should raise ValueError."""
        x = paddle.randn([2, 3, 4])
        with self.assertRaises(ValueError):
            F.layer_norm(x, normalized_shape=[5])

    def test_layer_norm_normalized_dim_too_large(self):
        """normalized_shape larger than input dims should raise ValueError."""
        x = paddle.randn([2, 3, 4])
        with self.assertRaises(ValueError):
            F.layer_norm(x, normalized_shape=[2, 3, 4, 5])

    def test_layer_norm_basic(self):
        """Basic layer_norm should work."""
        x = paddle.randn([2, 3, 4])
        out = F.layer_norm(x, normalized_shape=[3, 4])
        self.assertEqual(out.shape, [2, 3, 4])

    def test_layer_norm_with_weight_bias(self):
        """layer_norm with 1D weight and bias matching normalized_shape."""
        x = paddle.randn([2, 4])
        w = paddle.ones([4])
        b = paddle.zeros([4])
        out = F.layer_norm(x, normalized_shape=[4], weight=w, bias=b)
        self.assertEqual(out.shape, [2, 4])

    def test_layer_norm_epsilon(self):
        """layer_norm with custom epsilon."""
        x = paddle.randn([2, 4])
        out = F.layer_norm(x, normalized_shape=[4], epsilon=1e-8)
        self.assertEqual(out.shape, [2, 4])


if __name__ == '__main__':
    unittest.main()
