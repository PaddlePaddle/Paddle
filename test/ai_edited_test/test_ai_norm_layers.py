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

# Unit test for paddle.nn.layer.norm (InstanceNorm, GroupNorm, BatchNorm layers)
# Target: cover _InstanceNormBase._check_input_dim, GroupNorm invalid data_format,
#   _BatchNormBase extra_repr, _check_input_dim, _check_data_format

import unittest

import paddle
from paddle import nn
from paddle.nn.layer.norm import _BatchNormBase, _InstanceNormBase


class TestInstanceNormBaseCheckInputDim(unittest.TestCase):
    """Test that _InstanceNormBase._check_input_dim raises NotImplementedError.
    The subclass InstanceNorm1D overrides this, so we must call the base class method.
    """

    def test_base_check_input_dim_raises(self):
        """Base class _check_input_dim should raise NotImplementedError."""
        layer = _InstanceNormBase(3)
        with self.assertRaises(NotImplementedError):
            layer._check_input_dim(paddle.randn([2, 3, 10]))


class TestInstanceNormInputDimCheck(unittest.TestCase):
    """Test InstanceNorm1D checks input dimension via forward."""

    def test_instance_norm1d_wrong_dim_raises(self):
        """InstanceNorm1D should reject non-2D/3D input."""
        layer = nn.InstanceNorm1D(3)
        # 4D input should be rejected by InstanceNorm1D._check_input_dim
        x = paddle.randn([2, 3, 4, 4])
        with self.assertRaises(ValueError):
            layer(x)


class TestGroupNormErrorPaths(unittest.TestCase):
    """Test GroupNorm error paths."""

    def test_invalid_data_format(self):
        """Invalid data_format should raise ValueError."""
        with self.assertRaises(ValueError):
            nn.GroupNorm(2, 4, data_format='INVALID')

    def test_group_norm_nchw(self):
        """GroupNorm with NCHW format."""
        layer = nn.GroupNorm(2, 4, data_format='NCHW')
        x = paddle.randn([2, 4, 8, 8])
        out = layer(x)
        self.assertEqual(out.shape, [2, 4, 8, 8])

    def test_group_norm_nhwc(self):
        """GroupNorm with NHWC format."""
        layer = nn.GroupNorm(2, 4, data_format='NHWC')
        x = paddle.randn([2, 8, 8, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 8, 8, 4])

    def test_group_norm_ncl(self):
        """GroupNorm with NCL format (1D)."""
        layer = nn.GroupNorm(2, 4, data_format='NCL')
        x = paddle.randn([2, 4, 10])
        out = layer(x)
        self.assertEqual(out.shape, [2, 4, 10])

    def test_group_norm_no_affine(self):
        """GroupNorm without affine transformation."""
        layer = nn.GroupNorm(2, 4, affine=False)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)

    def test_group_norm_with_affine(self):
        """GroupNorm with affine transformation."""
        layer = nn.GroupNorm(2, 4, affine=True)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)


class TestBatchNormBaseErrorPaths(unittest.TestCase):
    """Test _BatchNormBase error paths and extra_repr."""

    def test_batch_norm_base_check_input_dim_raises(self):
        """_BatchNormBase._check_input_dim should raise NotImplementedError."""
        layer = _BatchNormBase(3)
        with self.assertRaises(NotImplementedError):
            layer._check_input_dim(paddle.randn([2, 3, 4, 4]))

    def test_batch_norm_base_check_data_format_raises(self):
        """_BatchNormBase._check_data_format should raise NotImplementedError."""
        layer = _BatchNormBase(3)
        with self.assertRaises(NotImplementedError):
            layer._check_data_format('NCHW')

    def test_batch_norm_base_extra_repr(self):
        """_BatchNormBase extra_repr should contain num_features, momentum, epsilon."""
        layer = _BatchNormBase(3, momentum=0.9, epsilon=1e-5)
        repr_str = layer.extra_repr()
        self.assertIn('num_features=3', repr_str)
        self.assertIn('momentum=0.9', repr_str)
        self.assertIn('epsilon=1e-05', repr_str)

    def test_batch_norm_base_extra_repr_no_name(self):
        """_BatchNormBase extra_repr without name should not include name."""
        layer = _BatchNormBase(3)
        repr_str = layer.extra_repr()
        self.assertNotIn('name=', repr_str)

    def test_batch_norm_base_extra_repr_nhwc(self):
        """_BatchNormBase extra_repr with NHWC data_format."""
        layer = _BatchNormBase(3, data_format='NHWC')
        repr_str = layer.extra_repr()
        self.assertIn('NHWC', repr_str)

    def test_batch_norm_base_extra_repr_with_name(self):
        """_BatchNormBase extra_repr with name parameter should include name."""
        layer = _BatchNormBase(3, name='my_bn')
        repr_str = layer.extra_repr()
        self.assertIn('name=my_bn', repr_str)

    def test_batch_norm_base_no_weight_bias(self):
        """_BatchNormBase with weight_attr=False and bias_attr=False."""
        layer = _BatchNormBase(3, weight_attr=False, bias_attr=False)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)


class TestBatchNormSubclassForward(unittest.TestCase):
    """Test BatchNorm1D/2D/3D forward passes."""

    def test_batch_norm2d_basic(self):
        """BatchNorm2D basic forward."""
        layer = nn.BatchNorm2D(3)
        x = paddle.randn([2, 3, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm1d_basic(self):
        """BatchNorm1D basic forward."""
        layer = nn.BatchNorm1D(3)
        x = paddle.randn([2, 3, 10])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 10])

    def test_batch_norm3d_basic(self):
        """BatchNorm3D basic forward."""
        layer = nn.BatchNorm3D(3)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_batch_norm2d_eval_mode(self):
        """BatchNorm2D in eval mode."""
        layer = nn.BatchNorm2D(3)
        layer.eval()
        x = paddle.randn([2, 3, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm2d_momentum(self):
        """BatchNorm2D with custom momentum."""
        layer = nn.BatchNorm2D(3, momentum=0.1)
        x = paddle.randn([2, 3, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm2d_no_weight_bias(self):
        """BatchNorm2D without weight and bias."""
        layer = nn.BatchNorm2D(3, weight_attr=False, bias_attr=False)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)


class TestInstanceNormLayers(unittest.TestCase):
    """Test InstanceNorm layers."""

    def test_instance_norm1d_basic(self):
        """InstanceNorm1D basic forward."""
        layer = nn.InstanceNorm1D(3)
        x = paddle.randn([2, 3, 10])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 10])

    def test_instance_norm2d_basic(self):
        """InstanceNorm2D basic forward."""
        layer = nn.InstanceNorm2D(3)
        x = paddle.randn([2, 3, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_instance_norm3d_basic(self):
        """InstanceNorm3D basic forward."""
        layer = nn.InstanceNorm3D(3)
        x = paddle.randn([2, 3, 4, 4, 4])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_instance_norm_no_affine(self):
        """InstanceNorm without affine transformation."""
        layer = nn.InstanceNorm1D(3, weight_attr=False, bias_attr=False)
        self.assertIsNone(layer.scale)
        self.assertIsNone(layer.bias)

    def test_instance_norm_eval(self):
        """InstanceNorm in eval mode."""
        layer = nn.InstanceNorm1D(3)
        layer.eval()
        x = paddle.randn([2, 3, 10])
        out = layer(x)
        self.assertEqual(out.shape, [2, 3, 10])


if __name__ == '__main__':
    unittest.main()
