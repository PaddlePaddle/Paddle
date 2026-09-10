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

# Unit test for paddle.nn.functional.pooling helper functions
# Target: cover uncovered helper functions and error paths in pooling.py

import unittest

import paddle
import paddle.nn.functional as F
from paddle.nn.functional.pooling import (
    _channel_last,
    _check_input,
    _check_instance,
    _check_value_limitation,
    _update_padding_nd,
)


class TestCheckInput(unittest.TestCase):
    """Test _check_input dimension validation."""

    def test_check_input_1d_ok(self):
        """1D tensor should pass for dimension=1."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        _check_input(x, 1)  # should not raise

    def test_check_input_2d_ok(self):
        """2D tensor should pass for dimension=2."""
        x = paddle.to_tensor([[1.0, 2.0]])
        _check_input(x, 2)

    def test_check_input_wrong_dimension(self):
        """Wrong dimension should raise ValueError."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            _check_input(x, 3)


class TestCheckInstance(unittest.TestCase):
    """Test _check_instance type validation."""

    def test_check_instance_int_ok(self):
        """Integer value should pass."""
        _check_instance(3, 'kernel_size', (int,))

    def test_check_instance_float_ok(self):
        """Float value should pass."""
        _check_instance(1.5, 'padding', (int, float))

    def test_check_instance_wrong_type(self):
        """Wrong type should raise ValueError."""
        with self.assertRaises(ValueError):
            _check_instance("bad", 'kernel_size', (int, float))


class TestCheckValueLimitation(unittest.TestCase):
    """Test _check_value_limitation min value check.
    The function only raises for int values where x < min_limit.
    """

    def test_check_value_above_min(self):
        """Values above min_limit should pass."""
        _check_value_limitation([2, 3], 'kernel_size', min_limit=1)

    def test_check_value_below_min(self):
        """Int value below min_limit should raise ValueError."""
        with self.assertRaises(ValueError):
            _check_value_limitation([0], 'kernel_size', min_limit=1)

    def test_check_value_equal_min(self):
        """Int value equal to min_limit should NOT raise (only x < min_limit raises)."""
        _check_value_limitation([1], 'kernel_size', min_limit=1)

    def test_check_value_float_not_checked(self):
        """Float values are not checked by _check_value_limitation (only ints)."""
        _check_value_limitation([0.001], 'kernel_size', min_limit=1)


class TestChannelLast(unittest.TestCase):
    """Test _channel_last for different data_format and num_dims."""

    def test_1d_nlc(self):
        """1D NLC format should be channel_last."""
        self.assertTrue(_channel_last('NLC', 1))

    def test_1d_ncl(self):
        """1D NCL format should not be channel_last."""
        self.assertFalse(_channel_last('NCL', 1))

    def test_2d_nhwc(self):
        """2D NHWC format should be channel_last."""
        self.assertTrue(_channel_last('NHWC', 2))

    def test_2d_nchw(self):
        """2D NCHW format should not be channel_last."""
        self.assertFalse(_channel_last('NCHW', 2))

    def test_3d_ndhwc(self):
        """3D NDHWC format should be channel_last."""
        self.assertTrue(_channel_last('NDHWC', 3))

    def test_3d_ncdhw(self):
        """3D NCDHW format should not be channel_last."""
        self.assertFalse(_channel_last('NCDHW', 3))

    def test_invalid_1d_format(self):
        """Invalid 1D format should raise ValueError."""
        with self.assertRaises(ValueError):
            _channel_last('NCHW', 1)

    def test_invalid_2d_format(self):
        """Invalid 2D format should raise ValueError."""
        with self.assertRaises(ValueError):
            _channel_last('NCL', 2)

    def test_invalid_3d_format(self):
        """Invalid 3D format should raise ValueError."""
        with self.assertRaises(ValueError):
            _channel_last('NCHW', 3)


class TestUpdatePaddingNd(unittest.TestCase):
    """Test _update_padding_nd logic.
    _update_padding_nd returns (padding, padding_algorithm) tuple,
    where padding is first and padding_algorithm is second.
    """

    def test_same_padding(self):
        """SAME padding should return padding_algorithm='SAME'."""
        result = _update_padding_nd('SAME', num_dims=2)
        self.assertEqual(result[1], 'SAME')

    def test_valid_padding(self):
        """VALID padding should return padding_algorithm='VALID'."""
        result = _update_padding_nd('VALID', num_dims=2)
        self.assertEqual(result[1], 'VALID')

    def test_valid_padding_with_ceil_mode(self):
        """VALID padding with ceil_mode=True should raise ValueError."""
        with self.assertRaises(ValueError):
            _update_padding_nd('VALID', num_dims=2, ceil_mode=True)

    def test_unknown_padding_string(self):
        """Unknown padding string should raise ValueError."""
        with self.assertRaises(ValueError):
            _update_padding_nd('UNKNOWN', num_dims=2)

    def test_explicit_padding_with_batch_channel_channel_last(self):
        """Explicit padding including batch and channel dims (channel_last=True)."""
        padding = [[0, 0], [1, 1], [0, 0]]
        result = _update_padding_nd(padding, num_dims=1, channel_last=True)
        self.assertEqual(result[1], 'EXPLICIT')

    def test_non_zero_batch_padding_raises(self):
        """Non-zero batch/channel padding should raise ValueError."""
        padding = [[1, 0], [1, 1], [0, 0]]
        with self.assertRaises(ValueError):
            _update_padding_nd(padding, num_dims=1, channel_last=True)

    def test_explicit_padding_no_batch_channel(self):
        """Explicit padding without batch/channel dims."""
        result = _update_padding_nd([1, 2], num_dims=2)
        self.assertEqual(result[1], 'EXPLICIT')
        self.assertEqual(result[0], [1, 2])

    def test_same_padding_produces_zero_padding(self):
        """SAME padding should produce zero padding list."""
        result = _update_padding_nd('SAME', num_dims=2)
        self.assertEqual(result[0], [0, 0])


class TestPool2DErrorPaths(unittest.TestCase):
    """Test avg_pool2d / max_pool2d / avg_pool3d / max_pool3d operations."""

    def setUp(self):
        paddle.disable_static()

    def test_avg_pool2d_basic(self):
        """Basic avg_pool2d should work correctly."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.avg_pool2d(x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avg_pool2d_adaptive(self):
        """Adaptive avg_pool2d should work."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.adaptive_avg_pool2d(x, output_size=4)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_max_pool2d_basic(self):
        """Basic max_pool2d should work."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.max_pool2d(x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avg_pool3d_basic(self):
        """Basic avg_pool3d should work."""
        x = paddle.randn([2, 3, 4, 4, 4])
        out = F.avg_pool3d(x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_max_pool3d_basic(self):
        """Basic max_pool3d should work."""
        x = paddle.randn([2, 3, 4, 4, 4])
        out = F.max_pool3d(x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_avg_pool2d_adaptive_global(self):
        """Adaptive avg_pool2d with output_size=1 acts as global pooling."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.adaptive_avg_pool2d(x, output_size=1)
        self.assertEqual(out.shape, [2, 3, 1, 1])

    def test_avg_pool2d_with_padding(self):
        """avg_pool2d with explicit padding."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(out.shape, [2, 3, 8, 8])


if __name__ == '__main__':
    unittest.main()
