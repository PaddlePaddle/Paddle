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

# [AUTO-GENERATED] Unit test for paddle.framework.recall_error.check_naninf
# Target: cover uncovered lines 25-28, 30 in recall_error.py

import unittest

import numpy as np

import paddle
from paddle.framework.recall_error import (
    LOSS_INF_ERROR,
    LOSS_NAN_ERROR,
    check_naninf,
)


class TestCheckNaninf(unittest.TestCase):
    """Test cases for paddle.framework.recall_error.check_naninf function."""

    def setUp(self):
        paddle.disable_static()

    def test_finite_tensor_returns_none(self):
        """When all values in the tensor are finite, check_naninf should return None."""
        tensor = paddle.to_tensor([1.0, 2.0, 3.0, -1.0, 0.0])
        result = check_naninf(tensor)
        self.assertIsNone(result)

    def test_nan_tensor_returns_nan_error(self):
        """When the tensor contains NaN, check_naninf should return LOSS_NAN_ERROR."""
        data = np.array([1.0, float('nan'), 3.0], dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_NAN_ERROR)

    def test_inf_tensor_returns_inf_error(self):
        """When the tensor contains Inf (but no NaN), check_naninf should return LOSS_INF_ERROR."""
        data = np.array([1.0, float('inf'), 3.0], dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_INF_ERROR)

    def test_neg_inf_tensor_returns_inf_error(self):
        """When the tensor contains -Inf, check_naninf should return LOSS_INF_ERROR."""
        data = np.array([1.0, float('-inf'), 3.0], dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_INF_ERROR)

    def test_nan_and_inf_tensor_returns_nan_error(self):
        """When the tensor contains both NaN and Inf, check_naninf should return LOSS_NAN_ERROR
        because NaN check takes priority over Inf (isfinite fails, then isnan is True)."""
        data = np.array([float('nan'), float('inf'), 3.0], dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_NAN_ERROR)

    def test_scalar_finite_tensor(self):
        """Test with a scalar tensor that is finite."""
        tensor = paddle.to_tensor(42.0)
        result = check_naninf(tensor)
        self.assertIsNone(result)

    def test_scalar_nan_tensor(self):
        """Test with a scalar NaN tensor."""
        data = np.array(float('nan'), dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_NAN_ERROR)

    def test_scalar_inf_tensor(self):
        """Test with a scalar Inf tensor."""
        data = np.array(float('inf'), dtype='float32')
        tensor = paddle.to_tensor(data)
        result = check_naninf(tensor)
        self.assertEqual(result, LOSS_INF_ERROR)


if __name__ == '__main__':
    unittest.main()
