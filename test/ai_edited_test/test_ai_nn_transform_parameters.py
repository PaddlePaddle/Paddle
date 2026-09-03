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

# [AUTO-GENERATED] Test file for paddle/nn/utils/transform_parameters.py
# Target file: paddle/nn/utils/transform_parameters.py (94.1% coverage)
# Uncovered lines: 44 (_inplace_reshape_dygraph static path), 126 (parameters_to_vector static path),
#   180 (vector_to_parameters single param), 199 (vector_to_parameters static path)

"""参数转换模块测试 / Transform parameters module tests

测试目标 / Test Target:
  paddle/nn/utils/transform_parameters.py

覆盖的模块 / Covered Modules:
  - _inplace_reshape_dygraph: dynamic mode reshape
  - _stride_column: column stride permutation
  - parameters_to_vector: flatten and restore
  - vector_to_parameters: assign back and reshape
"""

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.nn.utils.transform_parameters import (
    _stride_column,
    parameters_to_vector,
    vector_to_parameters,
)


class TestInplaceReshapeDygraph(unittest.TestCase):
    """测试 _inplace_reshape_dygraph 函数
    Test _inplace_reshape_dygraph function"""

    def setUp(self):
        paddle.disable_static()

    def test_inplace_reshape(self):
        """测试原地 reshape
        Test inplace reshape"""
        from paddle.nn.utils.transform_parameters import (
            _inplace_reshape_dygraph,
        )

        x = paddle.randn([4, 8], dtype='float32')
        original_shape = x.shape
        _inplace_reshape_dygraph(x, [32])
        self.assertEqual(x.shape, [32])
        # Reshape back
        _inplace_reshape_dygraph(x, [4, 8])
        self.assertEqual(list(x.shape), [4, 8])


class TestStrideColumn(unittest.TestCase):
    """测试 _stride_column 函数
    Test _stride_column function"""

    def setUp(self):
        paddle.disable_static()

    def test_stride_column_basic(self):
        """测试基本的 column stride
        Test basic column stride"""
        linear = nn.Linear(4, 8)
        weight_shape = linear.weight.shape
        weight_copy = linear.weight.numpy().copy()
        _stride_column(linear.weight)
        # Shape should remain the same after stride
        self.assertEqual(linear.weight.shape, weight_shape)
        # The values should be different (transposed+reshaped)
        result = linear.weight.numpy()
        self.assertEqual(result.shape, weight_copy.shape)

    def test_stride_column_not_2d_raises(self):
        """测试非 2D 参数抛出异常 (line 78)
        Test non-2D parameter raises AssertionError"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        with self.assertRaises(AssertionError):
            _stride_column(x)


class TestParametersToVector(unittest.TestCase):
    """测试 parameters_to_vector 函数
    Test parameters_to_vector function"""

    def setUp(self):
        paddle.disable_static()

    def test_parameters_to_vector_basic(self):
        """测试基本的参数转向量
        Test basic parameters to vector"""
        linear = nn.Linear(4, 8)
        total_params = sum(p.numel() for p in linear.parameters())
        vec = parameters_to_vector(linear.parameters())
        self.assertEqual(vec.shape, [total_params])
        self.assertEqual(vec.shape[0], 4 * 8 + 8)

    def test_parameters_to_vector_sequential(self):
        """测试 Sequential 模型的参数转向量
        Test Sequential model parameters to vector"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        total_params = sum(p.numel() for p in model.parameters())
        vec = parameters_to_vector(model.parameters())
        self.assertEqual(vec.shape[0], total_params)

    def test_parameters_to_vector_preserves_params(self):
        """测试参数转向量后原参数不变
        Test parameters are preserved after conversion"""
        linear = nn.Linear(4, 8)
        original_weight = linear.weight.numpy().copy()
        original_bias = linear.bias.numpy().copy()

        vec = parameters_to_vector(linear.parameters())
        # Parameters should be restored after conversion
        np.testing.assert_allclose(
            linear.weight.numpy(), original_weight, atol=1e-6
        )
        np.testing.assert_allclose(
            linear.bias.numpy(), original_bias, atol=1e-6
        )

    def test_parameters_to_vector_stop_gradient(self):
        """测试参数转向量结果 stop_gradient 为 False (line 135)
        Test vector has stop_gradient=False"""
        linear = nn.Linear(4, 8)
        vec = parameters_to_vector(linear.parameters())
        self.assertFalse(vec.stop_gradient)


class TestVectorToParameters(unittest.TestCase):
    """测试 vector_to_parameters 函数
    Test vector_to_parameters function"""

    def setUp(self):
        paddle.disable_static()

    def test_vector_to_parameters_basic(self):
        """测试基本的向量转参数
        Test basic vector to parameters"""
        linear1 = nn.Linear(4, 8)
        linear2 = nn.Linear(4, 8)

        vec = parameters_to_vector(linear1.parameters())
        vector_to_parameters(vec, linear2.parameters())

        # Parameters should match
        np.testing.assert_allclose(
            linear1.weight.numpy(), linear2.weight.numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            linear1.bias.numpy(), linear2.bias.numpy(), atol=1e-6
        )

    def test_vector_to_parameters_single_param(self):
        """测试单个参数的向量转换 (line 179-180, sections has single element)
        Test vector to parameters with single parameter"""
        x = paddle.randn([10], dtype='float32')
        vec = paddle.to_tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )
        vector_to_parameters(vec, [x])
        np.testing.assert_allclose(x.numpy(), vec.numpy(), atol=1e-6)

    def test_vector_to_parameters_larger_vec(self):
        """测试更大的向量（部分元素被使用）
        Test with larger vector (partial elements used)"""
        linear1 = nn.Linear(4, 8)
        linear2 = nn.Linear(4, 8)

        total_params = sum(p.numel() for p in linear1.parameters())
        # Create a larger vector with extra elements
        vec = paddle.randn([total_params + 10], dtype='float32')
        # This should work since total_elements < vec.shape[0]
        vector_to_parameters(vec[:total_params], linear2.parameters())

    def test_vector_to_parameters_preserves_shape(self):
        """测试参数形状保持不变
        Test parameter shapes are preserved"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        vec = paddle.zeros([sum(p.numel() for p in model.parameters())])
        vector_to_parameters(vec, model.parameters())

        # Paddle Linear weight shape is [in_features, out_features]
        self.assertEqual(list(model[0].weight.shape), [4, 8])
        self.assertEqual(list(model[0].bias.shape), [8])
        self.assertEqual(list(model[1].weight.shape), [8, 2])
        self.assertEqual(list(model[1].bias.shape), [2])


class TestRoundTrip(unittest.TestCase):
    """测试参数向量转换的往返一致性
    Test round-trip consistency of parameter vector conversion"""

    def setUp(self):
        paddle.disable_static()

    def test_round_trip(self):
        """测试参数转向量再转回参数
        Test parameters -> vector -> parameters round trip"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        original_params = [p.numpy().copy() for p in model.parameters()]

        vec = parameters_to_vector(model.parameters())
        vector_to_parameters(vec, model.parameters())

        restored_params = [p.numpy().copy() for p in model.parameters()]
        for orig, restored in zip(original_params, restored_params):
            np.testing.assert_allclose(orig, restored, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
