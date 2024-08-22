#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
import scipy.stats

import paddle
from paddle import base


class TestGeometricInplaceDtype(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_geometrictype(self):
        def test_fp32():
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.geometric_(probs=0.3)
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.geometric_(probs=0.3)
            self.assertEqual(tensor_fp64.dtype, paddle.float64)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_fp32()
            test_fp64()


class TestGeometricIsInplace(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_geometric_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.geometric_(probs=0.3)
        self.assertTrue(tensor_a is tensor_b)


class TestGeometricInplaceSeedIsZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_geometric_inplace_op_not_equal(self):
        tensor = paddle.ones(self.shape)
        tensor.geometric_(probs=0.3)
        tensor_data_first = tensor.numpy()
        tensor.geometric_(probs=0.3)
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())


class TestGeometricInplaceOpShape(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_geometric_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.geometric_(probs=0.3)
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestGeometricInplaceDistribution(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.probs = 0.3

    def test_geometric_inplace_distribution(self):
        a = paddle.ones(self.shape)
        a.geometric_(self.probs)
        np.testing.assert_allclose(
            a.mean(axis=0), scipy.stats.geom.mean(self.probs), rtol=0.7, atol=0
        )
        np.testing.assert_allclose(
            a.var(axis=0), scipy.stats.geom.var(self.probs), rtol=0.7, atol=0
        )


class TestGeometricInplaceEmptyTensor(unittest.TestCase):
    def test_geometric_inplace_op_empty_tensor(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        test_shapes = [(200, 1), (1, 200)]
        for place in places:
            paddle.set_device(place)
            for test_shape in test_shapes:
                tensor = paddle.empty(shape=test_shape)
                tensor.geometric_(probs=0.3)
                tensor_shape_np = np.array(tensor.shape)
                origin_shape = np.array(test_shape)
                self.assertTrue((tensor_shape_np == origin_shape).all())


class TestGeometricInplaceGrad(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def run_(self):
        def test_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.geometric_(probs=0.3)
            loss = tensor_b.sum()
            loss.backward()
            geometric_grad = tensor_b.grad.numpy()
            self.assertTrue((geometric_grad == 0).all())

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_grad()

    def test_geometric_inplace_grad(self):
        self.run_()


if __name__ == '__main__':
    unittest.main()
