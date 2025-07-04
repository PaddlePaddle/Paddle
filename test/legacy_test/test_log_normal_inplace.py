#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base


def log_normal_mean(mean, std):
    return np.exp(mean + np.power(std, 2) / 2.0)


def log_normal_var(mean, std):
    var = np.power(std, 2)
    return (np.exp(var) - 1.0) * np.exp(2 * mean + var)


class TestLogNormalRandomInplaceOpDtype(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_log_normal_inplace_op_dtype(self):
        def test_fp32():
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.log_normal_()
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.log_normal_()
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


class TestLogNormalRandomInplaceOpIsInplace(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_log_normal_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.log_normal_()
        self.assertTrue(tensor_a is tensor_b)


class TestLogNormalRandomInplaceOpSeedIsZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_log_normal_inplace_op_not_equal(self):
        tensor = paddle.ones(self.shape)
        tensor.log_normal_()
        tensor_data_first = tensor.numpy()
        tensor.log_normal_()
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())


class TestLogNormalRandomInplaceOpShape(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_log_normal_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.log_normal_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestLogNormalRandomInplaceOpDistribution(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.mean = -1
        self.std = 1

    def test_log_normal_inplace_op_distribution(self):
        tensor = paddle.ones(self.shape)
        tensor.log_normal_(self.mean, self.std)
        mean = np.mean(tensor.numpy())
        var = np.var(tensor.numpy())
        mean_ref = log_normal_mean(self.mean, self.std)
        var_ref = log_normal_var(self.mean, self.std)
        np.testing.assert_allclose(mean_ref, mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(var_ref, var, rtol=0.2, atol=0.2)


class TestLogNormalRandomInplaceOpEmptyTensor(unittest.TestCase):
    def test_log_normal_inplace_op_empty_tensor(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        test_shapes = [(200, 0), (0, 200)]
        for place in places:
            paddle.set_device(place)
            for test_shape in test_shapes:
                tensor = paddle.empty(shape=test_shape)
                tensor.log_normal_()
                tensor_shape_np = np.array(tensor.shape)
                origin_shape = np.array(test_shape)
                self.assertTrue((tensor_shape_np == origin_shape).all())


class TestLogNormalRandomInplaceGrad(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def run_(self):
        def test_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.log_normal_(mean=-2.0, std=2.0)
            loss = tensor_b.sum()
            loss.backward()
            log_normal_grad = tensor_b.grad.numpy()
            self.assertTrue((log_normal_grad == 0).all())

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

    def test_log_normal_inplace_grad(self):
        self.run_()


if __name__ == '__main__':
    unittest.main()
