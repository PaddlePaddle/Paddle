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

import unittest

import numpy as np

import paddle
from paddle import base


def output_hist(out):
    hist, _ = np.histogram(out, range=(-1, 1))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


class TestNormalRandomInplaceOpDtype(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_normal_inplace_op_dtype(self):
        def test_fp32():
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.normal_()
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.normal_()
            self.assertEqual(tensor_fp64.dtype, paddle.float64)

        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_fp32()
            test_fp64()


class TestNormalRandomInplaceOpIsInplace(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_normal_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.normal_()
        self.assertTrue(tensor_a is tensor_b)


class TestNormalRandomInplaceOpSeedIsZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_normal_inplace_op_not_equal(self):
        tensor = paddle.ones(self.shape)
        tensor.normal_()
        tensor_data_first = tensor.numpy()
        tensor.normal_()
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())


class TestNormalRandomInplaceOpShape(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_normal_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.normal_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestNormalRandomInplaceOpDistribution(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.mean = -3
        self.std = 5

    def test_normal_inplace_op_distribution(self):
        tensor = paddle.ones(self.shape)
        tensor.normal_(self.mean, self.std)
        ones = paddle.ones(self.shape)
        zeros = paddle.zeros(self.shape)
        all_num = self.shape[0] * self.shape[1]

        std_probs = [0.68, 0.95, 0.997]
        for index, prob in enumerate(std_probs):
            left = self.mean - (index + 1) * self.std
            right = self.mean + (index + 1) * self.std
            cond = paddle.logical_and(tensor >= left, tensor <= right)
            c_sum = paddle.where(cond, ones, zeros).sum()
            np.testing.assert_allclose((c_sum / all_num), prob, 1e-2)


class TestNormalRandomInplaceOpEmptyTensor(unittest.TestCase):
    def test_normal_inplace_op_empty_tensor(self):
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        test_shapes = [(200, 0), (0, 200)]
        for place in places:
            paddle.set_device(place)
            for test_shape in test_shapes:
                tensor = paddle.empty(shape=test_shape)
                tensor.normal_()
                tensor_shape_np = np.array(tensor.shape)
                origin_shape = np.array(test_shape)
                self.assertTrue((tensor_shape_np == origin_shape).all())


class TestNormalRandomInplaceGrad(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def run_(self):
        def test_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.normal_(mean=-2, std=2)
            loss = tensor_b.sum()
            loss.backward()
            normal_grad = tensor_b.grad.numpy()
            self.assertTrue((normal_grad == 0).all())

        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_grad()

    def test_normal_inplace_grad(self):
        self.run_()


if __name__ == '__main__':
    unittest.main()
