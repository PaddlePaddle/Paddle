#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_uint16_to_float

import paddle
from paddle import base
from paddle.base import core


def output_hist(out):
    hist, _ = np.histogram(out, range=(-1, 1))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


class TestUniformRandomInplaceOpDtype(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_dtype(self):
        def test_fp32():
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.uniform_()
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.uniform_()
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


class TestUniformRandomInplaceFP16Op(OpTest):
    def setUp(self):
        self.op_type = "uniform_random_inplace"
        self.dtype = np.float16
        self.shape = (1000, 784)
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        self.inputs = {"X": x}
        self.outputs = {"Out": y}
        self.init_attrs()

    def init_attrs(self):
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0.015, atol=0.001)

    # TODO: Due to the lack of the self.python_api=paddle.uniform_random_inplace setting, the dynamic graph is temporarily turned off, set check_dygraph=False
    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_dygraph=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestUniformRandomInplaceBF16Op(OpTest):
    def setUp(self):
        self.op_type = "uniform_random_inplace"
        self.dtype = np.uint16
        self.shape = (1000, 784)
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y}
        self.init_attrs()
        self.place = core.CUDAPlace(0)

    def init_attrs(self):
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_with_place_customized(self.verify_output, self.place)

    def verify_output(self, outs):
        result = convert_uint16_to_float(np.array(outs[0]))
        hist, prob = self.output_hist(result)
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.002)

    # TODO: Due to the lack of the self.python_api=paddle.uniform_random_inplace setting, the dynamic graph is temporarily turned off, set check_dygraph=False
    def test_check_grad(self):
        grads = [paddle.zeros(self.shape, dtype=self.dtype)]
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            check_dygraph=False,
            user_defined_grads=grads,
        )


class TestUniformRandomInplaceOpIsInplace(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.uniform_()
        self.assertTrue(tensor_a is tensor_b)


class TestUniformRandomInplaceOpSeedIsZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.seed = 0

    def test_uniform_random_inplace_op_seed_is_zero(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(seed=self.seed)
        tensor_data_first = tensor.numpy()
        tensor.uniform_(seed=self.seed)
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())


class TestUniformRandomInplaceOpSeedIsNotZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.seed = 10

    def test_uniform_random_inplace_op_seed_is_not_zero(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(seed=self.seed)
        tensor_data_first = tensor.numpy()
        tensor.uniform_(seed=self.seed)
        tensor_data_second = tensor.numpy()
        self.assertTrue((tensor_data_first == tensor_data_second).all())


class TestUniformRandomInplaceOpWithinRange(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.min = -2
        self.max = 1
        self.seed = 10

    def test_uniform_random_inplace_op_within_range(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(min=self.min, max=self.max, seed=self.seed)
        tensor_data = tensor.numpy()
        self.assertTrue(
            (tensor_data > self.min).all() and (tensor_data < self.max).all()
        )


class TestUniformRandomInplaceOpShape(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestUniformRandomInplaceOpDistribution(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.min = -3
        self.max = 5
        self.seed = 10
        self.bins = 100

    def test_uniform_random_inplace_op_distribution(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(self.min, self.max, self.seed)

        hist, _ = np.histogram(tensor.numpy()[0], bins=self.bins)
        prob = hist / float(self.shape[0])
        prob_expect = np.ones((self.bins,)) / float(self.bins)
        np.testing.assert_allclose(prob, prob_expect, rtol=0, atol=0.01)


class TestUniformRandomInplaceOpError(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_error(self):
        def test_attr_error():
            tensor = paddle.ones(self.shape)
            tensor.uniform_(shape=self.shape, min=-2, max=2)

        self.assertRaises(TypeError, test_attr_error)


class TestUniformRandomInplaceOpEmptyTensor(unittest.TestCase):
    def test_uniform_random_inplace_op_empty_tensor(self):
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
                tensor.uniform_()
                tensor_shape_np = np.array(tensor.shape)
                origin_shape = np.array(test_shape)
                self.assertTrue((tensor_shape_np == origin_shape).all())


class TestUniformRandomInplaceGrad(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def run_(self):
        def test_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.uniform_(min=-2, max=2)
            loss = tensor_b.sum()
            loss.backward()
            uniform_grad = tensor_b.grad.numpy()
            self.assertTrue((uniform_grad == 0).all())

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

    def test_uniform_random_inplace_grad(self):
        self.run_()


if __name__ == '__main__':
    unittest.main()
