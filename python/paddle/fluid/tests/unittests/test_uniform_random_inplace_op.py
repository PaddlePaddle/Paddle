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

import unittest

import numpy as np
from op_test import OpTest,convert_float_to_uint16
import paddle
import paddle.fluid as fluid

class TestUniformInplaceOp(OpTest):
    def init_args(self):
        self.shape = (2, 3)
        self.dtype = np.float32
        self.low = -1.0
        self.high = 1.0
        self.seed = 123

    def setUp(self):
        self.op_type = "uniform"
        self.python_api = paddle.fluid.layers.uniform
        self.init_args()
        np.random.seed(self.seed)
        x = np.random.randn(*self.shape).astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'shape': self.shape, 'low': self.low, 'high': self.high, 'seed': self.seed}
        out = np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
        self.outputs = {'Out': out}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out', check_dygraph=False)

class TestUniformInplaceFP16Op(OpTest):
    def init_args(self):
        self.shape = (2, 3)
        self.dtype = np.float16
        self.low = -1.0
        self.high = 1.0
        self.seed = 123

    def setUp(self):
        self.op_type = "uniform"
        self.python_api = paddle.fluid.layers.uniform
        self.init_args()
        np.random.seed(self.seed)
        x = np.random.randn(*self.shape).astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'shape': self.shape, 'low': self.low, 'high': self.high, 'seed': self.seed}
        out = np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
        self.outputs = {'Out': out}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out', check_dygraph=False)

class TestUniformInplaceBF16(OpTest):
    def init_args(self):
        self.shape = (2, 3)
        self.dtype = np.uint16
        self.low = -1.0
        self.high = 1.0
        self.seed = 123

    def setUp(self):
        self.op_type = "uniform"
        self.python_api = paddle.fluid.layers.uniform
        self.init_args()
        np.random.seed(self.seed)
        x = np.random.randn(*self.shape).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'shape': self.shape, 'low': self.low, 'high': self.high, 'seed': self.seed}
        out = np.random.uniform(self.low, self.high, self.shape)
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out', check_dygraph=False)
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

        places = ['cpu']
        if fluid.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_fp32()
            test_fp64()


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
        places = ['cpu']
        if fluid.core.is_compiled_with_cuda():
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

        places = ['cpu']
        if fluid.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_grad()

    def test_uniform_random_inplace_grad(self):
        self.run_()


if __name__ == '__main__':
    unittest.main()
