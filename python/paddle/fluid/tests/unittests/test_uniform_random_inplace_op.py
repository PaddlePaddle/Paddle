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
<<<<<<< HEAD

import numpy as np

import paddle
import paddle.fluid as fluid


class TestUniformRandomInplaceOpDtype(unittest.TestCase):
=======
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.framework import _enable_legacy_dygraph, _disable_legacy_dygraph


class TestUniformRandomInplaceOpDtype(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_dtype(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.uniform_()
        self.assertTrue(tensor_a is tensor_b)


class TestUniformRandomInplaceOpSeedIsZero(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)
        self.min = -2
        self.max = 1
        self.seed = 10

    def test_uniform_random_inplace_op_within_range(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(min=self.min, max=self.max, seed=self.seed)
        tensor_data = tensor.numpy()
<<<<<<< HEAD
        self.assertTrue(
            (tensor_data > self.min).all() and (tensor_data < self.max).all()
        )


class TestUniformRandomInplaceOpShape(unittest.TestCase):
=======
        self.assertTrue((tensor_data > self.min).all()
                        and (tensor_data < self.max).all())


class TestUniformRandomInplaceOpShape(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestUniformRandomInplaceOpDistribution(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        prob_expect = np.ones((self.bins,)) / float(self.bins)
=======
        prob_expect = np.ones((self.bins, )) / float(self.bins)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np.testing.assert_allclose(prob, prob_expect, rtol=0, atol=0.01)


class TestUniformRandomInplaceOpError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_error(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def test_attr_error():
            tensor = paddle.ones(self.shape)
            tensor.uniform_(shape=self.shape, min=-2, max=2)

        self.assertRaises(TypeError, test_attr_error)


class TestUniformRandomInplaceOpEmptyTensor(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.shape = (1000, 784)

    def run_(self):
<<<<<<< HEAD
=======
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def test_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
<<<<<<< HEAD
            tensor_b.retain_grads()
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_uniform_random_inplace_grad(self):
        self.run_()

<<<<<<< HEAD
=======
    def test_uniform_random_inplace_grad_old_dygraph(self):
        _enable_legacy_dygraph()
        self.run_()
        _disable_legacy_dygraph()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
