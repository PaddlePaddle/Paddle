#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class TestDygraphInplace(unittest.TestCase):
    def setUp(self):
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def non_inplace_api_processing(self, var):
        return paddle.squeeze(var)

    def inplace_api_processing(self, var):
        return paddle.squeeze_(var)

    def test_inplace_api(self):
        with _test_eager_guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            inplace_var = self.inplace_api_processing(var)
            self.assertTrue(id(var) == id(inplace_var))

            inplace_var.exp_()
            self.assertTrue(np.array_equal(var.numpy(), inplace_var.numpy()))

    def test_forward_version(self):
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
                self.assertEqual(var.inplace_version, 0)

                inplace_var = self.inplace_api_processing(var)
                self.assertEqual(var.inplace_version, 1)

                inplace_var.exp_()
                self.assertEqual(var.inplace_version, 2)

                inplace_var = self.inplace_api_processing(inplace_var)
                self.assertEqual(var.inplace_version, 3)

    def test_leaf_inplace_var_error(self):
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
                var.stop_gradient = False

                def leaf_inplace_error():
                    self.inplace_api_processing(var)

                self.assertRaises(ValueError, leaf_inplace_error)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2

                # Here, the gradient computation will use the value of var_b
                var_c = var_b**2
                self.inplace_api_processing(var_b)

                loss = paddle.nn.functional.relu(var_c)
                with self.assertRaisesRegexp(
                        RuntimeError,
                        "received current_inplace_version:{} != inplace_version_snapshot_:{}".
                        format(1, 0)):
                    loss.backward()

    def test_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                # Here, the gradient computation will use the value of var_b
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_c = self.non_inplace_api_processing(var_b)
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a = var_a.grad.numpy()

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

    def test_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2

                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2

                var_c = self.non_inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a = var_a.grad.numpy()
        self.assertTrue(np.array_equal(grad_var_a_inplace, grad_var_a))

    # inplace + hook
    def test_backward_success_3(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        def double_hook(grad):
            grad = grad * 2
            return grad

        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False
                helper = var_a.register_hook(double_hook)

                var_b = var_a**2
                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                # Here, the gradient computation will use the value of var_b
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False
                helper = var_a.register_hook(double_hook)

                var_b = var_a**2
                var_c = self.non_inplace_api_processing(var_b)
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a = var_a.grad.numpy()

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

    # inplace + hook
    def test_backward_success_4(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        def double_hook(grad):
            grad = grad * 2
            return grad

        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False
                var_a.register_hook(double_hook)

                var_b = var_a**2

                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False
                var_a.register_hook(double_hook)

                var_b = var_a**2

                var_c = self.non_inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a = var_a.grad.numpy()
        self.assertTrue(np.array_equal(grad_var_a_inplace, grad_var_a))

    # inplace + hook
    def test_backward_success_5(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        def double_hook(grad):
            grad = grad * 2
            return grad

        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_b.register_hook(double_hook)
                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                # Here, the gradient computation will use the value of var_b
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_b.register_hook(double_hook)
                var_c = self.non_inplace_api_processing(var_b)
                var_d = var_c**2
                loss = var_d.sum()
                loss.backward()
                grad_var_a = var_a.grad.numpy()

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

    # inplace + hook
    def test_backward_success_6(self):
        # Although var_b is modified inplace before using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        def double_hook(grad):
            grad = grad * 2
            return grad

        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_b.register_hook(double_hook)
                var_c = self.inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.to_tensor(self.input_var_numpy).astype(
                    self.dtype)
                var_a.stop_gradient = False

                var_b = var_a**2
                var_b.register_hook(double_hook)
                var_c = self.non_inplace_api_processing(
                    var_b)  # var_b is modified inplace before using it

                var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
                loss = var_d.sum()

                loss.backward()
                grad_var_a = var_a.grad.numpy()
        self.assertTrue(np.array_equal(grad_var_a_inplace, grad_var_a))


class TestDygraphInplaceUnsqueeze(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.unsqueeze(var, -1)

    def inplace_api_processing(self, var):
        return paddle.unsqueeze_(var, -1)


class TestDygraphInplaceReshape(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.reshape(var, [-1])

    def inplace_api_processing(self, var):
        return paddle.reshape_(var, [-1])


class TestDygraphInplaceFlatten(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.flatten()

    def inplace_api_processing(self, var):
        return var.flatten_()


"""
# This case will fail while using `_C_ops.final_state_scatter`.
class TestDygraphInplaceScatter(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.array([[1, 1], [2, 2], [3, 3]])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

        return paddle.scatter(var, index, updates, overwrite=False)

    def inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

        return paddle.scatter_(var, index, updates, overwrite=False)
"""


class TestDygraphInplaceElu(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.elu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.elu_(var)


class TestDygraphInplaceRelu(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.relu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.relu_(var)


class TestDygraphInplaceSoftmax(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.softmax(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_(var)


class TestDygraphInplaceTanh(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.tanh(var)

    def inplace_api_processing(self, var):
        return paddle.tanh_(var)


class TestDygraphInplaceCeil(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.ceil()

    def inplace_api_processing(self, var):
        return var.ceil_()


class TestDygraphInplaceFloor(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.floor()

    def inplace_api_processing(self, var):
        return var.floor_()


class TestDygraphInplaceExp(TestDygraphInplace):
    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def non_inplace_api_processing(self, var):
        return var.exp()

    def inplace_api_processing(self, var):
        return var.exp_()


class TestDygraphInplaceReciprocal(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.reciprocal()

    def inplace_api_processing(self, var):
        return var.reciprocal_()


class TestDygraphInplaceRound(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.round()

    def inplace_api_processing(self, var):
        return var.round_()


class TestDygraphInplaceSqrt(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(0, 5, [10, 20, 1])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return var.sqrt()

    def inplace_api_processing(self, var):
        return var.sqrt_()


class TestDygraphInplaceRsqrt(TestDygraphInplaceSqrt):
    def non_inplace_api_processing(self, var):
        return var.rsqrt()

    def inplace_api_processing(self, var):
        return var.rsqrt_()


class TestDygraphInplaceClip(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.clip(0.6, 1.5)

    def inplace_api_processing(self, var):
        return var.clip_(0.6, 1.5)


class TestDygraphInplaceScale(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.scale(scale=2.0, bias=3.0)

    def inplace_api_processing(self, var):
        return var.scale_(scale=2.0, bias=3.0)


class TestDygraphInplaceAdd(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.rand(2, 3, 4)
        self.dtype = "float32"
        self.input_var_numpy_2 = np.random.rand(2, 3, 4).astype(self.dtype)

    def non_inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.add(input_var_2)

    def inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.add_(input_var_2)


class TestDygraphInplaceSubtract(TestDygraphInplaceAdd):
    def non_inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.subtract(input_var_2)

    def inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.subtract_(input_var_2)


class TestLossIsInplaceVar(unittest.TestCase):
    def test_loss_is_inplace_var(self):
        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.ones((2, 2))
                var_a.stop_gradient = False

                var_b = var_a * 2
                loss = var_b.tanh_()

                loss.backward()
                inplace_grad_var_a = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            with _test_eager_guard():
                var_a = paddle.ones((2, 2))
                var_a.stop_gradient = False

                var_b = var_a * 2
                loss = var_b.tanh()

                loss.backward()
                grad_var_a = var_a.grad.numpy()

        self.assertTrue(np.array_equal(inplace_grad_var_a, grad_var_a))


class TestContinuouslyInplace(unittest.TestCase):
    def test_continuously_inplace(self):
        with _test_eager_guard():
            a = paddle.rand([2, 3])
            a.stop_gradient = False
            b = a * 2

            b.reshape_([-1])
            b.reshape_([2, 3])
            b.reshape_([-1])

            b.backward()


if __name__ == '__main__':
    unittest.main()
