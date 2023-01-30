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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle


class TestInplace(unittest.TestCase):
    def test_forward_version(self):
=======
from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard, in_dygraph_mode


class TestInplace(unittest.TestCase):

    def func_test_forward_version(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)

            var[0] = 1.1
            self.assertEqual(var.inplace_version, 1)

            paddle.assign(paddle.ones(shape=[3]), var)

            # NOTE(liym27): assign(input, output) is an inplace operation for output.
            # There is inplace-related processing for api assign, var.inplace_version should be 2 not 1.
            self.assertEqual(var.inplace_version, 2)

            var[2] = 3
            self.assertEqual(var.inplace_version, 3)

<<<<<<< HEAD
    def test_backward_error(self):
=======
    def test_forward_version(self):
        with _test_eager_guard():
            self.func_test_forward_version()
        self.func_test_forward_version()

    def func_test_backward_error(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            var_b[1:2] = 3.3  # var_b is modified inplace after using it

            var_d = var_b**2

            loss = paddle.nn.functional.relu(var_c + var_d)
<<<<<<< HEAD
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:{} != wrapper_version_snapshot:{}".format(
                    1, 0
                ),
            ):
                loss.backward()

    def test_backward_success_1(self):
=======
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()

    def test_backward_error(self):
        with _test_eager_guard():
            self.func_test_backward_error()
        self.func_test_backward_error()

    def func_test_backward_success_1(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2
            var_b[1:2] = 3  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            loss = var_c.sum()
            loss.backward()

<<<<<<< HEAD
    def test_backward_success_2(self):
=======
    def test_backward_success_1(self):
        with _test_eager_guard():
            self.func_test_backward_success_1()
        self.func_test_backward_success_1()

    def func_test_backward_success_2(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            var_b[1:2] = 3  # var_b is modified inplace before using it

<<<<<<< HEAD
            var_c = (
                var_b + var_b
            )  # Here, the grad op of sum doesn't use the value of var_b
=======
            var_c = var_b + var_b  # Here, the grad op of sum doesn't use the value of var_b
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            loss = var_c.sum()

            var_b[1:2] = 3  # var_b is modified inplace after using it

            loss.backward()

<<<<<<< HEAD

class TestDygraphInplace(unittest.TestCase):
=======
    def test_backward_success_2(self):
        with _test_eager_guard():
            self.func_test_backward_success_2()
        self.func_test_backward_success_2()


class TestDygraphInplace(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
    def test_inplace_api(self):
=======
    def func_test_inplace_api(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        inplace_var = self.inplace_api_processing(var)
        self.assertTrue(id(var) == id(inplace_var))

<<<<<<< HEAD
        inplace_var[0] = 2.0
        np.testing.assert_array_equal(var.numpy(), inplace_var.numpy())

    def test_forward_version(self):
=======
        inplace_var[0] = 2.
        np.testing.assert_array_equal(var.numpy(), inplace_var.numpy())

    def test_inplace_api(self):
        with _test_eager_guard():
            self.func_test_inplace_api()
        self.func_test_inplace_api()

    def func_test_forward_version(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

<<<<<<< HEAD
            inplace_var[0] = 2.0
=======
            inplace_var[0] = 2.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)

<<<<<<< HEAD
    def test_leaf_inplace_var_error(self):
=======
    def test_forward_version(self):
        with _test_eager_guard():
            self.func_test_forward_version()
        self.func_test_forward_version()

    def func_test_leaf_inplace_var_error(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var.stop_gradient = False

            def leaf_inplace_error():
                self.inplace_api_processing(var)

            self.assertRaises(ValueError, leaf_inplace_error)

<<<<<<< HEAD
    def test_backward_error(self):
=======
    def test_leaf_inplace_var_error(self):
        with _test_eager_guard():
            self.func_test_leaf_inplace_var_error()
        self.func_test_leaf_inplace_var_error()

    def func_test_backward_error(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
<<<<<<< HEAD
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:{} != wrapper_version_snapshot:{}".format(
                    1, 0
                ),
            ):
                loss.backward()

    def test_backward_success_1(self):
=======
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()

    def test_backward_error(self):
        with _test_eager_guard():
            self.func_test_backward_error()
        self.func_test_backward_error()

    def func_test_backward_success_1(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.inplace_api_processing(
<<<<<<< HEAD
                var_b
            )  # var_b is modified inplace before using it
=======
                var_b)  # var_b is modified inplace before using it
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            # Here, the gradient computation will use the value of var_b
            var_d = var_c**2
            loss = var_d.sum()
            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.non_inplace_api_processing(var_b)
            var_d = var_c**2
            loss = var_d.sum()
            loss.backward()
            grad_var_a = var_a.grad.numpy()

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

<<<<<<< HEAD
    def test_backward_success_2(self):
=======
    def test_backward_success_1(self):
        with _test_eager_guard():
            self.func_test_backward_success_1()
        self.func_test_backward_success_1()

    def func_test_backward_success_2(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.inplace_api_processing(
<<<<<<< HEAD
                var_b
            )  # var_b is modified inplace before using it

            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
=======
                var_b)  # var_b is modified inplace before using it

            var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            loss = var_d.sum()

            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.non_inplace_api_processing(var_b)

<<<<<<< HEAD
            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
=======
            var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            loss = var_d.sum()

            loss.backward()
            grad_var_a = var_a.grad.numpy()
        np.testing.assert_array_equal(grad_var_a_inplace, grad_var_a)

<<<<<<< HEAD

class TestDygraphInplaceUnsqueeze(TestDygraphInplace):
=======
    def test_backward_success_2(self):
        with _test_eager_guard():
            self.func_test_backward_success_2()
        self.func_test_backward_success_2()


class TestDygraphInplaceUnsqueeze(TestDygraphInplace):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.unsqueeze(var, -1)

    def inplace_api_processing(self, var):
        return paddle.unsqueeze_(var, -1)


class TestDygraphInplaceReshape(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.reshape(var, [-1])

    def inplace_api_processing(self, var):
        return paddle.reshape_(var, [-1])


class TestDygraphInplaceReshapeTensor(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        shape = paddle.to_tensor(-1)
        return paddle.reshape(var, shape)

    def inplace_api_processing(self, var):
        shape = paddle.to_tensor(-1)
        return paddle.reshape_(var, shape)


class TestDygraphInplaceFlatten(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.flatten()

    def inplace_api_processing(self, var):
        return var.flatten_()


class TestDygraphInplaceScatter(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data(self):
        self.input_var_numpy = np.array([[1, 1], [2, 2], [3, 3]])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
<<<<<<< HEAD
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
        )
=======
        updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]],
                                   dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return paddle.scatter(var, index, updates, overwrite=False)

    def inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
<<<<<<< HEAD
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
        )
=======
        updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]],
                                   dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return paddle.scatter_(var, index, updates, overwrite=False)


class TestDygraphInplaceElu(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.elu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.elu_(var)


class TestDygraphInplaceRelu(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.relu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.relu_(var)


class TestDygraphInplaceSoftmax(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.softmax(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_(var)


class TestDygraphInplaceTanh(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return paddle.tanh(var)

    def inplace_api_processing(self, var):
        return paddle.tanh_(var)


class TestDygraphInplaceCeil(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.ceil()

    def inplace_api_processing(self, var):
        return var.ceil_()


class TestDygraphInplaceFloor(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.floor()

    def inplace_api_processing(self, var):
        return var.floor_()


class TestDygraphInplaceExp(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def non_inplace_api_processing(self, var):
        return var.exp()

    def inplace_api_processing(self, var):
        return var.exp_()


class TestDygraphInplaceReciprocal(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.reciprocal()

    def inplace_api_processing(self, var):
        return var.reciprocal_()


class TestDygraphInplaceRound(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.round()

    def inplace_api_processing(self, var):
        return var.round_()


class TestDygraphInplaceSqrt(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data(self):
        self.input_var_numpy = np.random.uniform(0, 5, [10, 20, 1])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return var.sqrt()

    def inplace_api_processing(self, var):
        return var.sqrt_()


class TestDygraphInplaceRsqrt(TestDygraphInplaceSqrt):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.rsqrt()

    def inplace_api_processing(self, var):
        return var.rsqrt_()


class TestDygraphInplaceClip(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.clip(0.6, 1.5)

    def inplace_api_processing(self, var):
        return var.clip_(0.6, 1.5)


class TestDygraphInplaceScale(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        return var.scale(scale=2.0, bias=3.0)

    def inplace_api_processing(self, var):
        return var.scale_(scale=2.0, bias=3.0)


class TestDygraphInplaceAdd(TestDygraphInplace):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.subtract(input_var_2)

    def inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.subtract_(input_var_2)


class TestDygraphInplaceRemainder(TestDygraphInplaceAdd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def non_inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.remainder(input_var_2)

    def inplace_api_processing(self, var):
        input_var_2 = paddle.to_tensor(self.input_var_numpy_2)
        return var.remainder_(input_var_2)

    def test_leaf_inplace_var_error(self):
        pass

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass


class TestLossIsInplaceVar(unittest.TestCase):
<<<<<<< HEAD
    def test_loss_is_inplace_var(self):
=======

    def func_test_loss_is_inplace_var(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones((2, 2))
            var_a.stop_gradient = False

            var_b = var_a * 2
            loss = var_b.tanh_()

            loss.backward()
            inplace_grad_var_a = var_a.grad.numpy()

        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones((2, 2))
            var_a.stop_gradient = False

            var_b = var_a * 2
            loss = var_b.tanh()

            loss.backward()
            grad_var_a = var_a.grad.numpy()

        np.testing.assert_array_equal(inplace_grad_var_a, grad_var_a)

<<<<<<< HEAD

class TestContinuouslyInplace(unittest.TestCase):
    def test_continuously_inplace(self):
=======
    def test_loss_is_inplace_var(self):
        with _test_eager_guard():
            self.func_test_loss_is_inplace_var()
        self.func_test_loss_is_inplace_var()


class TestContinuouslyInplace(unittest.TestCase):

    def func_test_continuously_inplace(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        a = paddle.rand([2, 3])
        a.stop_gradient = False
        b = a * 2

        b.reshape_([-1])
        b.reshape_([2, 3])
        b.reshape_([-1])

        b.backward()

<<<<<<< HEAD

class TestGetitemBeforeInplace(unittest.TestCase):
    def test_getitem_before_inplace(self):
        a = paddle.ones(shape=[4, 2, 3], dtype="float32")
        a.stop_gradient = False
        b = a**2
        b[0] = 3
        # getitem has no_need_buffer input
        c = b[0:2]
        loss = c.sum()
        b[1] = 2
        loss.backward()
=======
    def test_continuously_inplace(self):
        with _test_eager_guard():
            self.func_test_continuously_inplace()
        self.func_test_continuously_inplace()


class TestGetitemBeforeInplace(unittest.TestCase):

    def test_getitem_before_inplace(self):
        with _test_eager_guard():
            a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            a.stop_gradient = False
            b = a**2
            b[0] = 3
            # getitem has no_need_buffer input
            c = b[0:2]
            loss = c.sum()
            b[1] = 2
            loss.backward()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
