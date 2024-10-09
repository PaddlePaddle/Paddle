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

import functools
import unittest

import numpy as np

import paddle


class TestInplace(unittest.TestCase):
    def test_forward_version(self):
        with paddle.base.dygraph.guard():
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

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            var_b[1:2] = 3.3  # var_b is modified inplace after using it

            var_d = var_b**2

            loss = paddle.nn.functional.relu(var_c + var_d)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:1 != wrapper_version_snapshot:0",
            ):
                loss.backward()

    def test_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2
            var_b[1:2] = 3  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            loss = var_c.sum()
            loss.backward()

    def test_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            var_b[1:2] = 3  # var_b is modified inplace before using it

            var_c = (
                var_b + var_b
            )  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_c.sum()

            var_b[1:2] = 3  # var_b is modified inplace after using it

            loss.backward()


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
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        inplace_var = self.inplace_api_processing(var)
        self.assertTrue(id(var) == id(inplace_var))

        inplace_var[0] = 2
        np.testing.assert_array_equal(var.numpy(), inplace_var.numpy())

    def test_forward_result(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        no_inplace_var = self.non_inplace_api_processing(var)
        inplace_var = self.inplace_api_processing(var)
        np.testing.assert_array_equal(
            no_inplace_var.numpy(), inplace_var.numpy()
        )

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)

    def test_leaf_inplace_var_error(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var.stop_gradient = False

            def leaf_inplace_error():
                self.inplace_api_processing(var)

            self.assertRaises(ValueError, leaf_inplace_error)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)
            var_c = paddle.cast(var_c, "float32")

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:1 != wrapper_version_snapshot:0",
            ):
                loss.backward()

    def test_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.inplace_api_processing(
                var_b
            )  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_d = var_c**2
            var_d = paddle.cast(var_d, "float32")
            loss = var_d.sum()
            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.non_inplace_api_processing(var_b)
            var_d = var_c**2
            var_d = paddle.cast(var_d, "float32")
            loss = var_d.sum()
            loss.backward()
            grad_var_a = var_a.grad.numpy()
        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

    def test_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.inplace_api_processing(
                var_b
            )  # var_b is modified inplace before using it

            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
            var_d = paddle.cast(var_d, "float32")
            loss = var_d.sum()

            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.non_inplace_api_processing(var_b)

            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
            var_d = paddle.cast(var_d, "float32")
            loss = var_d.sum()

            loss.backward()
            grad_var_a = var_a.grad.numpy()
        np.testing.assert_array_equal(grad_var_a_inplace, grad_var_a)


class TestDygraphInplaceMaskedFill(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.masked_fill(var, self.mask, self.value)

    def inplace_api_processing(self, var):
        return paddle.masked_fill_(var, self.mask, self.value)

    def init_data(self):
        self.dtype = "float32"
        self.input_var_numpy = np.random.uniform(-5, 5, [30, 3])
        self.value = np.random.uniform(-10, 10)
        self.value = paddle.to_tensor(self.value, dtype=self.dtype)
        self.mask = np.random.randint(0, 2, [30, 3]).astype('bool')
        self.mask = paddle.to_tensor(self.mask, dtype='bool')

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 2)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 3)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 5)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                f"received tensor_version:{2} != wrapper_version_snapshot:{0}",
            ):
                loss.backward()


class TestDygraphInplaceMaskedFill2(TestDygraphInplaceMaskedFill):
    def non_inplace_api_processing(self, var):
        return paddle.masked_fill(var, self.mask, self.value)

    def inplace_api_processing(self, var):
        return paddle.masked_fill_(var, self.mask, self.value)

    def init_data(self):
        self.dtype = "float32"
        self.input_var_numpy = np.random.uniform(-5, 5, [30, 3])
        self.value = np.random.uniform(-10, 10)
        self.value = paddle.to_tensor(self.value, dtype=self.dtype)
        self.mask = np.random.randint(0, 2, [30, 1]).astype('bool')
        self.mask = paddle.to_tensor(self.mask, dtype='bool')


class TestDygraphInplaceMaskedScatter(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.masked_scatter(var, self.mask, self.value)

    def inplace_api_processing(self, var):
        return paddle.masked_scatter_(var, self.mask, self.value)

    def init_data(self):
        self.dtype = "float32"
        self.input_var_numpy = np.random.uniform(-5, 5, [30, 3])
        self.value = np.random.uniform(size=(30, 30))
        self.value = paddle.to_tensor(self.value, dtype=self.dtype)
        self.mask = np.random.randint(0, 2, [30, 1]).astype('bool')
        self.mask = paddle.to_tensor(self.mask, dtype='bool')


class TestDygraphInplaceWithContinuous(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"

    def set_np_compare_func(self):
        np_array_equal_with_nan = functools.partial(
            np.array_equal, equal_nan=True
        )
        self.np_compare = np_array_equal_with_nan

    def non_inplace_api_processing(self, var):
        return paddle.sin(var)

    def inplace_api_processing(self, var):
        return paddle.sin_(var)

    def test_continuous_inplace_backward(self):
        # The api that only relies on input to calculate the gradient will copy input before
        # the inplace calculation, so here supports continuous inplace backward calculation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.inplace_api_processing(var_b)
            var_d = self.inplace_api_processing(var_c)
            loss = var_d.sum()
            var_d = paddle.cast(var_d, "float32")
            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.non_inplace_api_processing(var_b)
            var_d = self.non_inplace_api_processing(var_c)
            var_d = paddle.cast(var_d, "float32")
            loss = var_d.sum()
            loss.backward()
            grad_var_a = var_a.grad.numpy()

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))


class TestDygraphInplaceCopysign(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.randn(10, 20)
        self.dtype = "float32"
        self.y = -3.0

    def inplace_api_processing(self, var):
        return paddle.copysign_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.copysign(var, self.y)

    def test_leaf_inplace_var_error(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var.stop_gradient = False
            self.y = paddle.rand([2, 10, 20])

            def leaf_inplace_error():
                self.inplace_api_processing(var)

            self.assertRaises(ValueError, leaf_inplace_error)


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


class TestDygraphInplaceReshapeTensor(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        shape = paddle.to_tensor([-1])
        return paddle.reshape(var, shape)

    def inplace_api_processing(self, var):
        shape = paddle.to_tensor([-1])
        return paddle.reshape_(var, shape)


class TestDygraphInplaceFlatten(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return var.flatten()

    def inplace_api_processing(self, var):
        return var.flatten_()


class TestDygraphInplaceFlattenStride(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.randn(2, 3, 2)
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return var.flatten(0, 1)

    def inplace_api_processing(self, var):
        return var.flatten_(0, 1)


class TestDygraphInplaceScatter(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.array([[1, 1], [2, 2], [3, 3]])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
        )

        return paddle.scatter(var, index, updates, overwrite=False)

    def inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
        )

        return paddle.scatter_(var, index, updates, overwrite=False)


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


class TestDygraphInplaceRemainder(TestDygraphInplaceAdd):
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
    def test_loss_is_inplace_var(self):
        with paddle.base.dygraph.guard():
            var_a = paddle.ones((2, 2))
            var_a.stop_gradient = False

            var_b = var_a * 2
            loss = var_b.tanh_()

            loss.backward()
            inplace_grad_var_a = var_a.grad.numpy()

        with paddle.base.dygraph.guard():
            var_a = paddle.ones((2, 2))
            var_a.stop_gradient = False

            var_b = var_a * 2
            loss = var_b.tanh()

            loss.backward()
            grad_var_a = var_a.grad.numpy()

        np.testing.assert_array_equal(inplace_grad_var_a, grad_var_a)


class TestContinuouslyInplace(unittest.TestCase):
    def test_continuously_inplace(self):
        a = paddle.rand([2, 3])
        a.stop_gradient = False
        b = a * 2

        b.reshape_([-1])
        b.reshape_([2, 3])
        b.reshape_([-1])

        b.backward()


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


class TestDygraphInplaceAsin(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.asin(var)

    def inplace_api_processing(self, var):
        return paddle.asin_(var)


class TestDygraphInplaceSinh(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.sinh(var)

    def inplace_api_processing(self, var):
        return paddle.sinh_(var)


class TestDygraphInplaceAsinh(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.asinh(var)

    def inplace_api_processing(self, var):
        return paddle.asinh_(var)


class TestDygraphInplaceAbs(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.abs(var)

    def inplace_api_processing(self, var):
        return paddle.abs_(var)


class TestDygraphInplaceCos(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.cos(var)

    def inplace_api_processing(self, var):
        return paddle.cos_(var)


class TestDygraphInplaceCosh(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.cosh(var)

    def inplace_api_processing(self, var):
        return paddle.cosh_(var)


class TestDygraphInplaceAcos(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.acos(var)

    def inplace_api_processing(self, var):
        return paddle.acos_(var)


class TestDygraphInplaceAcosh(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.acosh(var)

    def inplace_api_processing(self, var):
        return paddle.acosh_(var)


class TestDygraphInplaceTan(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.tan(var)

    def inplace_api_processing(self, var):
        return paddle.tan_(var)


class TestDygraphInplaceATan(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.atan(var)

    def inplace_api_processing(self, var):
        return paddle.atan_(var)


class TestDygraphInplaceATanh(TestDygraphInplaceWithContinuous):
    def non_inplace_api_processing(self, var):
        return paddle.atanh(var)

    def inplace_api_processing(self, var):
        return paddle.atanh_(var)


class TestDygraphInplaceAddMM(TestDygraphInplaceWithContinuous):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 10])
        self.dtype = "float32"
        self.x = paddle.randn([10, 10], dtype="float32")
        self.y = paddle.randn([10, 10], dtype="float32")

    def non_inplace_api_processing(self, var):
        return paddle.addmm(var, x=self.x, y=self.y)

    def inplace_api_processing(self, var):
        return paddle.addmm_(var, x=self.x, y=self.y)

    def test_errors(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        x1 = paddle.randn([10])
        self.assertRaises(ValueError, paddle.addmm_, var, x1, self.y)

        y1 = paddle.randn([12, 10])
        self.assertRaises(ValueError, paddle.addmm_, var, self.x, y1)
        x2 = paddle.randn([12, 10])
        self.assertRaises(ValueError, paddle.addmm_, var, x2, self.y)
        var1 = paddle.randn([1, 5])
        self.assertRaises(ValueError, paddle.addmm_, var1, x2, self.y)
        y2 = paddle.randn([10, 12])
        self.assertRaises(ValueError, paddle.addmm_, var, self.x, y2)
        var2 = paddle.randn([6])
        self.assertRaises(ValueError, paddle.addmm_, var2, self.x, self.y)
        var3 = paddle.randn([2, 3, 4])
        self.assertRaises(ValueError, paddle.addmm_, var3, self.x, self.y)


class TestDygraphInplacePowerScalar(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.pow_(var, 2)

    def non_inplace_api_processing(self, var):
        return paddle.pow(var, 2)

    def test_type_error(self):
        var = paddle.to_tensor(self.input_var_numpy, dtype=self.dtype)
        with self.assertRaisesRegex(
            TypeError,
            f'y must be scalar type, but received: {type([2])} ',
        ):
            paddle.pow_(var, [2])


class TestDygraphInplaceTriu(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.triu_(var, 0)

    def non_inplace_api_processing(self, var):
        return paddle.triu(var, 0)


class TestDygraphInplaceTril(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.tril_(var, 0)

    def non_inplace_api_processing(self, var):
        return paddle.tril(var, 0)


class TestDygraphInplaceLogit(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.logit_(var, 1e-3)

    def non_inplace_api_processing(self, var):
        return paddle.logit(var, 1e-3)


class TestDygraphInplaceLog(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.log_(var)

    def non_inplace_api_processing(self, var):
        return paddle.log(var)


class TestDygraphInplaceLog2(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.log2_(var)

    def non_inplace_api_processing(self, var):
        return paddle.log2(var)


class TestDygraphInplaceLog10(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.log10_(var)

    def non_inplace_api_processing(self, var):
        return paddle.log10(var)


class TestDygraphInplaceLog1p(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.log1p_(var)

    def non_inplace_api_processing(self, var):
        return paddle.log1p(var)


class TestDygraphInplaceTrunc(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.trunc_(var)

    def non_inplace_api_processing(self, var):
        return paddle.trunc(var)


class TestDygraphInplaceDigamma(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.digamma_(var)

    def non_inplace_api_processing(self, var):
        return paddle.digamma(var)


class TestDygraphInplaceMutilgammaln(TestDygraphInplaceWithContinuous):
    def init_data(self):
        self.input_var_numpy = np.random.rand(10, 20).astype('float32') + 1.0
        self.dtype = "float32"
        self.p = 2

    def inplace_api_processing(self, var):
        return paddle.multigammaln_(var, self.p)

    def non_inplace_api_processing(self, var):
        return paddle.multigammaln(var, self.p)

    def test_leaf_inplace_var_error(self):
        pass


class TestDygraphInplaceGammaln(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.gammaln_(var)

    def non_inplace_api_processing(self, var):
        return paddle.gammaln(var)


class TestDygraphInplaceNeg(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.neg_(var)

    def non_inplace_api_processing(self, var):
        return paddle.neg(var)


class TestDygraphInplaceGammaincc(TestDygraphInplace):
    def init_data(self):
        self.shape = (3, 40)
        self.dtype = "float32"
        self.input_var_numpy = (
            np.random.random(self.shape).astype(self.dtype) + 1
        )
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype) + 1

    def inplace_api_processing(self, var):
        return paddle.gammaincc_(var, y=self.y)

    def non_inplace_api_processing(self, var):
        return paddle.gammaincc(var, y=self.y)

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass


class TestDygraphInplaceGammainc(TestDygraphInplace):
    def init_data(self):
        self.shape = (3, 40)
        self.dtype = "float32"
        self.input_var_numpy = (
            np.random.random(self.shape).astype(self.dtype) + 1
        )
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype) + 1

    def inplace_api_processing(self, var):
        return paddle.gammainc_(var, y=self.y)

    def non_inplace_api_processing(self, var):
        return paddle.gammainc(var, y=self.y)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 3)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 4)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 7)

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass


class TestDygraphInplaceLgamma(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.lgamma_(var)

    def non_inplace_api_processing(self, var):
        return paddle.lgamma(var)


class TestDygraphInplaceFrac(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.frac_(var)

    def non_inplace_api_processing(self, var):
        return paddle.frac(var)


class TestDygraphInplaceI0(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.i0_(var)

    def non_inplace_api_processing(self, var):
        return paddle.i0(var)


class TestDygraphInplaceGcd(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.randint(2, size=200)
        self.input_var_numpy = self.input_var_numpy.reshape([10, 20])
        self.dtype = "int32"
        self.y = paddle.randint(low=-5, high=5, shape=[10, 20], dtype="int32")

    def inplace_api_processing(self, var):
        return paddle.gcd_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.gcd(var, self.y)

    def test_forward_version(self):
        pass

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass

    def test_error(self):
        x = paddle.randn([1, 10])
        y = paddle.randn([20, 1])
        self.assertRaises(ValueError, paddle.gcd_, x, y)


class TestDygraphInplaceHypot(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.randint(2, size=200)
        self.input_var_numpy = self.input_var_numpy.reshape([10, 20])
        self.dtype = "float32"
        self.y = paddle.randn(shape=[10, 20], dtype="float32")

    def inplace_api_processing(self, var):
        return paddle.hypot_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.hypot(var, self.y)

    def test_errors(self):
        x = 3.0
        self.assertRaises(TypeError, paddle.hypot_, x, self.y)
        self.assertRaises(TypeError, paddle.hypot_, self.y, x)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 3)

            inplace_var[0] = 2.0
            self.assertEqual(var.inplace_version, 4)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 7)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)
            var_c = paddle.cast(var_c, "float32")

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                f"received tensor_version:{3} != wrapper_version_snapshot:{0}",
            ):
                loss.backward()


class TestDygraphInplaceNanToNum(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.array(
            ["0.12334", "inf", "inf", "-inf", "nan", "-0.12342", "1.123", "nan"]
        )
        self.input_var_numpy = self.input_var_numpy.reshape([2, 4]).astype(
            np.float32
        )
        self.dtype = "float32"

    def inplace_api_processing(self, var):
        return paddle.nan_to_num_(var)

    def non_inplace_api_processing(self, var):
        return paddle.nan_to_num(var)

    def set_np_compare_func(self):
        np_array_equal_with_nan = functools.partial(
            np.array_equal, equal_nan=True
        )
        self.np_compare = np_array_equal_with_nan

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 3)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 4)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 7)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:3 != wrapper_version_snapshot:0",
            ):
                loss.backward()


class TestDygraphInplaceLcm(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.randint(2, size=200)
        self.input_var_numpy = self.input_var_numpy.reshape([10, 20])
        self.dtype = "int32"
        self.y = paddle.randint(low=-5, high=5, shape=[10, 20], dtype="int32")

    def inplace_api_processing(self, var):
        return paddle.lcm_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.lcm(var, self.y)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 4)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 5)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 9)

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass

    def test_leaf_inplace_var_error(self):
        pass


class TestDygraphInplaceLdexp(TestDygraphInplaceWithContinuous):
    def init_data(self):
        super().init_data()
        self.y = paddle.to_tensor([2], dtype="int32")

    def inplace_api_processing(self, var):
        return paddle.ldexp_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.ldexp(var, self.y)

    def test_leaf_inplace_var_error(self):
        pass

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 2)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 3)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 5)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype("float64")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:2 != wrapper_version_snapshot:0",
            ):
                loss.backward()

    def test_error(self):
        x = 1
        x_normal = paddle.randn([3, 4])
        y = 1
        with self.assertRaisesRegex(
            TypeError, f"x must be tensor type, but got {type(x)}"
        ):
            paddle.ldexp_(x, y)
        with self.assertRaisesRegex(
            TypeError, f"y must be tensor type, but got {type(y)}"
        ):
            paddle.ldexp_(x_normal, y)


class TestDygraphInplaceWhere(TestDygraphInplaceWithContinuous):
    def init_data(self):
        super().init_data()
        self.y = paddle.randn([10, 20, 1])

    def inplace_api_processing(self, var):
        return paddle.where_(var > self.y, var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.where(var > self.y, var, self.y)

    def test_error(self):
        cond = paddle.to_tensor([[False, True, False], [False, True, False]])
        self.assertRaises(ValueError, paddle.where_, cond, 1, 2)
        self.assertRaises(ValueError, paddle.where_, cond, None, None)


class TestDygraphInplaceWhereBroadcast(TestDygraphInplaceWithContinuous):
    def init_data(self):
        super().init_data()
        self.y = paddle.randn([10, 1, 1])

    def inplace_api_processing(self, var):
        return paddle.where_(var > self.y, var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.where(var > self.y, var, self.y)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 2)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 3)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 5)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:2 != wrapper_version_snapshot:0",
            ):
                loss.backward()


class TestDygraphInplacePolygamma(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.polygamma_(var, 1)

    def non_inplace_api_processing(self, var):
        return paddle.polygamma(var, 1)


class TestDygraphInplaceHardTanh(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.nn.functional.hardtanh_(var, -1.0, 1.0)

    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.hardtanh(var, -1.0, 1.0)


class TestDygraphInplaceLeakyRelu(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.nn.functional.leaky_relu_(var, 0.01)

    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.leaky_relu(var, 0.01)


class TestDygraphInplaceThresholdedRelu(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.nn.functional.thresholded_relu_(var, 1.0)

    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.thresholded_relu(var, 1.0)


class TestDygraphInplaceLogicAnd(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"
        self.y = paddle.randn([10, 20, 1])

    def test_forward_result(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        no_inplace_var = self.non_inplace_api_processing(var)
        inplace_var = self.inplace_api_processing(var)
        np.testing.assert_array_equal(
            no_inplace_var.numpy(), inplace_var.numpy()
        )

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)
            inplace_var[0] = True
            self.assertEqual(var.inplace_version, 2)

    def inplace_api_processing(self, var):
        return paddle.logical_and_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.logical_and(var, self.y)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([10, 1, 20])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass

    def test_leaf_inplace_var_error(self):
        pass


class TestDygraphInplaceLogicOr(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.logical_or_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.logical_or(var, self.y)


class TestDygraphInplaceLogicXor(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.logical_xor_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.logical_xor(var, self.y)


class TestDygraphInplaceLogicNot(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.logical_not_(var)

    def non_inplace_api_processing(self, var):
        return paddle.logical_not(var)

    def test_broadcast_error(self):
        pass


class TestDygraphInplaceLessThan(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.less_than_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.less_than(var, self.y)


class TestDygraphInplaceLessEqual(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.less_equal_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.less_equal(var, self.y)


class TestDygraphInplaceGreaterEqual(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.greater_equal_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.greater_equal(var, self.y)


class TestDygraphInplaceGreaterThan(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.greater_than_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.greater_than(var, self.y)


class TestDygraphInplaceEqual(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.equal_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.equal(var, self.y)


class TestDygraphInplaceNotEqual(TestDygraphInplaceLogicAnd):
    def inplace_api_processing(self, var):
        return paddle.not_equal_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.not_equal(var, self.y)


class TestDygraphInplacBitwiseAnd(TestDygraphInplaceLogicAnd):
    def init_data(self):
        self.input_var_numpy = paddle.randint(
            low=0, high=10, shape=[3, 4, 1], dtype="int32"
        )
        self.dtype = "int32"
        self.y = paddle.randint(low=0, high=10, shape=[3, 4, 1], dtype="int32")

    def inplace_api_processing(self, var):
        return paddle.bitwise_and_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_and(var, self.y)

    def test_forward_result(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        no_inplace_var = self.non_inplace_api_processing(var)
        inplace_var = self.inplace_api_processing(var)
        np.testing.assert_array_equal(
            no_inplace_var.numpy(), inplace_var.numpy()
        )

    def test_broadcast_error(self):
        broadcast_input = paddle.randint(
            low=0, high=10, shape=[3, 1, 4], dtype="int32"
        )
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplacBitwisOr(TestDygraphInplacBitwiseAnd):
    def inplace_api_processing(self, var):
        return paddle.bitwise_or_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_or(var, self.y)


class TestDygraphInplacBitwisXor(TestDygraphInplacBitwiseAnd):
    def inplace_api_processing(self, var):
        return paddle.bitwise_xor_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_xor(var, self.y)


class TestDygraphInplacBitwisNot(TestDygraphInplacBitwiseAnd):
    def inplace_api_processing(self, var):
        return paddle.bitwise_not_(var)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_not(var)

    def test_broadcast_error(self):
        pass


class TestDygraphInplaceDivide(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"
        self.y = paddle.randn([10, 20, 1])

    def inplace_api_processing(self, var):
        return paddle.divide_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.divide(var, self.y)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([10, 1, 20])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplaceCast(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.cast_(var, "float64")

    def non_inplace_api_processing(self, var):
        return paddle.cast(var, "float64")


class TestDygraphInplaceFloorDivide(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"
        self.y = paddle.randn([10, 20, 1])

    def inplace_api_processing(self, var):
        return paddle.floor_divide_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.floor_divide(var, self.y)

    def test_backward_error(self):
        pass

    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass

    def test_leaf_inplace_var_error(self):
        pass

    def test_error(self):
        x = paddle.randn([1, 10])
        y = paddle.randn([20, 1])
        self.assertRaises(ValueError, paddle.floor_divide_, x, y)


class TestDygraphInplaceCumsum(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.cumsum_(var, dtype="float32")

    def non_inplace_api_processing(self, var):
        return paddle.cumsum(var, dtype="float32")

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype("float64")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            paddle.cumsum_(var_b, -1, dtype="float32")

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:2 != wrapper_version_snapshot:0",
            ):
                loss.backward()

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)


class TestDygraphInplaceCumprod(TestDygraphInplace):
    def inplace_api_processing(self, var):
        return paddle.cumprod_(var, -1, dtype="float32")

    def non_inplace_api_processing(self, var):
        return paddle.cumprod(var, -1, dtype="float32")

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            paddle.cumprod_(var_b, -1, dtype="float64")

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:2 != wrapper_version_snapshot:0",
            ):
                loss.backward()

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)


class TestDygrapInplaceRenorm(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.renorm_(var, 1.0, -1, 2.05)

    def non_inplace_api_processing(self, var):
        return paddle.renorm(var, 1.0, -1, 2.05)

    def test_error(self):
        var = paddle.randn([3, 4, 1])
        self.assertRaises(ValueError, paddle.renorm_, var, 1.0, 5, 2.05)
        self.assertRaises(ValueError, paddle.renorm_, var, 1.0, -5, 2.05)


class TestDygrapInplaceMultiply(TestDygraphInplaceWithContinuous):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"
        self.y = paddle.randn([10, 20, 1])

    def inplace_api_processing(self, var):
        return paddle.multiply_(var, self.y)

    def non_inplace_api_processing(self, var):
        return paddle.multiply(var, self.y)


class TestDygrapInplaceT(TestDygraphInplaceWithContinuous):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20])
        self.dtype = "float32"

    def inplace_api_processing(self, var):
        return paddle.t_(var)

    def non_inplace_api_processing(self, var):
        return paddle.t(var)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)


class TestDygrapInplaceTranspose(TestDygraphInplaceWithContinuous):
    def inplace_api_processing(self, var):
        return paddle.transpose_(var, [1, 0, 2])

    def non_inplace_api_processing(self, var):
        return paddle.transpose(var, [1, 0, 2])

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)


class TestDygraphInplaceBitwiseLeftShift_arithmetic(TestDygraphInplaceLogicAnd):
    def init_data(self):
        self.input_var_numpy = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.input_var_numpy = paddle.to_tensor(self.input_var_numpy)
        self.dtype = "int32"
        self.y = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.y = paddle.to_tensor(self.y)
        self.is_arithmetic = True

    def inplace_api_processing(self, var):
        return paddle.bitwise_left_shift_(var, self.y, self.is_arithmetic)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_left_shift(var, self.y, self.is_arithmetic)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([4, 5])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplaceBitwiseRightShift_arithmetic(
    TestDygraphInplaceLogicAnd
):
    def init_data(self):
        self.input_var_numpy = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.input_var_numpy = paddle.to_tensor(self.input_var_numpy)
        self.dtype = "int32"
        self.y = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.y = paddle.to_tensor(self.y)
        self.is_arithmetic = True

    def inplace_api_processing(self, var):
        return paddle.bitwise_right_shift_(var, self.y, self.is_arithmetic)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_right_shift_(var, self.y, self.is_arithmetic)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([4, 5])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplaceBitwiseLeftShift_logic(TestDygraphInplaceLogicAnd):
    def init_data(self):
        self.input_var_numpy = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.input_var_numpy = paddle.to_tensor(self.input_var_numpy)
        self.dtype = "int32"
        self.y = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.y = paddle.to_tensor(self.y)
        self.is_arithmetic = False

    def inplace_api_processing(self, var):
        return paddle.bitwise_left_shift_(var, self.y, self.is_arithmetic)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_left_shift(var, self.y, self.is_arithmetic)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([4, 5])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplaceBitwiseRightShift_logic(TestDygraphInplaceLogicAnd):
    def init_data(self):
        self.input_var_numpy = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.input_var_numpy = paddle.to_tensor(self.input_var_numpy)
        self.dtype = "int32"
        self.y = np.random.randint(
            low=-(2**31), high=2**31, size=[3, 4, 5], dtype="int32"
        )
        self.y = paddle.to_tensor(self.y)
        self.is_arithmetic = False

    def inplace_api_processing(self, var):
        return paddle.bitwise_right_shift_(var, self.y, self.is_arithmetic)

    def non_inplace_api_processing(self, var):
        return paddle.bitwise_right_shift_(var, self.y, self.is_arithmetic)

    def test_broadcast_error(self):
        broadcast_input = paddle.randn([4, 5])
        with self.assertRaises(ValueError):
            self.inplace_api_processing(broadcast_input)


class TestDygraphInplaceIndexFill(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.random((20, 40))
        self.dtype = "float32"
        self.axis = 1
        self.index = paddle.to_tensor([0, 2])
        self.value = -1

    def inplace_api_processing(self, var):
        return paddle.index_fill_(var, self.index, self.axis, self.value)

    def non_inplace_api_processing(self, var):
        return paddle.index_fill(var, self.index, self.axis, self.value)

    def test_forward_version(self):
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 3)

            inplace_var[0] = 2
            self.assertEqual(var.inplace_version, 4)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 7)

    def test_backward_error(self):
        with paddle.base.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = var_b**2
            self.inplace_api_processing(var_b)
            var_c = paddle.cast(var_c, "float32")

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                f"received tensor_version:{3} != wrapper_version_snapshot:{0}",
            ):
                loss.backward()


class TestDygraphTensorApplyInplace(unittest.TestCase):
    def setUp(self):
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def non_inplace_api_processing(self, var, f):
        return var.apply(f)

    def inplace_api_processing(self, var, f):
        return var.apply_(f)

    def test_inplace_api(self):
        var = paddle.to_tensor(self.input_var_numpy, stop_gradient=True).astype(
            self.dtype
        )
        f = lambda x: 3 * x + 2
        non_inplace_var = self.non_inplace_api_processing(var, f)
        inplace_var = self.inplace_api_processing(var, f)
        self.assertTrue(id(var) == id(inplace_var))
        np.testing.assert_array_equal(
            non_inplace_var.numpy(), inplace_var.numpy()
        )


class TestDygraphInplaceBernoulli(unittest.TestCase):
    def setUp(self):
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.shape = (100, 1000)
        self.input_var_numpy = np.random.random(self.shape)
        self.dtype = "float32"
        self.p = 0.5

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def inplace_api_processing(self, var):
        return paddle.bernoulli_(var, p=self.p)

    def inplace_class_method_processing(self, var):
        return var.bernoulli_(self.p)

    def non_inplace_api_processing(self):
        return paddle.bernoulli(paddle.full(self.shape, self.p))

    def test_inplace_api(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        non_inplace_var = self.non_inplace_api_processing()
        inplace_var = self.inplace_api_processing(var)
        self.assertTrue(id(var) == id(inplace_var))
        np.testing.assert_allclose(
            non_inplace_var.numpy().mean(),
            inplace_var.numpy().mean(),
            atol=0.01,
        )
        np.testing.assert_allclose(
            non_inplace_var.numpy().var(), inplace_var.numpy().var(), atol=0.01
        )

    def test_inplace_api_backward(self):
        var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        var_a.stop_gradient = False
        var_b = var_a.clone()
        expected_gradient = np.zeros(self.shape)
        inplace_var = self.inplace_api_processing(var_b)
        inplace_var.backward()
        np.testing.assert_equal(
            var_a.grad.numpy(),
            expected_gradient,
        )

    def test_inplace_class_method(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        non_inplace_var = self.non_inplace_api_processing()
        inplace_var = self.inplace_class_method_processing(var)
        self.assertTrue(id(var) == id(inplace_var))
        np.testing.assert_allclose(
            non_inplace_var.numpy().mean(),
            inplace_var.numpy().mean(),
            atol=0.01,
        )
        np.testing.assert_allclose(
            non_inplace_var.numpy().var(), inplace_var.numpy().var(), atol=0.01
        )

    def test_inplace_class_method_backward(self):
        var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        var_a.stop_gradient = False
        var_b = var_a.clone()
        expected_gradient = np.zeros(self.shape)
        inplace_var = self.inplace_class_method_processing(var_b)
        inplace_var.backward()
        np.testing.assert_equal(
            var_a.grad.numpy(),
            expected_gradient,
        )


class TestDygraphInplaceBernoulli2(TestDygraphInplaceBernoulli):
    def init_data(self):
        self.shape = (100, 1000)
        self.input_var_numpy = np.random.random(self.shape)
        self.dtype = "float64"
        self.p = 0.5


class TestDygraphInplaceBernoulliError(unittest.TestCase):
    def test_broadcast_error(self):
        var = paddle.randn([3, 4])
        p = paddle.randn([5])
        with self.assertRaises(ValueError):
            var.bernoulli_(p)


if __name__ == '__main__':
    unittest.main()
