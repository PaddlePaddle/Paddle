# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import TestCase

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.wrapped_decorator import wrap_decorator
from paddle.vision.models import resnet50, resnet101


def _dygraph_guard_(func):
    def __impl__(*args, **kwargs):
        if fluid._non_static_mode():
            return func(*args, **kwargs)
        else:
            with fluid.dygraph.guard():
                return func(*args, **kwargs)

    return __impl__


dygraph_guard = wrap_decorator(_dygraph_guard_)


def random_var(size, low=-1, high=1, dtype='float32'):
    x_np = np.random.uniform(low=low, high=high, size=size).astype(dtype)
    return fluid.dygraph.to_variable(x_np)


class TestEagerGrad(TestCase):
    def func_simple_example_eager_grad(self):
        np.random.seed(2021)
        paddle.set_device('cpu')
        np_x = np.random.random((3, 3))
        np_y = np.random.random((3, 1))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64", stop_gradient=False)
        out = paddle.matmul(x, y)
        dx = fluid.dygraph.grad(out, x)

        dout = np.ones_like(np_y)
        expected_dx = np.matmul(dout, np.transpose(np_y))

        # stop_gradient = !create_graph, create_graph default false
        self.assertEqual(dx[0].stop_gradient, True)
        np.testing.assert_allclose(dx[0].numpy(), expected_dx, rtol=1e-05)

    def test_simple_example_eager_grad(self):
        with _test_eager_guard():
            self.func_simple_example_eager_grad()
        self.func_simple_example_eager_grad()

    def func_simple_example_eager_grad_allow_unused(self):
        np.random.seed(2021)
        paddle.set_device('cpu')
        np_x = np.random.random((3, 3))
        np_y = np.random.random((3, 1))
        np_z = np.random.random((3, 1))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64", stop_gradient=False)
        z = paddle.to_tensor(np_z, dtype="float64", stop_gradient=False)
        out_z = paddle.nn.functional.sigmoid(z)
        out = paddle.matmul(x, y)

        dx = fluid.dygraph.grad(out, [x, z], allow_unused=True)
        dout = np.ones_like(np_y)
        expected_dx = np.matmul(dout, np.transpose(np_y))
        np.testing.assert_allclose(dx[0].numpy(), expected_dx, rtol=1e-05)
        # stop_gradient = !create_graph, create_graph default false
        self.assertEqual(dx[0].stop_gradient, True)
        # x is unused input in the graph
        self.assertIsNone(dx[1])

    def test_simple_example_eager_grad_allow_unused(self):
        with _test_eager_guard():
            self.func_simple_example_eager_grad_allow_unused()
        self.func_simple_example_eager_grad_allow_unused()

    def func_simple_example_eager_grad_not_allow_unused(self):
        np.random.seed(2021)
        paddle.set_device('cpu')
        np_x = np.random.random((3, 3))
        np_y = np.random.random((3, 1))
        np_z = np.random.random((3, 1))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64", stop_gradient=False)
        z = paddle.to_tensor(np_z, dtype="float64", stop_gradient=False)
        out_z = paddle.nn.functional.sigmoid(z)
        out = paddle.matmul(x, y)

        try:
            # allow_unused is false in default
            dx = fluid.dygraph.grad(out, [x, z])
        except ValueError as e:
            error_msg = str(e)
            assert error_msg.find("allow_unused") > 0

    def test_simple_example_eager_grad_not_allow_unused(self):
        with _test_eager_guard():
            self.func_simple_example_eager_grad_not_allow_unused()
        self.func_simple_example_eager_grad_not_allow_unused()

    def func_simple_example_eager_grad_duplicate_input(self):
        np.random.seed(2021)
        paddle.set_device('cpu')
        np_x = np.random.random((3, 3))
        np_y = np.random.random((3, 1))
        np_z = np.random.random((3, 1))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64", stop_gradient=False)
        z = paddle.to_tensor(np_z, dtype="float64", stop_gradient=False)
        out_z = paddle.nn.functional.sigmoid(z)
        out = paddle.matmul(x, y)

        try:
            # duplicate input will arise RuntimeError errors
            dx = fluid.dygraph.grad(out, [x, x])
        except RuntimeError as e:
            error_msg = str(e)
            assert error_msg.find("duplicate") > 0

    def test_simple_example_eager_grad_duplicate_input(self):
        with _test_eager_guard():
            self.func_simple_example_eager_grad_duplicate_input()
        self.func_simple_example_eager_grad_duplicate_input()

    def func_simple_example_eager_grad_duplicate_output(self):
        np.random.seed(2021)
        paddle.set_device('cpu')
        np_x = np.random.random((3, 3))
        np_y = np.random.random((3, 1))
        np_z = np.random.random((3, 1))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64", stop_gradient=False)
        z = paddle.to_tensor(np_z, dtype="float64", stop_gradient=False)
        out_z = paddle.nn.functional.sigmoid(z)
        out = paddle.matmul(x, y)

        try:
            # duplicate output will arise RuntimeError errors
            dx = fluid.dygraph.grad([out, out], [x])
        except RuntimeError as e:
            error_msg = str(e)
            assert error_msg.find("duplicate") > 0

    def test_simple_example_eager_grad_duplicate_output(self):
        with _test_eager_guard():
            self.func_simple_example_eager_grad_duplicate_output()
        self.func_simple_example_eager_grad_duplicate_output()

    def test_simple_example_eager_two_grad_output(self):
        with _test_eager_guard():
            x1 = paddle.to_tensor([1.0, 2.0])
            x1.stop_gradient = False
            x2 = paddle.to_tensor([1.0, 2.0])
            x2.stop_gradient = False
            out1 = x1 * 2
            out2 = x2 * 2

            dout2_record_by_hook = []

            def record_hook(grad):
                dout2_record_by_hook.append(grad)

            out2.register_hook(record_hook)

            out3 = paddle.multiply(out1, out2)
            out4 = paddle.mean(out3)
            egr_dout2, egr_dout3 = paddle.grad([out4], [out2, out3])

            np.testing.assert_array_equal(
                dout2_record_by_hook[0].numpy(), np.array([1.0, 2.0])
            )

        x1 = paddle.to_tensor([1.0, 2.0])
        x1.stop_gradient = False
        x2 = paddle.to_tensor([1.0, 2.0])
        x2.stop_gradient = False
        out1 = x1 * 2
        out2 = x2 * 2

        out3 = paddle.multiply(out1, out2)
        out4 = paddle.mean(out3)
        dout2, dout3 = paddle.grad([out4], [out2, out3])

        self.assertEqual(dout2.stop_gradient, egr_dout2.stop_gradient)
        self.assertEqual(dout3.stop_gradient, egr_dout3.stop_gradient)
        np.testing.assert_array_equal(dout2.numpy(), egr_dout2.numpy())
        np.testing.assert_array_equal(dout3.numpy(), egr_dout3.numpy())


class TestDygraphDoubleGrad(TestCase):
    def setUp(self):
        self.sort_sum_gradient = False
        self.shape = [5, 10]

    def grad(
        self,
        outputs,
        inputs,
        grad_outputs=None,
        no_grad_vars=None,
        retain_graph=None,
        create_graph=False,
        allow_unused=False,
    ):
        fluid.set_flags({'FLAGS_sort_sum_gradient': self.sort_sum_gradient})
        return fluid.dygraph.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            no_grad_vars=no_grad_vars,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

    @dygraph_guard
    def func_exception(self):
        with self.assertRaises(AssertionError):
            self.grad(None, None)

        shape = self.shape

        with self.assertRaises(AssertionError):
            self.grad(1, random_var(shape))

        with self.assertRaises(AssertionError):
            self.grad(random_var(shape), 1)

        with self.assertRaises(AssertionError):
            self.grad([1], [random_var(shape)])

        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [1])

        with self.assertRaises(AssertionError):
            self.grad(
                [random_var(shape), random_var(shape)],
                [random_var(shape)],
                [random_var(shape)],
            )

        with self.assertRaises(AssertionError):
            self.grad(
                [random_var(shape)], [random_var(shape)], no_grad_vars=[1]
            )

        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [random_var(shape)], no_grad_vars=1)

    def test_exception(self):
        with _test_eager_guard():
            self.func_exception()
        self.func_exception()

    @dygraph_guard
    def func_simple_example(self):
        x = random_var(self.shape)
        x.stop_gradient = False
        y = x + 1

        for create_graph in [False, True]:
            (dx,) = self.grad(
                [x], [x], create_graph=create_graph, retain_graph=True
            )
            self.assertEqual(dx.shape, x.shape)
            self.assertTrue(np.all(dx.numpy() == 1))
            self.assertNotEqual(dx.stop_gradient, create_graph)

            (dx_mul_2,) = self.grad(
                [y, x], [x], create_graph=create_graph, retain_graph=True
            )
            self.assertEqual(dx_mul_2.shape, x.shape)
            self.assertTrue(np.all(dx_mul_2.numpy() == 2))
            self.assertNotEqual(dx_mul_2.stop_gradient, create_graph)

            (none_grad,) = self.grad(
                [x], [y], create_graph=create_graph, allow_unused=True
            )
            self.assertIsNone(none_grad)

            (grad_with_none_and_not_none,) = self.grad(
                [x, y], [y], create_graph=create_graph
            )
            self.assertTrue(grad_with_none_and_not_none.shape, x.shape)
            self.assertTrue(np.all(grad_with_none_and_not_none.numpy() == 1))
            self.assertNotEqual(
                grad_with_none_and_not_none.stop_gradient, create_graph
            )

    def test_simple_example(self):
        with _test_eager_guard():
            self.func_simple_example()
        self.func_simple_example()

    @dygraph_guard
    def func_example_no_grad_vars(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y1 = F.relu(x)
        y2 = F.relu(x)
        z = y1 + y2
        w = z * z

        w_mean = paddle.mean(w)
        del y1, z, w

        (dx_actual,) = self.grad(
            [w_mean], [x], create_graph=True, no_grad_vars=[y2]
        )

        self.assertFalse(y2.stop_gradient)
        self.assertFalse(dx_actual.stop_gradient)

        dx_expected = (
            1.0
            / float(numel)
            * (np.maximum(x_np, 0) + y2.numpy())
            * (x_np > 0)
            * 2
        ).astype('float32')

        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

    def test_example_no_grad_vars(self):
        with _test_eager_guard():
            self.func_example_no_grad_vars()
        self.func_example_no_grad_vars()

    @dygraph_guard
    def func_none_one_initial_gradient(self):
        numel = 1
        for s in self.shape:
            numel *= s

        half_numel = int(numel / 2)
        half_x_positive = np.random.uniform(low=1, high=2, size=[half_numel])
        half_x_negative = np.random.uniform(
            low=-2, high=-1, size=[numel - half_numel]
        )
        x_np = np.array(list(half_x_positive) + list(half_x_negative)).astype(
            'float32'
        )
        np.random.shuffle(x_np)

        x = fluid.dygraph.to_variable(x_np)
        x.stop_gradient = False

        alpha = 0.2
        y = paddle.nn.functional.leaky_relu(x, alpha)
        y = y * y
        z = y * y

        x_np = x.numpy()
        relu_x_np = np.maximum(x_np, alpha * x_np).astype('float32')
        relu_x_grad_np = ((x_np > 0) + (x_np < 0) * alpha).astype('float32')
        dy_expected = (relu_x_np * relu_x_grad_np * 2).astype('float32')
        dz_expected = (np.power(relu_x_np, 3) * relu_x_grad_np * 4).astype(
            'float32'
        )

        random_grad_y = random_var(y.shape, low=1, high=2)
        random_grad_z = random_var(z.shape, low=1, high=2)
        ones_grad_y = np.ones(y.shape).astype('float32')
        ones_grad_z = np.ones(z.shape).astype('float32')

        original_random_grad_y = random_grad_y.numpy()
        original_random_grad_z = random_grad_z.numpy()

        for grad_y in [random_grad_y]:
            for grad_z in [random_grad_z]:
                for create_graph in [False, True]:
                    (dx_actual,) = self.grad(
                        outputs=[y, z],
                        inputs=[x],
                        grad_outputs=[grad_y, grad_z],
                        create_graph=create_graph,
                        retain_graph=True,
                    )

                    grad_y_np = (
                        ones_grad_y if grad_y is None else grad_y.numpy()
                    )
                    grad_z_np = (
                        ones_grad_z if grad_z is None else grad_z.numpy()
                    )

                    dx_expected = (
                        dy_expected * grad_y_np + dz_expected * grad_z_np
                    )
                    np.testing.assert_allclose(
                        dx_actual.numpy(), dx_expected, rtol=1e-05
                    )

                    if grad_y is not None:
                        self.assertTrue(grad_y.stop_gradient)
                        np.testing.assert_array_equal(
                            grad_y.numpy(), original_random_grad_y
                        )

                    if grad_z is not None:
                        self.assertTrue(grad_z.stop_gradient)
                        np.testing.assert_array_equal(
                            grad_z.numpy(), original_random_grad_z
                        )

    def test_none_one_initial_gradient(self):
        with _test_eager_guard():
            self.func_none_one_initial_gradient()
        self.func_none_one_initial_gradient()

    @dygraph_guard
    def func_example_with_gradient_accumulation_and_create_graph(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y = F.relu(x)
        z = y + 1
        w = z * z

        w_mean = paddle.mean(w)
        del y, z, w

        (dx_actual,) = self.grad([w_mean], [x], create_graph=True)
        del w_mean

        self.assertFalse(dx_actual.stop_gradient)

        # Theoritical result based on math calculation
        dx_expected = (
            1.0 / float(numel) * (np.maximum(x_np, 0) + 1) * (x_np > 0) * 2
        ).astype('float32')
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

        loss = paddle.mean(dx_actual * dx_actual + x * x)
        loss.backward(retain_graph=True)

        x_grad_actual = x.gradient()
        x_grad_expected = (
            2.0
            / float(numel)
            * (x_np + dx_expected * (x_np > 0) * 2 / float(numel))
        ).astype('float32')
        np.testing.assert_allclose(x_grad_actual, x_grad_expected, rtol=1e-05)

        for i in range(5):
            loss.backward(retain_graph=True)
            x_grad_actual = x.gradient()
            x_grad_expected = (i + 2) * (
                2.0
                / float(numel)
                * (x_np + dx_expected * (x_np > 0) * 2 / float(numel))
            ).astype('float32')
            np.testing.assert_allclose(
                x_grad_actual, x_grad_expected, rtol=1e-05
            )

    def test_example_with_gradient_accumulation_and_create_graph(self):
        with _test_eager_guard():
            self.func_example_with_gradient_accumulation_and_create_graph()
        self.func_example_with_gradient_accumulation_and_create_graph()

    @dygraph_guard
    def func_example_with_gradient_accumulation_and_no_grad_vars(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y1 = F.relu(x)
        y2 = F.relu(x)
        z = y1 + y2
        w = z * z

        w_mean = paddle.mean(w)
        del y1, z, w

        (dx_actual,) = self.grad(
            [w_mean],
            [x],
            retain_graph=True,
            create_graph=True,
            no_grad_vars=[y2],
        )

        self.assertFalse(y2.stop_gradient)
        self.assertFalse(dx_actual.stop_gradient)

        dx_expected = (
            1.0
            / float(numel)
            * (np.maximum(x_np, 0) + y2.numpy())
            * (x_np > 0)
            * 2
        ).astype('float32')
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

        loss = paddle.mean(dx_actual * dx_actual + x * x)
        loss.backward()

        x_grad_actual = x.gradient()
        x_grad_expected = (
            2.0
            / float(numel)
            * (x_np + dx_expected * (x_np > 0) * 4 / float(numel))
        ).astype('float32')
        np.testing.assert_allclose(x_grad_actual, x_grad_expected, rtol=1e-05)

    def test_example_with_gradient_accumulation_and_no_grad_vars(self):
        with _test_eager_guard():
            self.func_example_with_gradient_accumulation_and_no_grad_vars()
        self.func_example_with_gradient_accumulation_and_no_grad_vars()

    @dygraph_guard
    def func_example_with_gradient_accumulation_and_not_create_graph(self):
        x = random_var(self.shape)
        x_np = x.numpy()
        numel = x_np.size
        x.stop_gradient = False

        y = F.relu(x)
        z = y + 1
        w = z * z

        w_mean = paddle.mean(w)
        del y, z, w

        (dx_actual,) = self.grad([w_mean], [x], create_graph=False)
        del w_mean

        self.assertTrue(dx_actual.stop_gradient)

        dx_expected = (
            1.0 / float(numel) * (np.maximum(x_np, 0) + 1) * (x_np > 0) * 2
        ).astype('float32')

        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)

        loss = paddle.mean(dx_actual * dx_actual + x * x)
        loss.backward()

        x_grad_actual = x.gradient()
        x_grad_expected = (2.0 * x_np / float(numel)).astype('float32')
        np.testing.assert_allclose(x_grad_actual, x_grad_expected, rtol=1e-05)

    def test_example_with_gradient_accumulation_and_not_create_graph(self):
        with _test_eager_guard():
            self.func_example_with_gradient_accumulation_and_not_create_graph()
        self.func_example_with_gradient_accumulation_and_not_create_graph()


class TestDygraphDoubleGradSortGradient(TestDygraphDoubleGrad):
    def setUp(self):
        self.sort_sum_gradient = True
        self.shape = [5, 10]


class TestDygraphDoubleGradVisitedUniq(TestCase):
    def func_compare(self):
        value = (
            np.random.uniform(-0.5, 0.5, 100)
            .reshape(10, 2, 5)
            .astype("float32")
        )

        def model_f(input):
            linear = paddle.nn.Linear(5, 3)
            for i in range(10):
                if i == 0:
                    out = linear(input)
                else:
                    out = out + linear(input)
            return out

        fluid.set_flags({'FLAGS_sort_sum_gradient': True})

        with fluid.dygraph.guard():
            paddle.seed(123)
            paddle.framework.random._manual_program_seed(123)
            a = fluid.dygraph.to_variable(value)
            a.stop_gradient = False

            out = model_f(a)

            dx = fluid.dygraph.grad(
                outputs=[out],
                inputs=[a],
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )

            grad_1 = dx[0].numpy()

        with fluid.dygraph.guard():
            paddle.seed(123)
            paddle.framework.random._manual_program_seed(123)
            a = fluid.dygraph.to_variable(value)
            a.stop_gradient = False

            out = model_f(a)
            out.backward()

            grad_2 = a.gradient()

        np.testing.assert_array_equal(grad_1, grad_2)

    def test_compare(self):
        with _test_eager_guard():
            self.func_compare()
        self.func_compare()


class TestRaiseNoDoubleGradOp(TestCase):
    def raise_no_grad_op(self):
        with fluid.dygraph.guard():
            x = paddle.ones(shape=[2, 3, 2, 2], dtype='float32')
            x.stop_gradient = False
            y = paddle.static.nn.group_norm(x, groups=1)

            dx = fluid.dygraph.grad(
                outputs=[y], inputs=[x], create_graph=True, retain_graph=True
            )[0]

            loss = paddle.mean(dx)
            loss.backward()

    def test_raise(self):
        self.assertRaises(RuntimeError, self.raise_no_grad_op)


class TestDoubleGradResNet(TestCase):
    def setUp(self):
        paddle.seed(123)
        paddle.framework.random._manual_program_seed(123)
        self.data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    @dygraph_guard
    def test_resnet_resnet50(self):
        with _test_eager_guard():
            model = resnet50(pretrained=False)
            egr_data = paddle.to_tensor(self.data)
            egr_data.stop_gradient = False
            egr_out = model(egr_data)
            egr_preds = paddle.argmax(egr_out, axis=1)
            egr_label_onehot = paddle.nn.functional.one_hot(
                paddle.to_tensor(egr_preds), num_classes=egr_out.shape[1]
            )
            egr_target = paddle.sum(egr_out * egr_label_onehot, axis=1)

            egr_g = paddle.grad(outputs=egr_target, inputs=egr_out)[0]
            egr_g_numpy = egr_g.numpy()
            self.assertEqual(list(egr_g_numpy.shape), list(egr_out.shape))

        model = resnet50(pretrained=False)
        data = paddle.to_tensor(self.data)
        data.stop_gradient = False
        out = model(data)
        preds = paddle.argmax(out, axis=1)
        label_onehot = paddle.nn.functional.one_hot(
            paddle.to_tensor(preds), num_classes=out.shape[1]
        )
        target = paddle.sum(out * label_onehot, axis=1)

        g = paddle.grad(outputs=target, inputs=out)[0]
        g_numpy = g.numpy()
        self.assertEqual(list(g_numpy.shape), list(out.shape))

        np.testing.assert_array_equal(egr_out, out)
        np.testing.assert_array_equal(egr_g_numpy, g_numpy)

    @dygraph_guard
    def test_resnet_resnet101(self):
        with _test_eager_guard():
            model = resnet101(pretrained=False)
            egr_data = paddle.to_tensor(self.data)
            egr_data.stop_gradient = False
            egr_out = model(egr_data)
            egr_preds = paddle.argmax(egr_out, axis=1)
            egr_label_onehot = paddle.nn.functional.one_hot(
                paddle.to_tensor(egr_preds), num_classes=egr_out.shape[1]
            )
            egr_target = paddle.sum(egr_out * egr_label_onehot, axis=1)

            egr_g = paddle.grad(outputs=egr_target, inputs=egr_out)[0]
            egr_g_numpy = egr_g.numpy()
            self.assertEqual(list(egr_g_numpy.shape), list(egr_out.shape))

        model = resnet101(pretrained=False)
        data = paddle.to_tensor(self.data)
        data.stop_gradient = False
        out = model(data)
        preds = paddle.argmax(out, axis=1)
        label_onehot = paddle.nn.functional.one_hot(
            paddle.to_tensor(preds), num_classes=out.shape[1]
        )
        target = paddle.sum(out * label_onehot, axis=1)

        g = paddle.grad(outputs=target, inputs=out)[0]
        g_numpy = g.numpy()
        self.assertEqual(list(g_numpy.shape), list(out.shape))

        np.testing.assert_array_equal(egr_out, out)
        np.testing.assert_array_equal(egr_g_numpy, g_numpy)


class TestDoubleGradBasics(TestCase):
    def test_matmul(self):
        input_numpy = np.ones([3, 3]) * 2
        with _test_eager_guard():
            x = paddle.to_tensor(
                input_numpy, stop_gradient=False, dtype='float32'
            )
            y = paddle.to_tensor(
                input_numpy, stop_gradient=False, dtype='float32'
            )
            grad_out = paddle.to_tensor(
                np.ones([3, 3]), stop_gradient=False, dtype='float32'
            )

            out = paddle.matmul(x, y, False, False)
            new_x_g, new_y_g = paddle.grad(
                [out], [x, y], [grad_out], retain_graph=True, create_graph=True
            )
            new_x_g.backward()

            out_ref = np.ones([3, 3]) * 12.0
            np.testing.assert_array_equal(out.numpy(), out_ref)

            new_x_g_ref = np.ones([3, 3]) * 6.0
            new_y_g_ref = np.ones([3, 3]) * 6.0
            np.testing.assert_array_equal(new_x_g.numpy(), new_x_g_ref)
            np.testing.assert_array_equal(new_y_g.numpy(), new_y_g_ref)

            x_grad_ref = np.ones([3, 3]) * 0.0
            np.testing.assert_array_equal(x.grad.numpy(), x_grad_ref)

            y_grad_ref = np.ones([3, 3]) * 3.0
            np.testing.assert_array_equal(y.grad.numpy(), y_grad_ref)

            grad_out_grad_ref = np.ones([3, 3]) * 6.0
            np.testing.assert_array_equal(
                grad_out.grad.numpy(), grad_out_grad_ref
            )


if __name__ == '__main__':
    unittest.main()
