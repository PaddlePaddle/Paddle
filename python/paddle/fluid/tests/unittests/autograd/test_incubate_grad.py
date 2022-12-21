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

import typing
import unittest

import config
import numpy as np
import utils

import paddle

paddle.enable_static()


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'stop_gradient'),
    (
        (
            'func_out_two',
            utils.o2,
            (np.random.rand(10), np.random.rand(10)),
            (np.ones(10), np.zeros(1)),
            False,
        ),
    ),
)
class TestGradAux(unittest.TestCase):
    """In this example, such cases of incubate.autograd.grad will be executed:
    in static and dygraph cases, two usage types of api, check of aux parameter."""

    def setUp(self):
        self.dtype = (
            str(self.xs[0].dtype)
            if isinstance(self.xs, typing.Sequence)
            else str(self.xs.dtype)
        )
        self._rtol = (
            config.TOLERANCE.get(str(self.dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self.dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def _expected_vjp(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = utils.gen_static_data_and_feed(
                self.xs, self.v, False
            )
            ys = (
                self.fun(*static_xs)
                if isinstance(static_xs, typing.Sequence)
                else self.fun(static_xs)
            )
            xs_grads = paddle.static.gradients(ys, static_xs, static_v)
        exe.run(sp)
        return exe.run(mp, feed=feed, fetch_list=[ys, xs_grads])

    def _static_aux_grad(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, _ = utils.gen_static_data_and_feed(
                self.xs, self.v, False
            )
            ys = (
                self.fun(*static_xs)
                if isinstance(static_xs, typing.Sequence)
                else self.fun(static_xs)
            )
            xs_grads = paddle.incubate.autograd.grad(
                ys, static_xs, has_aux=True
            )
        exe.run(sp)
        res = exe.run(mp, feed=feed, fetch_list=xs_grads)
        return res

    def _static_aux_grad_return_func(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, _ = utils.gen_static_data_and_feed(
                self.xs, self.v, False
            )
            ys = (
                self.fun(*static_xs)
                if isinstance(static_xs, typing.Sequence)
                else self.fun(static_xs)
            )
            xs_grads = paddle.incubate.autograd.grad(ys, has_aux=True)(
                static_xs
            )
        exe.run(sp)
        res = exe.run(mp, feed=feed, fetch_list=xs_grads)
        return res

    def _dygraph_aux_grad(self):
        paddle.disable_static()
        x1 = paddle.to_tensor(self.xs[0])
        x1.stop_gradient = False
        x2 = paddle.to_tensor(self.xs[1])
        x2.stop_gradient = False
        ys = self.fun(x1, x2)
        res = paddle.incubate.autograd.grad(ys, (x1, x2), has_aux=True)
        paddle.enable_static()
        return res

    def _dygraph_aux_grad_return_func(self):
        paddle.disable_static()
        x1 = paddle.to_tensor(self.xs[0])
        x1.stop_gradient = False
        x2 = paddle.to_tensor(self.xs[1])
        x2.stop_gradient = False
        ys = self.fun(x1, x2)
        res = paddle.incubate.autograd.grad(ys, has_aux=True)((x1, x2))
        paddle.enable_static()
        return res

    def test_aux_grad(self):
        """test cases related with aux parameter in incubate.autograd.grad"""
        expected = self._expected_vjp()
        expected_gradients = expected[2:]
        expected_aux = expected[1]

        dygraph_aux_gradient = self._dygraph_aux_grad()
        actual_gradient = dygraph_aux_gradient[:2]
        actual_aux = dygraph_aux_gradient[2]

        np.testing.assert_allclose(
            expected_aux, actual_aux, rtol=self._rtol, atol=self._atol
        )
        for i in range(len(expected_gradients)):
            np.testing.assert_allclose(
                expected_gradients[i],
                actual_gradient[i],
                rtol=self._rtol,
                atol=self._atol,
            )

        # test static and dygraph mode
        static_aux_gradient = self._static_aux_grad()
        for i in range(len(static_aux_gradient)):
            np.testing.assert_allclose(
                static_aux_gradient[i],
                dygraph_aux_gradient[i],
                rtol=self._rtol,
                atol=self._atol,
            )

        # test such usage of api: grad(outputs, has_aux)(inputs)
        static_aux_gradient_return_func = self._static_aux_grad_return_func()
        dygraph_aux_gradient_return_func = self._dygraph_aux_grad_return_func()
        self.assertEqual(
            type(static_aux_gradient), type(static_aux_gradient_return_func)
        )
        self.assertEqual(
            type(static_aux_gradient_return_func),
            type(dygraph_aux_gradient_return_func),
        )
        for i in range(len(static_aux_gradient)):
            np.testing.assert_allclose(
                static_aux_gradient_return_func[i],
                dygraph_aux_gradient_return_func[i],
                rtol=self._rtol,
                atol=self._atol,
            )

    def test_wrong_input(self):
        """test wrong aux type and wrong output type in aux case"""
        with self.assertRaises(TypeError):
            x1 = paddle.to_tensor(self.xs[0])
            x1.stop_gradient = False
            x2 = paddle.to_tensor(self.xs[1])
            x2.stop_gradient = False
            ys = self.fun(x1, x2)
            paddle.incubate.autograd.grad(ys, (x1, x2), has_aux=ys[1])
        with self.assertRaises(TypeError):
            x1 = self.xs[0]
            x2 = self.xs[1]
            ys = (x1, x2)
            paddle.incubate.autograd.grad(ys, (x1, x2), has_aux=True)


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'stop_gradient'),
    (
        (
            'func_out_one',
            utils.mul,
            (np.random.rand(10), np.random.rand(10)),
            (np.ones(10), np.zeros(1)),
            False,
        ),
    ),
)
class TestGradRaiseError(unittest.TestCase):
    def setUp(self):
        self.dtype = (
            str(self.xs[0].dtype)
            if isinstance(self.xs, typing.Sequence)
            else str(self.xs.dtype)
        )

    def test_wrong_output(self):
        """test wrong case: single output tensor in aux case"""
        with self.assertRaises(TypeError):
            x1 = paddle.to_tensor(self.xs[0])
            x1.stop_gradient = False
            x2 = paddle.to_tensor(self.xs[1])
            x2.stop_gradient = False
            ys = self.fun(x1, x2)
            paddle.incubate.autograd.grad(ys, (x1, x2), has_aux=True)
