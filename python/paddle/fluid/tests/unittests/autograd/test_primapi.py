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

import numpy as np
import paddle
from paddle.incubate.autograd import primapi

import config
import utils


class TestGradients(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_third_order(self):
        paddle.incubate.autograd.enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)

            grad1, = paddle.static.gradients([x4], [x])
            grad2, = paddle.static.gradients([grad1], [x])
            grad3, = paddle.static.gradients([grad2], [x])

            paddle.incubate.autograd.prim2orig(main.block(0))

        feed = {x.name: np.array([2.]).astype('float32')}
        fetch_list = [grad3.name]
        result = [np.array([48.])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        paddle.incubate.autograd.disable_prim()

    def test_fourth_order(self):
        paddle.incubate.autograd.enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)
            x5 = paddle.multiply(x4, x)
            out = paddle.sqrt(x5 + x4)

            grad1, = paddle.static.gradients([out], [x])
            grad2, = paddle.static.gradients([grad1], [x])
            grad3, = paddle.static.gradients([grad2], [x])
            grad4, = paddle.static.gradients([grad3], [x])

            paddle.incubate.autograd.prim2orig(main.block(0))

        feed = {x.name: np.array([2.]).astype('float32'), }
        fetch_list = [grad4.name]
        # (3*(-5*x^2-16*x-16))/(16*(x+1)^3.5)
        result = [np.array([-0.27263762711])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        paddle.incubate.autograd.disable_prim()


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'xs', 'v'), (
    ('matmul', paddle.matmul, (np.random.rand(2, 3), np.random.rand(3, 2)),
     None), ('multiply', paddle.multiply,
             (np.random.rand(2, 3), np.random.rand(2, 3)), None),
    ('add', paddle.add, (np.random.rand(2, 3), np.random.rand(2, 3)),
     None), ('input_not_sequence', paddle.tanh, np.random.rand(5, 5), None),
    ('input_gradients_not_none', paddle.matmul,
     (np.random.rand(3, 3), np.random.rand(3, 3)),
     (np.random.rand(3, 3), np.random.rand(3, 3)))))
class TestForwardGradients(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()
        self.dtype = str(self.xs[0].dtype) if isinstance(
            self.xs, typing.Sequence) else str(self.xs.dtype)
        self._rtol = config.TOLERANCE.get(str(self.dtype)).get(
            "first_order_grad").get("rtol")
        self._atol = config.TOLERANCE.get(str(self.dtype)).get(
            "first_order_grad").get("atol")

    def tearDown(self):
        paddle.incubate.autograd.disable_prim()
        paddle.disable_static()

    def test_forward_gradients(self):
        def expected():
            paddle.incubate.autograd.disable_prim()
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                _, ys_grad = paddle.autograd.jvp(self.fun, static_xs, static_v)
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.enable_prim()
            return out

        def actual():
            sp = paddle.static.Program()
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = primapi.forward_gradients(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(mp, feed=feed, fetch_list=ys_grad)

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_allclose(
            np.concatenate(actual), np.concatenate(expected))

        paddle.incubate.autograd.disable_prim()

    def test_disable_prim(self):
        paddle.incubate.autograd.disable_prim()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with self.assertRaises(RuntimeError):
            with paddle.static.program_guard(mp, sp):
                feed, static_xs, static_v = utils.gen_static_data_and_feed(
                    self.xs, self.v, stop_gradient=False)
                ys = self.fun(*static_xs) if isinstance(
                    static_xs, typing.Sequence) else self.fun(static_xs)
                ys_grad = primapi.forward_gradients(ys, static_xs, static_v)
                paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            exe.run(mp, feed=feed, fetch_list=ys_grad)
        paddle.incubate.autograd.enable_prim()


if __name__ == '__main__':
    unittest.main()
