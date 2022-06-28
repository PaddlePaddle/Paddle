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


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'dtype'),
    (('matmul', paddle.matmul,
      (np.random.rand(2, 3), np.random.rand(3, 2)), None, 'float32'),
     ('multiply', paddle.multiply,
      (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float64'),
     ('add', paddle.add,
      (np.random.rand(2, 3), np.random.rand(2, 3)), None, 'float32'),
     ('input_not_sequence', paddle.tanh,
      (np.random.rand(5, 5), ), None, 'float64'),
     ('input_gradients_not_none', paddle.matmul,
      (np.random.rand(3, 3), np.random.rand(3, 3)),
      (np.random.rand(3, 3), np.random.rand(3, 3)), 'float64')))
class TestForwardGradients(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.xs = tuple(x.astype(cls.dtype) for x in cls.xs)
        cls._rtol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("rtol")
        cls._atol = config.TOLERANCE.get(str(
            cls.dtype)).get("first_order_grad").get("atol")

    def setUp(self):
        paddle.enable_static()
        paddle.incubate.autograd.enable_prim()

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
            paddle.incubate.autograd.enable_prim()
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
            out = exe.run(mp, feed=feed, fetch_list=ys_grad)
            paddle.incubate.autograd.disable_prim()
            return out

        actual = actual()
        expected = expected()
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_allclose(np.concatenate(actual),
                                   np.concatenate(expected),
                                   rtol=self._rtol,
                                   atol=self._atol)

    def test_prim_disabled(self):
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

    def test_illegal_param(self):
        paddle.incubate.autograd.enable_prim()
        with self.assertRaises(TypeError):
            primapi.forward_gradients(1, paddle.static.data('inputs',
                                                            shape=[1]))

        with self.assertRaises(TypeError):
            primapi.forward_gradients(paddle.static.data('targets', shape=[1]),
                                      1)
        paddle.incubate.autograd.disable_prim()


if __name__ == '__main__':
    unittest.main()
