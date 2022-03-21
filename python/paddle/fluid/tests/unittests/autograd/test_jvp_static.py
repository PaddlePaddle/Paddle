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

import funcs
import config
import parameterize as param
import utils

paddle.enable_static()


@param.place(config.DEVICES)
@param.parameterize((param.TEST_CASE_NAME, 'fun', 'xs', 'v', 'stop_gradient'), (
    ('tensor-input', funcs.reduce, np.random.rand(2, 3), None, False),
    ('tensor-sequence-input', funcs.reduce, np.random.rand(2, 3), None, False),
    ('v-not-none', funcs.reduce, np.random.rand(2, 3), np.random.rand(1),
     False),
    ('stop_gradient', funcs.reduce, np.random.rand(2, 3), np.random.rand(1),
     True),
    ('mutmul', funcs.matmul, (np.random.rand(3, 2), np.random.rand(2, 3)), None,
     False),
    ('mul', funcs.mul, (np.random.rand(3, 3), np.random.rand(3, 3)), None,
     False),
    ('out-two', funcs.o2, (np.random.rand(10), np.random.rand(10)), None,
     False), ))
class TestVJP(unittest.TestCase):
    def setUp(self):
        self.dtype = str(self.xs[0].dtype) if isinstance(
            self.xs, typing.Sequence) else str(self.xs.dtype)

    def _vjp(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = gen_static_data_and_feed(
                self.xs, self.v, stop_gradient=self.stop_gradient)
            ys, xs_grads = paddle.autograd.vjp(self.fun, static_xs, static_v)
        exe.run(sp)
        return exe.run(mp, feed=feed, fetch_list=[ys, xs_grads])

    def _expected_vjp(self):
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = gen_static_data_and_feed(self.xs,
                                                                 self.v, False)
            ys = self.fun(*static_xs) if isinstance(
                static_xs, typing.Sequence) else self.fun(static_xs)
            xs_grads = paddle.static.gradients(ys, static_xs, static_v)
        exe.run(sp)
        return exe.run(mp, feed=feed, fetch_list=[ys, xs_grads])

    def test_vjp(self):
        actual = self._vjp()
        expected = self._expected_vjp()
        self.assertEqual(len(actual), len(expected))
        for i in range(len(actual)):
            np.testing.assert_allclose(
                actual[i],
                expected[i],
                rtol=config.RTOL[self.dtype],
                atol=config.ATOL[self.dtype])


@param.place(config.DEVICES)
@param.parameterize(
    (param.TEST_CASE_NAME, 'fun', 'xs', 'v', 'expected_exception'), (
        ('v-shape-not-equal-ys', funcs.square, np.random.rand(3),
         np.random.rand(1), ValueError), ))
class TestVJPException(unittest.TestCase):
    def setUp(self):
        self.exe = paddle.static.Executor()

    def _vjp(self):
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            feed, static_xs, static_v = gen_static_data_and_feed(self.xs,
                                                                 self.v)
            ys, xs_grads = paddle.autograd.vjp(self.fun, static_xs, static_v)
        self.exe.run(sp)
        return self.exe.run(mp, feed, fetch_list=[ys, xs_grads])

    def test_vjp(self):
        with self.assertRaises(self.expected_exception):
            self._vjp()


def gen_static_data_and_feed(xs, v, stop_gradient=True):
    feed = {}
    if isinstance(xs, typing.Sequence):
        static_xs = []
        for i, x in enumerate(xs):
            x = paddle.static.data(f"x{i}", x.shape, x.dtype)
            x.stop_gradient = stop_gradient
            static_xs.append(x)
        feed.update({f'x{idx}': value for idx, value in enumerate(xs)})
    else:
        static_xs = paddle.static.data('x', xs.shape, xs.dtype)
        static_xs.stop_gradient = stop_gradient
        feed.update({'x': xs})

    if isinstance(v, typing.Sequence):
        static_v = []
        for i, e in enumerate(v):
            e = paddle.static.data(f'v{idx}', v.shape, v.dtype)
            e.stop_gradient = stop_gradient
            static_v.append(e)
        feed.update({f'v{idx}': value for idx, value in v})
    elif v is not None:
        static_v = paddle.static.data('v', v.shape, v.dtype)
        static_v.stop_gradient = stop_gradient
        feed.update({'v': v})
    else:
        static_v = v

    return feed, static_xs, static_v,


if __name__ == '__main__':
    unittest.main()
