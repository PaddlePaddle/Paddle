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
import numpy as np

import paddle
from paddle.autograd.primops import (
    neg, set_value, add, sub, mul, div, sqrt, tanh, reshape, broadcast,
    transpose, split, concat, reduce, matmul, slice_select, slice_assign,
    gather, scatter_add, fill_const)
from paddle.autograd.primx import Transform, topo_path, orig2prim, prim2orig
from paddle.autograd.primx import _gradients
from paddle.autograd.new_adam_optimizer import AdamOptimizer


def prog1(x, y):
    t = paddle.matmul(x, y)
    # z = paddle.sum(paddle.sqrt(x))
    return t


class TestPyPrimOps(unittest.TestCase):
    """ Test Python wrappers of primitive ops. """

    def setUp(self):
        paddle.enable_static()

    def test_ops(self):
        A = np.random.rand(1)
        B = np.random.rand(2)
        C = np.random.rand(2, 3)
        D = np.random.rand(2, 3)
        E = np.random.rand(3, 2)

        a = paddle.static.data(name='A', shape=A.shape, dtype='float32')
        b = paddle.static.data(name='B', shape=B.shape, dtype='float32')
        c = paddle.static.data(name='C', shape=C.shape, dtype='float32')
        d = paddle.static.data(name='D', shape=D.shape, dtype='float32')
        e = paddle.static.data(name='E', shape=E.shape, dtype='float32')

        add_1 = add(a, a)
        self.assertEqual(add_1.dtype, a.dtype)
        self.assertEqual(add_1.shape, a.shape)

        add_2 = add(c, d)
        self.assertEqual(add_2.dtype, c.dtype)
        self.assertEqual(add_2.shape, c.shape)

        sub_1 = sub(c, d)
        self.assertEqual(sub_1.dtype, c.dtype)
        self.assertEqual(sub_1.shape, c.shape)

        mul_1 = mul(c, d)
        self.assertEqual(mul_1.dtype, c.dtype)
        self.assertEqual(mul_1.shape, c.shape)

        div_1 = div(c, d)
        self.assertEqual(div_1.dtype, c.dtype)
        self.assertEqual(div_1.shape, c.shape)

        sqrt_1 = sqrt(b)
        self.assertEqual(sqrt_1.dtype, b.dtype)
        self.assertEqual(sqrt_1.shape, b.shape)

        tanh_1 = tanh(d)
        self.assertEqual(tanh_1.dtype, d.dtype)
        self.assertEqual(tanh_1.shape, d.shape)

        reshape_1 = reshape(c, d.shape)
        self.assertEqual(reshape_1.dtype, c.dtype)
        self.assertEqual(reshape_1.shape, d.shape)

        broadcast_1 = broadcast(b, e.shape)
        self.assertEqual(broadcast_1.dtype, b.dtype)
        self.assertEqual(broadcast_1.shape, e.shape)

        transpose_1 = transpose(c, axis=[1, 0])
        self.assertEqual(transpose_1.dtype, c.dtype)
        self.assertEqual(transpose_1.shape, e.shape)

        split_1_0, split_1_1 = split(c, num_or_sections=[1, 2], axis=1)
        self.assertEqual(split_1_0.dtype, c.dtype)
        self.assertEqual(split_1_0.shape, (2, 1))
        self.assertEqual(split_1_1.shape, (2, 2))

        concat_1 = concat([c, d], axis=0)
        self.assertEqual(concat_1.dtype, c.dtype)
        self.assertEqual(concat_1.shape, (4, 3))

        reduce_1 = reduce(d, axis=[1])
        self.assertEqual(reduce_1.dtype, d.dtype)
        self.assertEqual(reduce_1.shape, (2, ))

        reduce_2 = reduce(c, axis=[0, 1])
        self.assertEqual(reduce_2.dtype, c.dtype)
        self.assertEqual(reduce_2.shape, (1, ))
        # TODO: reduce + keepdim

        matmul_1 = matmul(d, e)
        self.assertEqual(matmul_1.dtype, d.dtype)
        self.assertEqual(matmul_1.shape, (2, 2))

        slice_select_1 = slice_select(
            e, axis=[0], starts=[0], ends=[2], strides=[1])
        self.assertEqual(slice_select_1.dtype, e.dtype)
        self.assertEqual(slice_select_1.shape, (2, 2))

        slice_select_2 = slice_select(
            d, axis=[0, 1], starts=[0, 1], ends=[2, 3], strides=[1, 2])
        self.assertEqual(slice_select_2.dtype, d.dtype)
        self.assertEqual(slice_select_2.shape, (2, 1))

        y = broadcast(b, [2, 2])
        slice_assign_1 = slice_assign(
            d, y, axis=[1], starts=[1], ends=[3], strides=[1])
        self.assertEqual(slice_assign_1.dtype, d.dtype)
        self.assertEqual(slice_assign_1.shape, d.shape)

        index = paddle.static.data('index', shape=[5], dtype='int32')
        gather_1 = gather(e, index, axis=0)
        self.assertEqual(gather_1.dtype, e.dtype)
        self.assertEqual(gather_1.shape, (5, 2))

        y = paddle.rand([5, 2], dtype='float32')
        scatter_add_1 = scatter_add(e, y, index, axis=0)
        self.assertEqual(scatter_add_1.dtype, e.dtype)
        self.assertEqual(scatter_add_1.shape, e.shape)

        fill_const_1 = fill_const(value=10, shape=a.shape, dtype=a.dtype)
        self.assertEqual(fill_const_1.shape, a.shape)
        self.assertEqual(fill_const_1.dtype, a.dtype)

        neg_1 = neg(x=b)
        self.assertEqual(neg_1.shape, b.shape)
        self.assertEqual(neg_1.dtype, b.dtype)

        set_value_1 = set_value(
            d, a, axis=[1], starts=[1], ends=[3], strides=[1], out=d)
        self.assertEqual(set_value_1.shape, d.shape)
        self.assertEqual(set_value_1.dtype, d.dtype)

    def test_vjp_set1(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            X = paddle.static.data('Input', shape=[100, 2], dtype='float32')
            W = paddle.static.data('Weight', shape=[5, 2], dtype='float32')
            T = concat([X, W], axis=0)
            Z = reduce(T, [0, 1])
            ad = Transform(X.block)
            xs_dot, ys_dot = ad.linearize([X, W], [Z])
            ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot)
            assert xs_bar[0].shape == X.shape
            assert xs_bar[1].shape == W.shape

            print(f'-------test_vjp_set1-------')
            for op in topo_path(ys_bar, xs_bar):
                print(op)

    def test_vjp_set2(self):
        X = paddle.static.data('Input', shape=[100, 2], dtype='float32')
        W = paddle.static.data('Weight', shape=[5, 2], dtype='float32')
        act = tanh
        W_ = broadcast(W, shape=[100, 5, 2])
        X_ = reshape(X, shape=[100, 2, 1])
        Z = tanh(matmul(W_, X_))
        Y = reduce(Z, axis=[1, 2])

        def loss(y, x):
            ad = Transform(y.block)
            xs_dot, ys_dot = ad.linearize([x], [y])
            ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot)
            # ad = Transform(y.block)
            # xs_dot, ys_dot = ad.linearize([x], xs_bar)
            # for op in topo_path(xs_dot, ys_dot):
            # print(op)
            # ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot)
            return ys_bar, xs_bar

        vs, grads = loss(Y, W)
        assert grads[0].shape == W.shape

        print(f'-------test_vjp_set2-------')
        for op in topo_path(vs, grads):
            print(op)

    def test_first_order_gradients(self):
        x = np.random.rand(100, 1, 2)
        y = np.random.rand(100, 2, 5)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            X = paddle.static.data('X', shape=[100, 1, 2], dtype='float32')
            Y = paddle.static.data('Y', shape=[100, 2, 5], dtype='float32')
            Z = prog1(X, Y)
            X_grad, W_grad = _gradients([Z], [X, Y])
        # exe = paddle.static.Executor()
        # exe.run(startup)
        # z = exe.run(main, feed={'X': x, 'Y': y}, fetch_list=[Z])
        # print(z)

    def test_lower(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data('x', shape=[2, 20], dtype='float32')
            w = paddle.static.data('w', shape=[20, 2], dtype='float32')
            bias = paddle.static.data('bias', shape=[2], dtype='float32')
            y = paddle.tanh(paddle.matmul(x, w) + bias)
            print(f'-------test_orig2prim: orig-------')
            print(x.block)

            orig2prim(x.block)

            print(f'-------test_orig2prim: prim-------')
            print(x.block)

            prim2orig(x.block)

            print(f'-------test_orig2prim: orig-------')
            print(x.block)

    def test_minimize(self):
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data('x', shape=[2, 20], dtype='float32')
            x.stop_gradient = False
            w = paddle.static.create_parameter(
                shape=[20, 2], dtype='float32', is_bias=False)
            bias = paddle.static.create_parameter(
                shape=[2], dtype='float32', is_bias=True)
            y = paddle.tanh(paddle.matmul(x, w) + bias)
            loss = paddle.norm(y, p=2)
            opt = AdamOptimizer(0.01)
            opt.minimize(loss)
            # prim2orig(x.block, update_var_list=[loss])

            print(f'-------test_minimize: orig-------')
            print(x.block)
        # exe.run(startup)
        # for i in range(10):
        #     y_d = exe.run(main, feed = {'x': np.rand((2, 20))}, fetch_list=[y.name])


if __name__ == '__main__':
    unittest.main()
