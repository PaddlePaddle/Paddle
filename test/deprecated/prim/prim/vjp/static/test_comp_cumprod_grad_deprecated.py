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

from paddle.base import core

core._set_prim_backward_enabled(True)

import random

import numpy as np
import parameterized as param

import paddle


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        tmp = self.fc(x)
        out = paddle.cumprod(tmp, -1)
        return out


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (
            np.random.uniform(1, 5, (50,)),
            np.random.uniform(1, 5, (50,)),
            np.float32,
        ),
        (np.random.rand(10, 10), np.random.rand(10, 10), np.float32),
        (np.random.rand(3, 4, 5), np.random.rand(3, 4, 5), np.float32),
        (np.random.rand(2, 3, 4, 5), np.random.rand(2, 3, 4, 5), np.float32),
        (
            np.random.rand(2, 3, 2, 4, 5),
            np.random.rand(2, 3, 2, 4, 5),
            np.float32,
        ),
        (np.random.randint(1, 20, (10, 10)), np.random.rand(10, 10), np.int64),
    ],
)
class TestCumprodGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)
        cls.zero_nums = [0, 1, 10, int(np.prod(cls.primal.shape))]

    def train(self, use_prim, use_cinn):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x)
        res = paddle.autograd.grad(out, [self.x])

        return res

    def test_cumprod_grad_comp(self):
        paddle.enable_static()

        def actual(primal, cotangent, dim):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.cumprod(x, dim)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        def desired(primal, cotangent, dim):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.cumprod(x, dim)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        for zero_num in self.zero_nums:
            shape = self.primal.shape
            x = self.primal.flatten()
            indices = random.sample(range(x.size), zero_num)
            for i in indices:
                x[i] = 0
            x = np.reshape(x, shape)
            for i in range(len(self.primal.shape)):
                np.testing.assert_allclose(
                    actual=actual(x, self.cotangent, i),
                    desired=desired(x, self.cotangent, i),
                    rtol=1e-6,
                    atol=0,
                )
        core._set_prim_backward_enabled(False)
        paddle.disable_static()


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (
            np.random.uniform(1, 5, ()),
            np.random.uniform(1, 5, ()),
            np.float32,
        )
    ],
)
class TestCumprodGradComp0D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    def test_cumprod_grad_comp_0d(self):
        paddle.enable_static()

        def actual(primal, cotangent, dim):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.cumprod(x, dim)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        def desired(primal, cotangent, dim):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.cumprod(x, dim)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent, 0),
            desired=desired(self.primal, self.cotangent, 0),
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
