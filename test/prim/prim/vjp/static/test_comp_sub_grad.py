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
import parameterized as param

import paddle
from paddle.base import core


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x, y):
        tmp = self.fc(x)
        out = paddle.subtract(tmp, y)
        return out


@param.parameterized_class(
    ('primal0', 'primal1', 'dtype'),
    [
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 3, 4),
            np.float32,
        ),
        (
            np.random.rand(2, 3, 3, 4),
            np.random.rand(3, 1, 4),
            np.float32,
        ),
        (
            np.random.rand(2, 3, 3, 4),
            np.random.rand(2, 3, 1, 4),
            np.float32,
        ),
        (np.random.rand(2, 3, 3, 4), np.random.rand(2, 3, 1, 4), np.float32),
        (
            np.random.rand(2, 1, 3, 4),
            np.random.rand(2, 3, 1, 4),
            np.float32,
        ),
        (
            np.random.rand(2, 3, 3, 4),
            np.random.rand(2, 1, 1, 4),
            np.float32,
        ),
    ],
)
class TestSubGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)

    def train(self, use_prim, use_cinn):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.y = paddle.randn([2, 4])
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x, self.y)
        res = paddle.autograd.grad(out, [self.x, self.y])

        return res

    def test_cinn(self):
        paddle.disable_static()
        dy_res = self.train(use_prim=False, use_cinn=False)
        comp_st_cinn_res = self.train(use_prim=True, use_cinn=True)

        for i in range(len(dy_res)):
            np.testing.assert_allclose(
                comp_st_cinn_res[i].numpy(),
                dy_res[i].numpy(),
                rtol=1e-7,
                atol=1e-7,
            )
        paddle.enable_static()

    def test_tanh_grad_comp(self):
        def actual(primal0, primal1):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal0', primal0.shape, primal0.dtype)
                y = paddle.static.data('primal1', primal1.shape, primal1.dtype)
                x.stop_gradient = False
                y.stop_gradient = False
                out = paddle.subtract(x, y)
                res = paddle.static.gradients([out], [x, y])
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                program=mp,
                feed={
                    'primal0': primal0,
                    'primal1': primal1,
                },
                fetch_list=[res[0].name, res[1].name],
            )
            return out[0], out[1]

        def desired(primal0, primal1):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data(
                    'primal0', self.primal0.shape, self.primal0.dtype
                )
                y = paddle.static.data(
                    'primal1', self.primal1.shape, self.primal1.dtype
                )
                x.stop_gradient = False
                y.stop_gradient = False
                out = paddle.subtract(x, y)
                res = paddle.static.gradients([out], [x, y])
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                program=mp,
                feed={
                    'primal0': self.primal0,
                    'primal1': self.primal1,
                },
                fetch_list=[res[0].name, res[1].name],
            )
            return out[0], out[1]

        dx, dy = actual(self.primal0, self.primal1)

        ddx, ddy = desired(self.primal0, self.primal1)

        np.testing.assert_allclose(
            actual=dx,
            desired=ddx,
            rtol=1e-6,
            atol=0,
        )
        np.testing.assert_allclose(
            actual=dy,
            desired=ddy,
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
