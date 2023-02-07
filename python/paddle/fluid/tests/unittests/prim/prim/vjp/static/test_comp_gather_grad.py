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
from paddle.fluid import core


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super(PrimeNet, self).__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x, index, axis):
        tmp = self.fc(x)
        out = paddle.gather(tmp, index, axis)
        return out


@param.parameterized_class(
    ('primal0', 'index', 'axis', 'x_dtype', 'index_dtype', 'v'),
    [
        (
            np.random.rand(100),
            np.array([1, 3, 5]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3),
        ),
        (
            np.random.rand(10, 20),
            np.array([1, 3, 5]),
            0,
            np.float64,
            np.int64,
            np.random.rand(3, 20),
        ),
        (
            np.random.rand(10, 20),
            np.array([1, 1, 3]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3, 20),
        ),
        (
            np.random.rand(3, 88, 30),
            np.array([1, 3, 5]),
            1,
            np.float32,
            np.int32,
            np.random.rand(3, 3, 30),
        ),
        (
            np.random.rand(10, 88, 10),
            np.array([1, 3, 5]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3, 88, 10),
        ),
    ],
)
class TestGatherGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.x_dtype)
        cls.index = cls.index.astype(cls.index_dtype)
        cls.v = cls.v.astype(cls.x_dtype)

    def train(self, use_prim, use_cinn):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.index = paddle.to_tensor(np.array([0, 1]))
        self.x.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x, self.index, 0)
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
                rtol=1e-6,
                atol=1e-6,
            )
        paddle.enable_static()

    def test_tanh_grad_comp(self):
        paddle.enable_static()

        def actual(primal0, index, axis, v):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal0', primal0.shape, primal0.dtype)
                index_tmp = paddle.static.data(
                    'index', index.shape, index.dtype
                )
                x.stop_gradient = False
                index_tmp.stop_gradient = True
                z = paddle.gather(x, index_tmp, axis)
                z_grad = paddle.static.data('v', z.shape, z.dtype)
                res = paddle.static.gradients([z], [x], [z_grad])
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                program=mp,
                feed={
                    'primal0': primal0,
                    'index': index,
                    'v': v,
                },
                fetch_list=[res[0].name],
            )
            return out[0]

        def desired(primal0, index, axis, v):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal0', primal0.shape, primal0.dtype)
                index_tmp = paddle.static.data(
                    'index', index.shape, index.dtype
                )
                x.stop_gradient = False
                index_tmp.stop_gradient = True
                z = paddle.gather(x, index_tmp, axis)
                z_grad = paddle.static.data('v', z.shape, z.dtype)
                res = paddle.static.gradients([z], [x], [z_grad])
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                program=mp,
                feed={
                    'primal0': primal0,
                    'index': index,
                    'v': v,
                },
                fetch_list=[res[0].name],
            )
            return out[0]

        dx = actual(self.primal0, self.index, self.axis, self.v)

        ddx = desired(self.primal0, self.index, self.axis, self.v)

        np.testing.assert_allclose(
            actual=dx,
            desired=ddx,
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
