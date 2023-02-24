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

limit = {
    'float16': {'atol': 1e-3, 'rtol': 1e-3},
    'float32': {'atol': 1e-6, 'rtol': 1e-6},
    'float64': {'atol': 1e-15, 'rtol': 1e-15},
}


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super(PrimeNet, self).__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x, k, axis, largest, sorted):
        tmp = self.fc(x)
        out = paddle.topk(tmp, k, axis, largest, sorted)
        return out


@param.parameterized_class(
    ('primal', 'k', 'axis', 'largest', 'sorted', 'x_dtype', 'v'),
    [
        (
            np.random.rand(5),
            3,
            0,
            False,
            False,
            np.float32,
            np.random.rand(3),
        ),
        (
            np.random.rand(3, 3),
            3,
            0,
            True,
            True,
            np.float32,
            np.random.rand(3, 3),
        ),
        (
            np.random.rand(10, 10, 10),
            5,
            0,
            True,
            False,
            np.float32,
            np.random.rand(5, 10, 10),
        ),
        (
            np.random.rand(4, 8, 16, 16),
            3,
            1,
            False,
            True,
            np.float64,
            np.random.rand(4, 3, 16, 16),
        ),
    ],
)
class TestTopkGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_backward_enabled(True)
        cls.primal = cls.primal.astype(cls.x_dtype)
        cls.v = cls.v.astype(cls.x_dtype)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_backward_enabled(False)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def train(self, use_prim, use_cinn):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.k = 3
        self.axis = 1
        self.largest = True
        self.sorted = True
        self.x.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x, self.k, self.axis, self.largest, self.sorted)
        res = paddle.autograd.grad(out, [self.x])
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

    def test_topk_grad_comp(self):
        def actual(primal, k, axis, largest, sorted, v):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                y_v, _ = paddle.topk(x, k, axis, largest, sorted)
                y_grad = paddle.static.data('v', y_v.shape, y_v.dtype)
                res = paddle.static.gradients([y_v], [x], [y_grad])
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'v': v},
                fetch_list=[res[0].name],
            )[0]

        def desired(primal, k, axis, largest, sorted, v):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                y_v, _ = paddle.topk(x, k, axis, largest, sorted)
                y_grad = paddle.static.data('v', y_v.shape, y_v.dtype)
                res = paddle.static.gradients([y_v], [x], [y_grad])
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'v': v},
                fetch_list=[res[0].name],
            )[0]

        if (
            paddle.device.get_device() == "cpu"
            and self.primal.dtype == np.float16
        ):
            print("pass cpu+float16 case")
        else:
            np.testing.assert_allclose(
                actual=actual(
                    self.primal,
                    self.k,
                    self.axis,
                    self.largest,
                    self.sorted,
                    self.v,
                ),
                desired=desired(
                    self.primal,
                    self.k,
                    self.axis,
                    self.largest,
                    self.sorted,
                    self.v,
                ),
                rtol=limit[str(self.primal.dtype)]['rtol'],
                atol=limit[str(self.primal.dtype)]['atol'],
            )


if __name__ == '__main__':
    unittest.main()
