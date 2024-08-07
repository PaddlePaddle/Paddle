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
from paddle.base import core, framework


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
        out = paddle.reshape(tmp, [2, 1, 4])
        return out


@param.parameterized_class(
    ('primal', 'shape', 'cotangent', 'dtype', "rtol"),
    [
        (
            np.random.rand(10, 1, 10),
            [10, 10],
            np.random.rand(10, 10),
            np.float32,
            1e-5,
        ),
        (
            np.random.rand(2, 60),
            [12, 10],
            np.random.rand(12, 10),
            np.float32,
            1e-5,
        ),
        (
            np.random.rand(10, 1, 10),
            [10, 10],
            np.random.rand(10, 10),
            np.float64,
            1e-15,
        ),
        (
            np.random.rand(2, 60),
            [12, 10],
            np.random.rand(12, 10),
            np.float64,
            1e-15,
        ),
        (
            np.random.rand(10, 1, 10),
            [10, 10],
            np.random.rand(10, 10),
            np.float16,
            1e-3,
        ),
        (
            np.random.rand(2, 60),
            [12, 10],
            np.random.rand(12, 10),
            np.float16,
            1e-3,
        ),
    ],
)
class TestReshapeGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

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

    def test_reshape_grad_comp(self):
        paddle.enable_static()

        def actual(primal, shape, cotangent):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.reshape(x, shape)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        def desired(primal, shape, cotangent):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.reshape(x, shape)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=[x_cotangent[0]],
            )[0]

        if (self.dtype == np.float16) and isinstance(
            framework._current_expected_place(), framework.core.CPUPlace
        ):
            # reshape doesn't support fp16 kernel in cpu
            pass
        else:
            np.testing.assert_allclose(
                actual=actual(self.primal, self.shape, self.cotangent),
                desired=desired(self.primal, self.shape, self.cotangent),
                rtol=self.rtol,
                atol=self.rtol,
            )
        core._set_prim_backward_enabled(False)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
