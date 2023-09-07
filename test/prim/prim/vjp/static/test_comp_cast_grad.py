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
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        tmp = self.fc(x)
        out = paddle.cast(tmp, paddle.float64)
        return out


@param.parameterized_class(
    ('primal', 'cotangent', 'src_dtype', 'dst_type'),
    [
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float32,
            np.float64,
        ),
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float64,
            np.float32,
        ),
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float32,
            np.float32,
        ),
    ],
)
class TestCastGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.src_dtype)
        cls.cotangent = cls.cotangent.astype(cls.src_dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

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

    def test_cinn(self):
        paddle.disable_static()
        use_cinn = True
        if isinstance(
            framework._current_expected_place(), framework.core.CPUPlace
        ):
            # TODO(jiabin): CINN will crashed in this case open it when fixed
            use_cinn = False

        dy_res = self.train(use_prim=False, use_cinn=False)
        comp_st_cinn_res = self.train(use_prim=True, use_cinn=use_cinn)

        for i in range(len(dy_res)):
            np.testing.assert_allclose(
                comp_st_cinn_res[i].numpy(),
                dy_res[i].numpy(),
                rtol=1e-15,
                atol=1e-15,
            )
        paddle.enable_static()

    def test_cast_grad_comp(self):
        core._set_prim_backward_enabled(True)

        def actual(primal, cotangent):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.cast(x, self.dst_type)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=mp.blocks[0].ops[-1].output('Out')[0],
            )[0]

        def desired(primal, cotangent):
            return (cotangent * np.ones_like(primal)).astype(primal.dtype)

        actual = actual(self.primal, self.cotangent)
        desired = desired(self.primal, self.cotangent)

        self.assertEqual(actual.dtype, desired.dtype)
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
