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

import autograd
import autograd.numpy
import numpy as np
import parameterized as param

import paddle
from paddle.base import core


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (np.random.rand(10, 10), np.random.rand(10, 10), np.float32),
        (np.random.rand(10, 10), None, np.float32),
    ],
)
class TestExpGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_backward_enabled(True)
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_backward_enabled(False)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_exp_grad_comp(self):
        def actual(primal, cotangent):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = (
                    None
                    if cotangent is None
                    else paddle.static.data(
                        'cotangent', cotangent.shape, cotangent.dtype
                    )
                )
                y = paddle.exp(x)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=x_cotangent,
            )[0]

        def desired(primal, cotangent):
            cotangent = (
                np.ones_like(cotangent, dtype=primal.dtype)
                if cotangent is None
                else cotangent
            )
            return autograd.make_vjp(autograd.numpy.exp)(primal)[0](cotangent)

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent),
            desired=desired(self.primal, self.cotangent),
            rtol=1e-6,
            atol=0,
        )

    def test_stop_gradient(self):
        def actual(primal, cotangent):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = True
                v = (
                    None
                    if cotangent is None
                    else paddle.static.data(
                        'cotangent', cotangent.shape, cotangent.dtype
                    )
                )
                y = paddle.exp(x)
                x_cotangent = paddle.static.gradients(y, x, v)
            if x_cotangent == [None]:
                x_cotangent = []
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=x_cotangent,
            )

        def desired(primal, cotangent):
            return []

        self.assertEqual(
            actual(self.primal, self.cotangent),
            desired(self.primal, self.cotangent),
        )


if __name__ == '__main__':
    unittest.main()
