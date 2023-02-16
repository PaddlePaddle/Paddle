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


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (np.random.rand(10, 10, 10), np.random.rand(10, 10, 10), np.float32),
        (
            np.random.rand(4, 8, 16, 16),
            np.random.rand(4, 8, 16, 16),
            np.float64,
        ),
    ],
)
class TestCumsumGradComp(unittest.TestCase):
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

    def test_cumsum_grad_comp(self):
        def actual(primal, cotangent):
            core._set_prim_backward_enabled(True)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.to_tensor(
                    cotangent, dtype=cotangent.dtype, stop_gradient=False
                )
                y = paddle.cumsum(x, -1)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal},
                fetch_list=x_cotangent,
            )[0]

        def desired(primal, cotangent):
            core._set_prim_backward_enabled(False)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.to_tensor(
                    cotangent, dtype=cotangent.dtype, stop_gradient=False
                )
                y = paddle.cumsum(x, -1)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal},
                fetch_list=x_cotangent,
            )[0]

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent),
            desired=desired(self.primal, self.cotangent),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
