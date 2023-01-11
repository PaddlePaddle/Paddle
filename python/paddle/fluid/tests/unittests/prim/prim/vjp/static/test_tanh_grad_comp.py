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

from paddle.fluid import core

core.set_prim_enabled(True)

import autograd
import autograd.numpy
import numpy as np
import parameterized as param

import paddle


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (np.random.rand(10, 10), np.random.rand(10, 10), np.float32),
    ],
)
class TestTanhGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_tanh_grad_comp(self):
        def actual(primal, cotangent):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.tanh(x)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=mp.blocks[0].ops[-1].output('Out')[0],
            )[0]

        def desired(primal, cotangent):
            return autograd.make_vjp(autograd.numpy.tanh)(primal)[0](cotangent)

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent),
            desired=desired(self.primal, self.cotangent),
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_enabled(False)


if __name__ == '__main__':
    unittest.main()
