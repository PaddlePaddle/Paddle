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


@param.parameterized_class(
    ('name', 'primals', 'stop_gradients', 'cotangents', 'dtype'),
    (
        (
            'test_normal_case',
            (np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)),
            (False, False),
            (np.random.rand(2, 3, 4),),
            np.float32,
        ),
        (
            'test_broadcast_diff_rank',
            (np.random.rand(2, 3, 1, 4), np.random.rand(3, 3, 4)),
            (False, False),
            (np.random.rand(2, 3, 3, 4),),
            np.float32,
        ),
        (
            'test_broadcast_same_rank',
            (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)),
            (False, False),
            (np.random.rand(2, 3, 3, 4),),
            np.float32,
        ),
        (
            'test_stop_gradient',
            (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)),
            (False, True),
            (np.random.rand(2, 3, 3, 4),),
            np.float32,
        ),
    ),
)
class TestMultiplyGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primals = tuple(primal.astype(cls.dtype) for primal in cls.primals)
        cls.cotangents = tuple(co.astype(cls.dtype) for co in cls.cotangents)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def as_tuple(self, x):
        return (x,) if isinstance(x, framework.Variable) else x

    def net(self):
        primals, cotangents = self.primals, self.cotangents
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            primals = tuple(
                paddle.static.data(f'primal{i}', primal.shape, primal.dtype)
                for i, primal in enumerate(primals)
            )
            for primal, flag in zip(primals, self.stop_gradients):
                primal.stop_gradient = flag
            cotangents = tuple(
                paddle.static.data(f'cotangent{i}', co.shape, co.dtype)
                for i, co in enumerate(cotangents)
            )
            out = self.as_tuple(paddle.tanh(paddle.multiply(*primals)))
            grads = paddle.static.gradients(out, primals)
        exe = paddle.static.Executor()
        exe.run(sp)
        return exe.run(
            program=mp,
            feed={
                **{
                    f'primal{i}': primal
                    for i, primal in enumerate(self.primals)
                },
                **{f'cotangent{i}': co for i, co in enumerate(self.cotangents)},
            },
            fetch_list=[g for g in grads if g is not None],
        )

    def test_comp(self):
        core._set_prim_backward_enabled(True)
        actual = self.net()

        core._set_prim_backward_enabled(False)
        desired = self.net()

        self.assertEqual(len(actual), len(desired))
        for i, j in zip(actual, desired):
            np.testing.assert_allclose(
                i,
                j,
                rtol=1e-6,
                atol=0,
            )


if __name__ == '__main__':
    unittest.main()
