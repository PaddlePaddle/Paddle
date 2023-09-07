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


@param.parameterized_class(
    ('name', 'primal', 'cotangent', 'shape', 'dtype'),
    (
        (
            'same_shape',
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            (10, 10),
            np.float32,
        ),
        (
            'same_rank',
            np.random.rand(1, 10),
            np.random.rand(10, 10),
            (10, 10),
            np.float32,
        ),
        (
            'same_rank',
            np.random.rand(10, 1, 10, 1),
            np.random.rand(10, 10, 10, 10),
            (10, 10, 10, 10),
            np.float32,
        ),
        (
            'diff_rank',
            np.random.rand(1, 10, 1),
            np.random.rand(10, 10, 10, 10),
            (10, 10, 10, 10),
            np.float32,
        ),
        (
            'single_direction_broadcast',
            np.random.rand(10, 10, 10, 10),
            np.random.rand(1, 10, 1),
            (10, 10, 10, 10),
            np.float32,
        ),
    ),
)
class TestExpandGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)
        paddle.enable_static()

    @classmethod
    def tearDownClass(cls):
        paddle.disable_static()
        core._set_prim_backward_enabled(False)

    def test_comp(self):
        def func(primal, cotangent, shape):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                y = paddle.expand(x, shape)
                x_cotangent = paddle.static.gradients(y, x)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(
                program=mp,
                feed={'primal': primal, 'cotangent': cotangent},
                fetch_list=x_cotangent,
            )[0]

        def actual(primal, cotangent, shape):
            core._set_prim_backward_enabled(True)
            return func(primal, cotangent, shape)

        def desired(primal, cotangent, shape):
            core._set_prim_backward_enabled(False)
            return func(primal, cotangent, shape)

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent, self.shape),
            desired=desired(self.primal, self.cotangent, self.shape),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
