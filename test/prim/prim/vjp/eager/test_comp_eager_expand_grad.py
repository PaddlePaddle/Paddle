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
    ),
)
class TestExpandGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        core.set_prim_eager_enabled(False)

    def test_comp(self):
        def func(primal, cotangent, shape):
            primal = paddle.to_tensor(primal)
            primal.stop_gradient = False
            cotangent = paddle.to_tensor(cotangent)
            return paddle.grad(paddle.expand(primal, shape), primal, cotangent)[
                0
            ]

        def actual(primal, cotangent, shape):
            core.set_prim_eager_enabled(True)
            return func(primal, cotangent, shape)

        def desired(primal, cotangent, shape):
            core.set_prim_eager_enabled(False)
            return func(primal, cotangent, shape)

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent, self.shape),
            desired=desired(self.primal, self.cotangent, self.shape),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
