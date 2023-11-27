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
    ],
)
class TestExpGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.set_prim_eager_enabled(True)
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        core.set_prim_eager_enabled(False)

    def test_exp_grad_comp(self):
        def actual(primal, cotangent):
            primal = paddle.to_tensor(primal)
            primal.stop_gradient = False
            return paddle.grad(
                paddle.exp(primal), primal, paddle.to_tensor(cotangent)
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

    def test_stop_gradients(self):
        with self.assertRaises(ValueError):
            primal = paddle.to_tensor(self.primal)
            primal.stop_gradient = True
            return paddle.grad(
                paddle.exp(primal), primal, paddle.to_tensor(self.cotangent)
            )


if __name__ == '__main__':
    unittest.main()
