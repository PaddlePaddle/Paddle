# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
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

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_sigmoid_grad_comp(self):
        def actual(primal, cotangent):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()

            x = paddle.to_tensor(primal)
            dout = paddle.to_tensor(cotangent)
            x.stop_gradient = False
            return paddle.grad(F.sigmoid(x), x, dout)[0]

        def desired(primal, cotangent):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()

            x = paddle.to_tensor(primal)
            dout = paddle.to_tensor(cotangent)
            x.stop_gradient = False
            return paddle.grad(F.sigmoid(x), x, dout)[0]

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent),
            desired=desired(self.primal, self.cotangent),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
