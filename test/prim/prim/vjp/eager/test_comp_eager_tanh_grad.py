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

core.set_prim_eager_enabled(True)


@param.parameterized_class(
    ('primal', 'dtype'),
    [
        (
            np.random.rand(2, 3, 4),
            np.float32,
        ),
        (
            np.random.rand(2, 3, 3, 4),
            np.float32,
        ),
    ],
)
class TestTanhGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)

    def test_tanh_grad_comp(self):
        def actual(primal):
            paddle.disable_static()
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.tanh(x)
            x_cotangent = paddle.grad(
                y, x, create_graph=True, retain_graph=True
            )
            return x_cotangent[0]

        def desired(primal):
            paddle.disable_static()
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.tanh(x)
            x_cotangent = paddle.grad(
                y, x, create_graph=True, retain_graph=True
            )
            return x_cotangent[0]

        np.testing.assert_allclose(
            actual=actual(self.primal),
            desired=desired(self.primal),
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
