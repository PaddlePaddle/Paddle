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
class TestCumprodGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)

    def test_cumprod_grad_comp(self):
        def actual(primal, dim):
            paddle.disable_static()
            core.set_prim_eager_enabled(True)
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.cumprod(x, dim=dim)
            x_cotangent = paddle.grad(
                y, x, create_graph=True, retain_graph=True
            )
            return x_cotangent[0]

        def desired(primal, dim):
            paddle.disable_static()
            core.set_prim_eager_enabled(False)
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.cumprod(x, dim=dim)
            x_cotangent = paddle.grad(
                y, x, create_graph=False, retain_graph=True
            )
            return x_cotangent[0]

        for i in range(len(self.primal.shape)):
            np.testing.assert_allclose(
                actual=actual(self.primal, i),
                desired=desired(self.primal, i),
                rtol=1e-6,
                atol=0,
            )
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
