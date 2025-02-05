# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
    ('primal', 'axis', 'cotangent', 'dtype'),
    [
        (np.random.rand(16, 32), [1], np.random.rand(16, 32), np.float32),
        (np.random.rand(16, 32), [0], np.random.rand(16, 32), np.float32),
    ],
)
class TestMinGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)

    def test_min_grad_comp(self):
        def actual(primal0, axis):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.min(x, axis)
            res = paddle.grad(out, [x], create_graph=False)
            return res[0].numpy()

        def desired(primal0, axis):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.min(x, axis)
            res = paddle.grad(out, [x], create_graph=False)
            return res[0].numpy()

        dx = actual(self.primal, self.axis)

        ddx = desired(self.primal, self.axis)

        np.testing.assert_allclose(
            actual=dx,
            desired=ddx,
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
