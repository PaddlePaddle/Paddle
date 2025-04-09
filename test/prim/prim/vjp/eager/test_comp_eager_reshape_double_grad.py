# Copyright (c) 2024 PaddlePaddle Authors. All Rights ddxerved.
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
    ('primal', 'shape', 'cotangent', 'dtype'),
    [
        (
            np.random.rand(10, 1, 10),
            [10, 10],
            np.random.rand(10, 10),
            np.float32,
        ),
        (np.random.rand(2, 60), [12, 10], np.random.rand(12, 10), np.float32),
    ],
)
class TestReshapeDoubleGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)

    def test_reshape_double_grad_comp(self):
        def actual(primal0, shape):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            # disable rshape_grad to trigger the composite double_grad
            core._set_prim_backward_blacklist("reshape_grad")

            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            # wrap by tanh for >= 2 order derivative
            out = paddle.tanh(out)
            dx = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            ddx = paddle.grad(dx, [x], create_graph=True, retain_graph=True)
            return ddx[0].numpy()

        def desired(primal0, shape):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            # wrap by tanh for >= 2 order derivative
            out = paddle.tanh(out)
            dx = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            ddx = paddle.grad(dx, [x], create_graph=True, retain_graph=True)
            return ddx[0].numpy()

        actual_result = actual(self.primal, self.shape)

        desired_result = desired(self.primal, self.shape)

        np.testing.assert_allclose(
            actual=actual_result,
            desired=desired_result,
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
