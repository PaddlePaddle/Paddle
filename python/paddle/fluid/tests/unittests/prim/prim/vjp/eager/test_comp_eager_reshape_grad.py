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
from paddle.fluid import core

core._set_prim_backward_enabled(True)


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
        (np.random.rand(), [1], np.random.rand(1), np.float32),
    ],
)
class TestSqrtGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_reshape_grad_comp(self):
        def actual(primal0, shape):
            core._set_prim_backward_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()

        def desired(primal0, shape):
            core._set_prim_backward_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()

        dx = actual(self.primal0, self.shape)

        ddx = desired(self.primal0, self.shape)

        np.testing.assert_allclose(
            actual=dx,
            desired=ddx,
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
