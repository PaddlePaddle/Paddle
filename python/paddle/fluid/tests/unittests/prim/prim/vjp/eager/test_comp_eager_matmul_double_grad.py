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

# vector * vector out.shape = (1)
# matrix * vector out.shape = (2)
# vector * matrix out.shape = (3)
# batched matrix * batched matrix 4 for trans out.shape = (2, 3, 5)
# batched matrix * broadcasted vector out.shape = (2, 3)
# batched matrix * broadcasted matrix out.shape = (2, 3, 5, 4)


@param.parameterized_class(
    ('primal0', 'primal1', 'trans_0', 'trans_1', 'dtype'),
    [
        (
            np.random.rand(2),
            np.random.rand(2),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 3),
            np.random.rand(3),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2),
            np.random.rand(2, 3),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 4, 5),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 4, 5),
            True,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 5, 4),
            False,
            True,
            np.float32,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 5, 4),
            True,
            True,
            np.float32,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 1, 5, 2),
            np.random.rand(1, 3, 2, 4),
            False,
            False,
            np.float32,
        ),
    ],
)
class TestTanhGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_matmul_grad_comp(self):
        def actual(primal0, primal1, trans_0, trans_1):
            core._set_prim_backward_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.matmul(x, y, trans_0, trans_1)
            res = paddle.grad(out, [x, y], create_graph=True, retain_graph=True)
            res_double = paddle.grad(
                res, [x, y], create_graph=True, retain_graph=True
            )
            return res_double[0].numpy(), res_double[1].numpy()

        def desired(primal0, primal1, trans_0, trans_1):
            core._set_prim_backward_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.matmul(x, y, trans_0, trans_1)
            res = paddle.grad(out, [x, y], create_graph=True, retain_graph=True)
            res_double = paddle.grad(
                res, [x, y], create_graph=True, retain_graph=True
            )
            return res_double[0].numpy(), res_double[1].numpy()

        dx, dy = actual(self.primal0, self.primal1, self.trans_0, self.trans_1)

        ddx, ddy = desired(
            self.primal0, self.primal1, self.trans_0, self.trans_1
        )

        np.testing.assert_allclose(
            actual=dx,
            desired=ddx,
            rtol=1e-6,
            atol=0,
        )
        np.testing.assert_allclose(
            actual=dy,
            desired=ddy,
            rtol=1e-6,
            atol=0,
        )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
