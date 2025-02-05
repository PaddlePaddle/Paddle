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

# vector * vector out.shape = (1)
# matrix * vector out.shape = (2)
# vector * matrix out.shape = (3)
# batched matrix * batched matrix 4 for trans out.shape = (2, 3, 5)
# batched matrix * broadcasted vector out.shape = (2, 3)
# batched matrix * broadcasted matrix out.shape = (2, 3, 5, 4)

TOLERANCE = {
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "float32": {"rtol": 1e-6, "atol": 1e-6},
    "float64": {"rtol": 1e-15, "atol": 1e-15},
}


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
            np.random.rand(2),
            np.random.rand(3, 2),
            False,
            True,
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
        (
            np.random.rand(2),
            np.random.rand(2),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 3),
            np.random.rand(3),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2),
            np.random.rand(2, 3),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2),
            np.random.rand(3, 2),
            False,
            True,
            np.float16,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 4, 5),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 4, 5),
            True,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 5, 4),
            False,
            True,
            np.float16,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 5, 4),
            True,
            True,
            np.float16,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 1, 5, 2),
            np.random.rand(1, 3, 2, 4),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2),
            np.random.rand(2),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 3),
            np.random.rand(3),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2),
            np.random.rand(2, 3),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2),
            np.random.rand(3, 2),
            False,
            True,
            np.float64,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 5, 4),
            False,
            True,
            np.float64,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 4, 5),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 4, 5),
            True,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 5, 4),
            True,
            True,
            np.float64,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 1, 5, 2),
            np.random.rand(1, 3, 2, 4),
            False,
            False,
            np.float64,
        ),
    ],
)
class TestMatmulDoubleGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)
        cls.trans_0 = cls.trans_0
        cls.trans_1 = cls.trans_1

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_matmul_grad_comp(self):
        def actual(primal0, primal1, trans_0, trans_1, dtype_):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype=dtype_, stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype=dtype_, stop_gradient=False)
            out = paddle.matmul(x, y, trans_0, trans_1)
            dout = paddle.ones_like(out, dtype=dtype_)
            dout.stop_gradient = False
            res = paddle.grad(
                [out], [x, y], dout, create_graph=True, retain_graph=True
            )
            res_double = paddle.grad(
                res, [x, y, dout], create_graph=True, retain_graph=True
            )
            return (
                res_double[0].numpy(),
                res_double[1].numpy(),
                res_double[2].numpy(),
            )

        def desired(primal0, primal1, trans_0, trans_1, dtype_):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype=dtype_, stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype=dtype_, stop_gradient=False)
            out = paddle.matmul(x, y, trans_0, trans_1)
            dout = paddle.ones_like(out, dtype=dtype_)
            dout.stop_gradient = False
            res = paddle.grad(
                out, [x, y], dout, create_graph=True, retain_graph=True
            )
            res_double = paddle.grad(
                res, [x, y, dout], create_graph=True, retain_graph=True
            )
            return (
                res_double[0].numpy(),
                res_double[1].numpy(),
                res_double[2].numpy(),
            )

        d_type = "float32"
        if self.primal0.dtype == np.float16:
            d_type = "float16"
        elif self.primal0.dtype == np.float64:
            d_type = "float64"

        if paddle.device.get_device() == "cpu" and d_type == "float16":
            # matmul fp16 cpu not supposed
            pass
        else:
            dx, dy, ddout = actual(
                self.primal0, self.primal1, self.trans_0, self.trans_1, d_type
            )

            dx_, dy_, ddout_ = desired(
                self.primal0, self.primal1, self.trans_0, self.trans_1, d_type
            )

            np.testing.assert_allclose(
                actual=dx,
                desired=dx_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=dy,
                desired=dy_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=ddout,
                desired=ddout_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )


@param.parameterized_class(
    ('primal0', 'primal1', 'trans_0', 'trans_1', 'dtype'),
    [
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(4),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 2, 3),
            np.random.rand(2, 3, 2),
            False,
            False,
            np.float16,
        ),
        (
            np.random.rand(2, 2, 3),
            np.random.rand(2, 3, 2),
            False,
            False,
            np.float32,
        ),
        (
            np.random.rand(2, 2, 3),
            np.random.rand(2, 3, 2),
            False,
            False,
            np.float64,
        ),
        (
            np.random.rand(2, 4, 3),
            np.random.rand(2, 5, 4),
            True,
            True,
            np.float64,
        ),
        (
            np.random.rand(2, 2, 3),
            np.random.rand(1, 3, 2),
            False,
            False,
            np.float64,
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
class TestMatmulTripleGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)
        cls.trans_0 = cls.trans_0
        cls.trans_1 = cls.trans_1

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_matmul_grad_comp(self):
        def actual(primal0, primal1, trans_0, trans_1, dtype_):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype=dtype_, stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype=dtype_, stop_gradient=False)
            out = paddle.matmul(x, y, trans_0, trans_1)
            dout = paddle.ones_like(out, dtype=dtype_)
            dout.stop_gradient = False
            ddx = paddle.ones_like(x, dtype=dtype_)
            ddx.stop_gradient = False
            ddy = paddle.ones_like(y, dtype=dtype_)
            ddy.stop_gradient = False
            res = paddle.grad(
                [out], [x, y], dout, create_graph=True, retain_graph=True
            )
            res_double = paddle.grad(
                res,
                [x, y, dout],
                [ddx, ddy],
                create_graph=True,
                retain_graph=True,
            )

            res_triple = paddle.grad(
                res_double,
                [x, y, dout, ddx, ddy],
                create_graph=False,
                retain_graph=False,
            )
            return (
                res_double[0].numpy(),
                res_double[1].numpy(),
                res_double[2].numpy(),
                res_triple[0].numpy(),
                res_triple[1].numpy(),
            )

        def desired(primal0, primal1, trans_0, trans_1, dtype_):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype=dtype_, stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype=dtype_, stop_gradient=False)
            out = paddle.matmul(x, y, trans_0, trans_1)
            dout = paddle.ones_like(out, dtype=dtype_)
            dout.stop_gradient = False
            ddx = paddle.ones_like(x, dtype=dtype_)
            ddx.stop_gradient = False
            ddy = paddle.ones_like(y, dtype=dtype_)
            ddy.stop_gradient = False
            res = paddle.grad(
                [out], [x, y], [dout], create_graph=True, retain_graph=True
            )
            res_double = paddle.grad(
                res,
                [x, y, dout],
                [ddx, ddy],
                create_graph=True,
                retain_graph=True,
            )
            res_triple = paddle.grad(
                res_double,
                [x, y, dout, ddx, ddy],
                create_graph=False,
                retain_graph=True,
            )
            return (
                res_double[0].numpy(),
                res_double[1].numpy(),
                res_double[2].numpy(),
                res_triple[0].numpy(),
                res_triple[1].numpy(),
            )

        d_type = "float32"
        if self.primal0.dtype == np.float16:
            d_type = "float16"
        elif self.primal0.dtype == np.float64:
            d_type = "float64"

        if paddle.device.get_device() == "cpu" and d_type == "float16":
            # matmul fp16 cpu not supposed
            pass
        else:
            dx, dy, ddout, dx2, dy2 = actual(
                self.primal0, self.primal1, self.trans_0, self.trans_1, d_type
            )

            dx_, dy_, ddout_, dx2_, dy2_ = desired(
                self.primal0, self.primal1, self.trans_0, self.trans_1, d_type
            )

            np.testing.assert_allclose(
                actual=dx,
                desired=dx_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=dy,
                desired=dy_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=ddout,
                desired=ddout_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=dx2,
                desired=dx2_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )
            np.testing.assert_allclose(
                actual=dy2,
                desired=dy2_,
                rtol=TOLERANCE[d_type]['rtol'],
                atol=TOLERANCE[d_type]['atol'],
            )

    core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
