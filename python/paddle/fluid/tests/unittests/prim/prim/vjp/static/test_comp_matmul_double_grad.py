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
    ('primal0', 'primal1', 'primal2', 'trans_0', 'trans_1', 'dtype'),
    [
        # (
        #     np.random.rand(2),
        #     np.random.rand(2),
        #     np.random.rand(1),
        #     False,
        #     False,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2, 3),
        #     np.random.rand(3),
        #     np.random.rand(2),
        #     False,
        #     False,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2),
        #     np.random.rand(2, 3),
        #     np.random.rand(3),
        #     False,
        #     False,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2),
        #     np.random.rand(3, 2),
        #     False,
        #     True,
        #     np.float32,
        # ),
        (
            np.random.rand(2, 3, 4),
            np.random.rand(2, 4, 5),
            np.random.rand(2, 3, 5),
            False,
            False,
            np.float32,
        ),
        # (
        #     np.random.rand(2, 4, 3),
        #     np.random.rand(2, 4, 5),
        #     np.random.rand(2, 3, 5),
        #     True,
        #     False,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2, 3, 4),
        #     np.random.rand(2, 5, 4),
        #     np.random.rand(2, 3, 5),
        #     False,
        #     True,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2, 4, 3),
        #     np.random.rand(2, 5, 4),
        #     np.random.rand(2, 3, 5),
        #     True,
        #     True,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2, 3, 4),
        #     np.random.rand(4),
        #     np.random.rand(2, 3),
        #     False,
        #     False,
        #     np.float32,
        # ),
        # (
        #     np.random.rand(2, 1, 5, 2),
        #     np.random.rand(1, 3, 2, 4),
        #     np.random.rand(2, 3, 5, 4),
        #     False,
        #     False,
        #     np.float32,
        # ),
    ],
)
class TestMatmulDoubleGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)
        cls.primal2 = cls.primal2.astype(cls.dtype)
        cls.trans_0 = cls.trans_0
        cls.trans_1 = cls.trans_1

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_matmul_grad_comp(self):
        def actual(primal0, primal1, primal2, trans_0, trans_1):
            core._set_prim_backward_enabled(True)
            paddle.enable_static()
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal0', primal0.shape, primal0.dtype)
                y = paddle.static.data('primal1', primal1.shape, primal1.dtype)
                z = paddle.static.data('primal2', primal2.shape, primal2.dtype)
                x.stop_gradient = False
                y.stop_gradient = False
                z.stop_gradient = False
                out = paddle.matmul(x, y, trans_0, trans_1)

                res = paddle.static.gradients([out], [x, y], z)
                res_double = paddle.static.gradients(res, [x, y, z])

                exe = paddle.static.Executor()
                exe.run(sp)
                print("program = ", mp)
                out = exe.run(
                    program=mp,
                    feed={
                        'primal0': primal0,
                        'primal1': primal1,
                        'primal2': primal2,
                    },
                    fetch_list=[
                        res_double[0].name,
                        res_double[1].name,
                        res_double[2].name,
                    ],
                )

            return out[0], out[1], out[2]

        def desired(primal0, primal1, primal2, trans_0, trans_1):
            core._set_prim_backward_enabled(False)
            paddle.enable_static()
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal0', primal0.shape, primal0.dtype)
                y = paddle.static.data('primal1', primal1.shape, primal1.dtype)
                z = paddle.static.data('primal2', primal2.shape, primal2.dtype)
                x.stop_gradient = False
                y.stop_gradient = False
                z.stop_gradient = False
                out = paddle.matmul(x, y, trans_0, trans_1)
                dout = paddle.ones_like(out, dtype='float32')
                dout.stop_gradient = False
                res = paddle.static.gradients([out], [x, y], dout)
                res_double = paddle.static.gradients(res, [x, y, dout])

                exe = paddle.static.Executor()
                exe.run(sp)
                out = exe.run(
                    program=mp,
                    feed={
                        'primal0': primal0,
                        'primal1': primal1,
                        'primal2': primal2,
                    },
                    fetch_list=[
                        res_double[0].name,
                        res_double[1].name,
                        res_double[2].name,
                    ],
                )

            return out[0], out[1], out[2]

        dx, dy, ddout = actual(
            self.primal0, self.primal1, self.primal2, self.trans_0, self.trans_1
        )

        # dx_, dy_, ddout_ = desired(
        #     self.primal0, self.primal1, self.primal2, self.trans_0, self.trans_1
        # )

        print("dx:", dx)
        print("dy:", dy)
        print("ddout:", ddout)
        # print("dx_: ", dx_)
        # print("dy_: ", dy_)
        # print("ddout_: ", ddout_)

        # np.testing.assert_allclose(
        #     actual=dx,
        #     desired=dx_,
        #     rtol=1e-6,
        #     atol=0,
        # )
        # np.testing.assert_allclose(
        #     actual=dy,
        #     desired=dy_,
        #     rtol=1e-6,
        #     atol=0,
        # )
        # np.testing.assert_allclose(
        #     actual=ddout,
        #     desired=ddout_,
        #     rtol=1e-6,
        #     atol=0,
        # )


core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
