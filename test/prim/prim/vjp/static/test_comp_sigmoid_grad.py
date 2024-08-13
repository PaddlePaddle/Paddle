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
            core._set_prim_backward_enabled(True)
            paddle.enable_static()

            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                dout = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                x.stop_gradient = False
                res = F.sigmoid(x)
                x_grad = paddle.static.gradients(res, [x], dout)

                exe = paddle.static.Executor()
                exe.run(sp)
                out = exe.run(
                    program=mp,
                    feed={
                        'primal': primal,
                        'cotangent': cotangent,
                    },
                    fetch_list=[
                        x_grad[0],
                    ],
                )

            return out[0]

        def desired(primal, cotangent):
            core._set_prim_backward_enabled(False)
            paddle.enable_static()

            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                dout = paddle.static.data(
                    'cotangent', cotangent.shape, cotangent.dtype
                )
                x.stop_gradient = False
                res = F.sigmoid(x)
                x_grad = paddle.static.gradients(res, [x], dout)

                exe = paddle.static.Executor()
                exe.run(sp)
                out = exe.run(
                    program=mp,
                    feed={
                        'primal': primal,
                        'cotangent': cotangent,
                    },
                    fetch_list=[
                        x_grad[0],
                    ],
                )

            return out[0]

        np.testing.assert_allclose(
            actual=actual(self.primal, self.cotangent),
            desired=desired(self.primal, self.cotangent),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
