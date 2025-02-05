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


@param.parameterized_class(
    ('shape', 'porder', 'axis', 'keepdim', 'dtype'),
    [
        [[], -float("inf"), -1, True, "float32"],
        [[], -float("inf"), -1, False, "float32"],
        [[], -float("inf"), 0, True, "float32"],
        [[], -float("inf"), 0, False, "float32"],
        [[], float("inf"), -1, True, "float32"],
        [[], float("inf"), -1, False, "float32"],
        [[], float("inf"), 0, True, "float32"],
        [[], float("inf"), 0, False, "float32"],
        [[], 0, -1, True, "float32"],
        [[], 0, -1, False, "float32"],
        [[], 0, 0, True, "float32"],
        [[], 0, 0, False, "float32"],
        [[], 1, -1, True, "float32"],
        [[], 1, -1, False, "float32"],
        [[], 1, 0, True, "float32"],
        [[], 1, 0, False, "float32"],
        [[], 2, -1, True, "float32"],
        [[], 2, -1, False, "float32"],
        [[], 2, 0, True, "float32"],
        [[], 2, 0, False, "float32"],
        [[], 0.3, -1, True, "float32"],
        [[], 0.3, -1, False, "float32"],
        [[], 0.3, 0, True, "float32"],
        [[], 0.3, 0, False, "float32"],
        [[], 1.5, -1, True, "float32"],
        [[], 1.5, -1, False, "float32"],
        [[], 1.5, 0, True, "float32"],
        [[], 1.5, 0, False, "float32"],
        [[], 3.7, -1, True, "float32"],
        [[], 3.7, -1, False, "float32"],
        [[], 3.7, 0, True, "float32"],
        [[], 3.7, 0, False, "float32"],
        [[1], -float("inf"), -1, True, "float32"],
        [[1], -float("inf"), -1, False, "float32"],
        [[1], -float("inf"), 0, True, "float32"],
        [[1], -float("inf"), 0, False, "float32"],
        [[1], float("inf"), -1, True, "float32"],
        [[1], float("inf"), -1, False, "float32"],
        [[1], float("inf"), 0, True, "float32"],
        [[1], float("inf"), 0, False, "float32"],
        [[1], 0, -1, True, "float32"],
        [[1], 0, -1, False, "float32"],
        [[1], 0, 0, True, "float32"],
        [[1], 0, 0, False, "float32"],
        [[1], 1, -1, True, "float32"],
        [[1], 1, -1, False, "float32"],
        [[1], 1, 0, True, "float32"],
        [[1], 1, 0, False, "float32"],
        [[1], 2, -1, True, "float32"],
        [[1], 2, -1, False, "float32"],
        [[1], 2, 0, True, "float32"],
        [[1], 2, 0, False, "float32"],
        [[1], 0.3, -1, True, "float32"],
        [[1], 0.3, -1, False, "float32"],
        [[1], 0.3, 0, True, "float32"],
        [[1], 0.3, 0, False, "float32"],
        [[1], 1.5, -1, True, "float32"],
        [[1], 1.5, -1, False, "float32"],
        [[1], 1.5, 0, True, "float32"],
        [[1], 1.5, 0, False, "float32"],
        [[1], 3.7, -1, True, "float32"],
        [[1], 3.7, -1, False, "float32"],
        [[1], 3.7, 0, True, "float32"],
        [[1], 3.7, 0, False, "float32"],
        [[100, 100], -float("inf"), -1, True, "float32"],
        [[100, 100], -float("inf"), -1, False, "float32"],
        [[100, 100], -float("inf"), 0, True, "float32"],
        [[100, 100], -float("inf"), 0, False, "float32"],
        [[100, 100], float("inf"), -1, True, "float32"],
        [[100, 100], float("inf"), -1, False, "float32"],
        [[100, 100], float("inf"), 0, True, "float32"],
        [[100, 100], float("inf"), 0, False, "float32"],
        [[100, 100], 0, -1, True, "float32"],
        [[100, 100], 0, -1, False, "float32"],
        [[100, 100], 0, 0, True, "float32"],
        [[100, 100], 0, 0, False, "float32"],
        [[100, 100], 1, -1, True, "float32"],
        [[100, 100], 1, -1, False, "float32"],
        [[100, 100], 1, 0, True, "float32"],
        [[100, 100], 1, 0, False, "float32"],
        [[100, 100], 2, -1, True, "float32"],
        [[100, 100], 2, -1, False, "float32"],
        [[100, 100], 2, 0, True, "float32"],
        [[100, 100], 2, 0, False, "float32"],
        [[100, 100], 0.3, -1, True, "float32"],
        [[100, 100], 0.3, -1, False, "float32"],
        [[100, 100], 0.3, 0, True, "float32"],
        [[100, 100], 0.3, 0, False, "float32"],
        [[100, 100], 1.5, -1, True, "float32"],
        [[100, 100], 1.5, -1, False, "float32"],
        [[100, 100], 1.5, 0, True, "float32"],
        [[100, 100], 1.5, 0, False, "float32"],
        [[100, 100], 3.7, -1, True, "float32"],
        [[100, 100], 3.7, -1, False, "float32"],
        [[100, 100], 3.7, 0, True, "float32"],
        [[100, 100], 3.7, 0, False, "float32"],
        [[3, 4, 5, 6, 8], -float("inf"), -1, True, "float32"],
        [[3, 4, 5, 6, 8], -float("inf"), -1, False, "float32"],
        [[3, 4, 5, 6, 8], -float("inf"), 0, True, "float32"],
        [[3, 4, 5, 6, 8], -float("inf"), 0, False, "float32"],
        [[3, 4, 5, 6, 8], float("inf"), -1, True, "float32"],
        [[3, 4, 5, 6, 8], float("inf"), -1, False, "float32"],
        [[3, 4, 5, 6, 8], float("inf"), 0, True, "float32"],
        [[3, 4, 5, 6, 8], float("inf"), 0, False, "float32"],
        [[3, 4, 5, 6, 8], 0, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 0, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 0, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 0, 0, False, "float32"],
        [[3, 4, 5, 6, 8], 1, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 1, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 1, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 1, 0, False, "float32"],
        [[3, 4, 5, 6, 8], 2, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 2, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 2, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 2, 0, False, "float32"],
        [[3, 4, 5, 6, 8], 0.3, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 0.3, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 0.3, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 0.3, 0, False, "float32"],
        [[3, 4, 5, 6, 8], 1.5, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 1.5, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 1.5, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 1.5, 0, False, "float32"],
        [[3, 4, 5, 6, 8], 3.7, -1, True, "float32"],
        [[3, 4, 5, 6, 8], 3.7, -1, False, "float32"],
        [[3, 4, 5, 6, 8], 3.7, 0, True, "float32"],
        [[3, 4, 5, 6, 8], 3.7, 0, False, "float32"],
    ],
)
class TestPNormGradComp(unittest.TestCase):
    def test_p_norm_grad_comp(self):
        paddle.disable_static()
        ndim = len(self.shape)
        norm_axis = self.axis
        if norm_axis < 0:
            norm_axis += ndim

        # skip invalid case
        if 0 < norm_axis < ndim:
            primal = np.random.randn(*self.shape)
            cot_shape = list(self.shape)
            if self.keepdim:
                cot_shape[self.axis] = 1
            else:
                cot_shape.pop(self.axis)

            cotagent = np.random.randn(*cot_shape)
            np.testing.assert_allclose(
                actual=self.actual(
                    primal,
                    self.porder,
                    self.axis,
                    self.keepdim,
                    self.dtype,
                    cotagent,
                ),
                desired=self.desired(
                    primal,
                    self.porder,
                    self.axis,
                    self.keepdim,
                    self.dtype,
                    cotagent,
                ),
                rtol=1e-6,
                atol=0,
            )

    def actual(self, primal, porder, axis, keepdim, dtype, cotagent):
        core.set_prim_eager_enabled(False)
        x = paddle.to_tensor(primal, dtype=dtype, stop_gradient=False)
        y = paddle.norm(x, porder, axis, keepdim=keepdim)
        v = paddle.to_tensor(cotagent, dtype=dtype, stop_gradient=False)
        v.stop_gradient = False
        x_cotangent = paddle.grad(y, x, v)
        return x_cotangent[0]

    def desired(self, primal, porder, axis, keepdim, dtype, cotagent):
        core.set_prim_eager_enabled(True)
        x = paddle.to_tensor(primal, dtype=dtype, stop_gradient=False)
        y = paddle.norm(x, porder, axis, keepdim=keepdim)
        v = paddle.to_tensor(cotagent, dtype=dtype, stop_gradient=False)
        v.stop_gradient = False
        x_cotangent = paddle.grad(y, x, v)
        return x_cotangent[0]


if __name__ == '__main__':
    unittest.main()
