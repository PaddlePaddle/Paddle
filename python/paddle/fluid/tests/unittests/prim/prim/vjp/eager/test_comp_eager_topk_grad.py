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

core._set_prim_backward_enabled(False)


limit = {
    'float16': {'atol': 1e-3, 'rtol': 1e-3},
    'float32': {'atol': 1e-6, 'rtol': 1e-6},
    'float64': {'atol': 1e-15, 'rtol': 1e-15},
}


@param.parameterized_class(
    ('primal', 'k', 'axis', 'largest', 'sorted', 'x_dtype', 'v'),
    [
        (
            np.random.rand(5),
            3,
            0,
            False,
            False,
            np.float32,
            np.random.rand(3),
        ),
        (
            np.random.rand(3, 3),
            3,
            0,
            True,
            True,
            np.float32,
            np.random.rand(3, 3),
        ),
        (
            np.random.rand(10, 10, 10),
            5,
            0,
            True,
            False,
            np.float32,
            np.random.rand(5, 10, 10),
        ),
        (
            np.random.rand(4, 8, 16, 16),
            3,
            1,
            False,
            True,
            np.float64,
            np.random.rand(4, 3, 16, 16),
        ),
    ],
)
class TestTopkGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.x_dtype)
        cls.v = cls.v.astype(cls.x_dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_topk_grad_comp(self):
        def actual(primal, k, axis, largest, sorted):
            paddle.disable_static()
            core._set_prim_backward_enabled(True)
            x = paddle.to_tensor(
                primal, dtype=primal.dtype, stop_gradient=False
            )
            x.stop_gradient = False
            y_v, _ = paddle.topk(x, k, axis, largest, sorted)
            return paddle.grad(
                [y_v], [x], create_graph=False, retain_graph=False
            )[0]

        def desired(primal, k, axis, largest, sorted):
            paddle.disable_static()
            core._set_prim_backward_enabled(False)
            x = paddle.to_tensor(
                primal, dtype=primal.dtype, stop_gradient=False
            )
            x.stop_gradient = False
            y_v, _ = paddle.topk(x, k, axis, largest, sorted)
            return paddle.grad(
                [y_v], [x], create_graph=True, retain_graph=True
            )[0]

        if (
            paddle.device.get_device() == "cpu"
            and self.primal.dtype == np.float16
        ):
            print("pass cpu+float16 case")
        else:
            np.testing.assert_allclose(
                actual=actual(
                    self.primal, self.k, self.axis, self.largest, self.sorted
                ),
                desired=desired(
                    self.primal, self.k, self.axis, self.largest, self.sorted
                ),
                rtol=limit[str(self.primal.dtype)]['rtol'],
                atol=limit[str(self.primal.dtype)]['atol'],
            )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
