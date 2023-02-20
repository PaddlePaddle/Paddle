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


limit = {
    'float16': {'atol': 1e-3, 'rtol': 1e-3},
    'float32': {'atol': 1e-6, 'rtol': 1e-6},
    'float64': {'atol': 1e-15, 'rtol': 1e-15},
}


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (np.random.rand(3, 3), np.random.rand(3, 3), np.float16),
        (np.random.rand(10, 10, 10), np.random.rand(10, 10, 10), np.float32),
        (
            np.random.rand(4, 8, 16, 16),
            np.random.rand(4, 8, 16, 16),
            np.float64,
        ),
    ],
)
class TestCumsumGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_cumsum_grad_comp(self):
        def actual(primal, cotangent):
            paddle.disable_static()
            core._set_prim_backward_enabled(True)
            x = paddle.to_tensor(
                primal, dtype=primal.dtype, stop_gradient=False
            )
            x.stop_gradient = False
            v = paddle.to_tensor(
                cotangent, dtype=cotangent.dtype, stop_gradient=False
            )
            y = paddle.cumsum(x, axis=-1)
            return paddle.grad(y, x, v, create_graph=True, retain_graph=True)[0]

        def desired(primal, cotangent):
            paddle.disable_static()
            core._set_prim_backward_enabled(False)
            x = paddle.to_tensor(
                primal, dtype=primal.dtype, stop_gradient=False
            )
            x.stop_gradient = False
            v = paddle.to_tensor(
                cotangent, dtype=cotangent.dtype, stop_gradient=False
            )
            y = paddle.cumsum(x, axis=-1)
            return paddle.grad(y, x, v, create_graph=True, retain_graph=True)[0]

        if (
            paddle.device.get_device() == "cpu"
            and self.primal.dtype == np.float16
        ):
            print("pass cpu+float16 case")
        else:
            np.testing.assert_allclose(
                actual=actual(self.primal, self.cotangent),
                desired=desired(self.primal, self.cotangent),
                rtol=limit[str(self.primal.dtype)]['rtol'],
                atol=limit[str(self.primal.dtype)]['atol'],
            )
        core._set_prim_backward_enabled(False)


if __name__ == '__main__':
    unittest.main()
