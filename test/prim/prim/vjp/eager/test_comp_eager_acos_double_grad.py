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
    ('primal', 'cotangent', 'dtype'),
    [
        (
            np.random.uniform(-0.95, 0.95, size=(10, 10)),
            np.random.rand(10, 10),
            np.float32,
        ),
    ],
)
class TestAcosDoubleGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    def test_acos_double_grad_comp_dygraph(self):
        paddle.disable_static()
        core.set_prim_eager_enabled(True)

        x = paddle.to_tensor(self.primal, dtype='float32', stop_gradient=False)
        x.stop_gradient = False
        y = paddle.acos(x)
        dx = paddle.grad(y, x, create_graph=True, retain_graph=True)[0]
        ddx = paddle.grad(dx, x, create_graph=True, retain_graph=True)[0]
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
