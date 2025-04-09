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


@param.parameterized_class(
    ('arr', 'indices', 'axis', 'cotangent', 'dtype'),
    [
        (
            np.random.randn(4, 3, 2),  # arr
            np.random.randint(0, 3, size=(1, 3, 2)),  # indices
            1,  # axis
            np.random.rand(1, 3, 2),  # cotangent
            np.float32,  # dtype
        ),
    ],
)
class TestTakeAlongAxisTanhDoubleGrad(unittest.TestCase):
    def test_take_along_axis_tanh_double_grad(self):
        arr_tensor = paddle.to_tensor(
            self.arr, dtype=self.dtype, stop_gradient=False
        )
        indices_tensor = paddle.to_tensor(self.indices, dtype="int64")
        dout_tensor = paddle.to_tensor(
            self.cotangent, dtype=self.dtype, stop_gradient=False
        )
        out = paddle.take_along_axis(arr_tensor, indices_tensor, axis=self.axis)

        out = paddle.tanh(out)

        dx = paddle.grad(out, arr_tensor, dout_tensor, create_graph=True)[0]

        ddx = paddle.grad(dx, dout_tensor, create_graph=True)[0]


if __name__ == '__main__':
    unittest.main()
