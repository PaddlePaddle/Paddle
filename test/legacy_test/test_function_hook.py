#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _legacy_C_ops


class TestCapture:
    def __init__(self):
        self.list = []


test_cap = TestCapture()


def grad_hook(grad):
    test_cap.list.append(2)

    return grad


class TestBackwardFunctionHookError(unittest.TestCase):
    def test_hook(self):
        input_data = np.ones([4, 4]).astype('float32')

        x = paddle.to_tensor(input_data.astype(np.float32), stop_gradient=False)
        z = paddle.to_tensor(input_data.astype(np.float32), stop_gradient=False)

        y = _legacy_C_ops.sigmoid(x)
        out = _legacy_C_ops.matmul_v2(y, z, 'trans_x', False, 'trans_y', False)

        y.register_hook(grad_hook)

        out.backward()

        assert test_cap.list == [2]


if __name__ == "__main__":
    unittest.main()
