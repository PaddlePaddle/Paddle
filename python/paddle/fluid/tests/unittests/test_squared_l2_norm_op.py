#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from numpy import linalg as LA
from op_test import OpTest
import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.framework import in_dygraph_mode


def test_squared_l2_norm(x):
    if in_dygraph_mode():
        return _C_ops.squared_l2_norm(x)
    else:
        return _legacy_C_ops.squared_l2_norm(x)


class TestL2LossOp(OpTest):
    """Test squared_l2_norm
    """

    def setUp(self):
        self.python_api = test_squared_l2_norm
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.square(LA.norm(X))}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'],
                        'Out',
                        max_relative_error=self.max_relative_error,
                        check_eager=True)


class TestL2LossDeterministic(unittest.TestCase):

    def check_place(self, place):
        with paddle.fluid.dygraph.guard(place):
            x_np = np.random.rand(5, 11, 13).astype('float32')
            x = paddle.to_tensor(x_np)
            y1 = _legacy_C_ops.squared_l2_norm(x)
            y2 = _legacy_C_ops.squared_l2_norm(x)
            np.testing.assert_array_equal(y1.numpy(), y2.numpy())

    def test_main(self):
        self.check_place(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.check_place(paddle.CUDAPlace(0))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
