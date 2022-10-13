#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append('..')
from op_test import OpTest
import paddle
from paddle import _C_ops, _legacy_C_ops

paddle.enable_static()


class TestL2LossOp(OpTest):
    """Test squared_l2_norm
    """

    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.square(LA.norm(X))}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=self.max_relative_error)


class TestL2LossDeterministic(unittest.TestCase):

    def check_place(self, place):
        with paddle.fluid.dygraph.guard(place):
            x_np = np.random.rand(5, 11, 13).astype('float32')
            x = paddle.to_tensor(x_np)
            y1 = _legacy_C_ops.squared_l2_norm(x)
            y2 = _legacy_C_ops.squared_l2_norm(x)
            np.testing.assert_allclose(y1.numpy(), y2.numpy())

    def test_main(self):
        self.check_place(paddle.CPUPlace())
        if paddle.is_compiled_with_mlu():
            self.check_place(paddle.device.MLUPlace(0))


if __name__ == "__main__":
    unittest.main()
