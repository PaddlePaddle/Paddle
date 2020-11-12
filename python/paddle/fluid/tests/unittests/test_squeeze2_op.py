#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

from op_test import OpTest
import paddle
paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):
    def setUp(self):
        self.op_type = "squeeze2"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float64")
        }

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed. 
class TestSqueezeOp3(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


class TestSqueezeApi(unittest.TestCase):
    def test_squeeze(self):
        paddle.disable_static()
        x = paddle.rand([1, 2, 3, 1])
        y = paddle.squeeze(x)
        y[0] = 2.
        self.assertNotEqual(x.shape, y.shape)

        x_numpy = x.numpy()
        y_numpy = y.numpy()
        self.assertTrue(np.array_equal(x_numpy.squeeze(), y_numpy))
        paddle.enable_static()

    def test_squeeze_(self):
        paddle.disable_static()
        x = paddle.rand([1, 2, 3, 1])
        y = x.squeeze_()
        y[0] = 2.
        self.assertEqual(x.shape, y.shape)

        x_numpy = x.numpy()
        y_numpy = y.numpy()
        self.assertTrue(np.array_equal(x_numpy, y_numpy))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
