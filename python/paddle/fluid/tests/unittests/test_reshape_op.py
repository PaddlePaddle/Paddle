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

from __future__ import print_function

import unittest
import numpy as np

from op_test import OpTest


class TestReshapeOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 25)
        self.new_shape = (5, 10)
        self.infered_shape = (5, 10)

    def test_check_output(self):

        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)


class TestReshapeOpWithInputShape(OpTest):
    def setUp(self):
        ori_shape = (6, 5)
        new_shape = (0, -1, 5)
        actual_shape = (2, 3, 5)

        self.op_type = "reshape2"
        self.inputs = {
            "X": np.random.random(ori_shape).astype("float32"),
            "Shape": np.array(
                actual_shape, dtype="int32")
        }
        self.attrs = {"shape": new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(actual_shape),
            'XShape': np.random.random(ori_shape).astype("float32")
        }

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOp_attr_tensor(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"

        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            "X": np.random.random(self.ori_shape).astype("float32"),
            'ShapeTensor': shape_tensor
        }
        self.attrs = {}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            'XShape': np.random.random(self.ori_shape).astype("float32")
        }

    def init_data(self):
        self.ori_shape = (2, 25)
        self.new_shape = (5, 10)
        self.infered_shape = (5, 10)

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1_attr_tensor(TestReshapeOp_attr_tensor):
    def init_data(self):
        self.ori_shape = (5, 10)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2_attr_tensor(TestReshapeOp_attr_tensor):
    def init_data(self):
        self.ori_shape = (2, 2, 6)
        self.new_shape = (2, 0, 3, -1)
        self.infered_shape = (2, 2, 3, -1)


if __name__ == "__main__":
    unittest.main()
