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

import unittest
import numpy as np

from op_test import OpTest


class TestReshapeOp(OpTest):
    def setUp(self):
        ori_shape = (2, 25)
        new_shape = (5, 10)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer1(OpTest):
    def setUp(self):
        ori_shape = (5, 10)
        new_shape = (5, -1, 5)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(self.attrs["shape"])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInfer2(OpTest):
    def setUp(self):
        ori_shape = (2, 2, 6)
        new_shape = (2, 0, 3, -1)
        infered_shape = (2, 2, 3, -1)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(infered_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpInplace(OpTest):
    def setUp(self):
        ori_shape = (2, 25)
        new_shape = (5, 10)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInferInplace1(OpTest):
    def setUp(self):
        ori_shape = (5, 10)
        new_shape = (5, -1, 5)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpDimInferInplace2(OpTest):
    def setUp(self):
        ori_shape = (2, 2, 6)
        new_shape = (2, 0, 3, -1)
        infered_shape = (2, 2, 3, -1)

        self.op_type = "reshape"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"shape": new_shape}
        self.outputs = {"Out": self.inputs["X"].reshape(infered_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestReshapeOpWithInputShape(OpTest):
    def setUp(self):
        ori_shape = (6, 5)
        new_shape = (0, -1, 5)
        actual_shape = (2, 3, 5)

        self.op_type = "reshape"
        self.inputs = {
            "X": np.random.random(ori_shape).astype("float32"),
            "Shape": np.array(
                actual_shape, dtype="int32")
        }
        self.attrs = {"shape": new_shape}
        self.outputs = {"Out": self.inputs["X"].reshape(actual_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == "__main__":
    unittest.main()
