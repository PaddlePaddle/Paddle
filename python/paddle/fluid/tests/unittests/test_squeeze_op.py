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


# Correct: General.
class TestSqueezeOp1(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = (0, 2)
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: There is mins axis.
class TestSqueezeOp2(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = (0, -2)
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: No axes input.
class TestSqueezeOp3(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = ()
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Just part of axes be squeezed. 
class TestSqueezeOp4(OpTest):
    def setUp(self):
        ori_shape = (3, 1, 5, 1, 4, 1)
        axes = (1, -1)
        new_shape = (3, 5, 1, 4)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace.
class TestSqueezeOpInplace1(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = (0, 2)
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace. There is mins axis.
class TestSqueezeOpInplace2(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = (0, -2)
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace. No axes input.
class TestSqueezeOpInplace3(OpTest):
    def setUp(self):
        ori_shape = (1, 3, 1, 5)
        axes = ()
        new_shape = (3, 5)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inpalce. Just part of axes be squeezed. 
class TestSqueezeOpInplace4(OpTest):
    def setUp(self):
        ori_shape = (3, 1, 5, 1, 4, 1)
        axes = (1, -1)
        new_shape = (3, 5, 1, 4)

        self.op_type = "squeeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == "__main__":
    unittest.main()
