# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
class TestUnsqueezeOp(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 2)
        new_shape = (1, 3, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Single input index.
class TestUnsqueezeOp1(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (-1, )
        new_shape = (3, 5, 1)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Mixed input axis.
class TestUnsqueezeOp2(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, -1)
        new_shape = (1, 3, 5, 1)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: There is duplicated axis.
class TestUnsqueezeOp3(OpTest):
    def setUp(self):
        ori_shape = (3, 2, 5)
        axes = (0, 3, 3)
        new_shape = (1, 3, 2, 1, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace.
class TestUnsqueezeOpInplace1(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 2)
        new_shape = (1, 3, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace. There is mins index.
class TestUnsqueezeOpInplace2(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, -2)
        new_shape = (1, 3, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


# Correct: Inplace. There is duplicated axis.
class TestUnsqueezeOpInplace3(OpTest):
    def setUp(self):
        ori_shape = (3, 2, 5)
        axes = (0, 3, 3)
        new_shape = (1, 3, 2, 1, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": True}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


'''
# Error: Output dimension is error.
class TestUnsqueezeOp4(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 3)
        new_shape = (1, 3, 1, 1, 5)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

# Error: Input axis is large than output range.
class TestUnsqueezeOp5(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 4)
        new_shape = (1, 3, 5, 1)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

        def test_check_grad(self):
            self.check_grad(["X"], "Out")

# Error: Input axes is large than Eigen limit.
class TestUnsqueezeOp6(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 2, 10)
        new_shape = (1, 3, 1, 5, 1)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

# Error: Input axes size is large than Eigen limit.
class TestUnsqueezeOp7(OpTest):
    def setUp(self):
        ori_shape = (3, 5)
        axes = (0, 2, 2, 2, 2, 2)
        new_shape = (1, 3, 1, 1, 5, 1)

        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(ori_shape).astype("float32")}
        self.attrs = {"axes": axes, "inplace": False}
        self.outputs = {"Out": self.inputs["X"].reshape(new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")
'''

if __name__ == "__main__":
    unittest.main()
