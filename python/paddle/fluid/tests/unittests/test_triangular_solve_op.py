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
# limitations under the License.w

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import sys
sys.path.append("..")
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

np.random.seed(2021)
paddle.enable_static()


class TestTriangularSolveOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = [15, 15]
        self.y_shape = [15, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "triangular_solve"
        self.config()
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }
        self.attrs = {
            'upper': self.upper,
            'transpose': self.transpose,
            'unitriangular': self.unitriangular,
        }
        self.outputs = {
            'Out': np.linalg.solve(self.inputs['X'], self.inputs['Y'])
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestTriangularSolveOp2(TestTriangularSolveOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = [6, 10, 10]
        self.y_shape = [10, 15]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"


class TestTriangularSolveOp3(TestTriangularSolveOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = [1, 6, 6]
        self.y_shape = [6, 6, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = True
        self.dtype = "float64"


class TestTriangularSolveOp4(TestTriangularSolveOp):
    """
    case 4
    """

    def config(self):
        self.x_shape = [6, 4, 4]
        self.y_shape = [1, 4, 3]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"


class TestTriangularSolveOp5(TestTriangularSolveOp):
    """
    case 5
    """

    def config(self):
        self.x_shape = [6, 12, 12]
        self.y_shape = [12, 5]
        self.upper = False
        self.transpose = True
        self.unitriangular = True
        self.dtype = "float32"


class TestTriangularSolveOp6(TestTriangularSolveOp):
    """
    case 6
    """

    def config(self):
        self.x_shape = [1, 6, 9, 9]
        self.y_shape = [3, 1, 9, 7]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"


class TestTriangularSolveOp7(TestTriangularSolveOp):
    """
    case 7
    """

    def config(self):
        self.x_shape = [6, 9, 9]
        self.y_shape = [3, 1, 9, 7]
        self.upper = True
        self.transpose = True
        self.unitriangular = True
        self.dtype = "float64"


class TestTriangularSolveOp8(TestTriangularSolveOp):
    """
    case 8
    """

    def config(self):
        self.x_shape = [7, 3, 3]
        self.y_shape = [6, 3, 1, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp9(TestTriangularSolveOp):
    """
    case 9
    """

    def config(self):
        self.x_shape = [6, 1, 6, 5, 5]
        self.y_shape = [6, 3, 1, 5, 6]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp10(TestTriangularSolveOp):
    """
    case 10
    """

    def config(self):
        self.x_shape = [7, 3, 3]
        self.y_shape = [6, 3, 1, 3, 1]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp11(TestTriangularSolveOp):
    """
    case 11
    """

    def config(self):
        self.x_shape = [6, 9, 9]
        self.y_shape = [3, 3, 1, 9, 7]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp12(TestTriangularSolveOp):
    """
    case 12
    """

    def config(self):
        self.x_shape = [5, 3, 7, 9, 9]
        self.y_shape = [3, 1, 9, 9]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp13(TestTriangularSolveOp):
    """
    case 13
    """

    def config(self):
        self.x_shape = [256, 256]
        self.y_shape = [3, 256, 512]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp14(TestTriangularSolveOp):
    """
    case 13
    """

    def config(self):
        self.x_shape = [256, 256]
        self.y_shape = [3, 256, 512]
        self.upper = True
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp15(TestTriangularSolveOp):
    """
    case 14
    """

    def config(self):
        self.x_shape = [3, 5, 6, 7, 7]
        self.y_shape = [3, 1, 1, 7, 4]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float32"


class TestTriangularSolveOp16(TestTriangularSolveOp):
    """
    case 15
    """

    def config(self):
        self.x_shape = [3, 1, 1, 3, 3]
        self.y_shape = [6, 3, 3, 1]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float32"


#--------------------test exception-------------------
#TODO: test complex

#--------------------test exception-------------------
#TODO: test exception, singular check

if __name__ == "__main__":
    unittest.main()
