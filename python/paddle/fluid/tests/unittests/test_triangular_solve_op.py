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

import sys
sys.path.append("..")
import paddle
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()

import torch


class TestTriangularSolveOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = [12, 12]
        self.y_shape = [12, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        self.output = np.linalg.solve(
            np.triu(self.inputs['X']), self.inputs['Y'])

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
        self.set_output()
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestTriangularSolveOp2(TestTriangularSolveOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [3, 10, 8]
        self.upper = False
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X']).transpose(1, 0)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp3(TestTriangularSolveOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = [1, 10, 10]
        self.y_shape = [6, 10, 12]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp4(TestTriangularSolveOp):
    """
    case 4
    """

    def config(self):
        self.x_shape = [3, 10, 10]
        self.y_shape = [1, 10, 12]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp5(TestTriangularSolveOp):
    """
    case 5
    """

    def config(self):
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.upper = True
        self.transpose = False
        self.unitriangular = True
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)

    def test_check_grad_normal(self):
        x = np.triu(self.inputs['X'])
        np.fill_diagonal(x, 1.)
        grad_out = np.ones([10, 10]).astype('float64')
        grad_y = np.linalg.solve(x.transpose(1, 0), grad_out)

        grad_x = -np.matmul(grad_y, self.output.transpose(1, 0))
        grad_x = np.triu(grad_x)
        np.fill_diagonal(grad_x, 0.)

        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[grad_x, grad_y],
            user_defined_grad_outputs=[grad_out])


class TestTriangularSolveOp6(TestTriangularSolveOp):
    """
    case 6
    """

    def config(self):
        self.x_shape = [1, 3, 10, 10]
        self.y_shape = [2, 1, 10, 5]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp7(TestTriangularSolveOp):
    """
    case 7
    """

    def config(self):
        self.x_shape = [2, 10, 10]
        self.y_shape = [5, 1, 10, 2]
        self.upper = True
        self.transpose = True
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.triu(self.inputs['X']).transpose(0, 2, 1)
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp8(TestTriangularSolveOp):
    """
    case 8
    """

    def config(self):
        self.x_shape = [12, 3, 3]
        self.y_shape = [2, 3, 12, 3, 2]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.linalg.solve(x, y)


class TestTriangularSolveOp9(TestTriangularSolveOp):
    """
    case 9
    """

    def config(self):
        self.x_shape = [2, 4, 2, 3, 3]
        self.y_shape = [4, 1, 3, 10]
        self.upper = False
        self.transpose = False
        self.unitriangular = False
        self.dtype = "float64"

    def set_output(self):
        x = np.tril(self.inputs['X'])
        y = self.inputs['Y']
        self.output = np.matmul(np.linalg.inv(x), y)


if __name__ == "__main__":
    unittest.main()
