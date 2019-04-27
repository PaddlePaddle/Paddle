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


def generate_compatible_shapes(dim_X, dim_Y, transpose_X, transpose_Y):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    if (dim_X == 1 and transpose_X) or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, 1))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((1, Y.size))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


class Generator(object):
    def setUp(self):
        self.op_type = "matmul"
        self.last_dim = [-1, -1, -1]
        self.X = np.random.random(self.shape_X).astype("float32")
        self.Y = np.random.random(self.shape_Y).astype("float32")
        self.Out = reference_matmul(self.X, self.Y, self.transpose_X,
                                    self.transpose_Y)
        if self.is_test:
            if self.output_mode and len(self.Out.shape) == 4:
                self.init_out_test_case()
            if self.input_mode & 1:
                self.init_x_test_case()
            if self.input_mode & 2:
                self.init_y_test_case()
        self.inputs = {'X': self.X, 'Y': self.Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y,
            'last_dim': self.last_dim,
            'is_test': self.is_test,
        }
        self.outputs = {'Out': self.Out}

    def test_check_output(self):
        self.check_output(atol=1e-3)

    def test_check_grad_normal(self):
        if self.is_test:
            pass
        else:
            self.check_grad(['X', 'Y'], 'Out', max_relative_error=1e-3)

    def test_check_grad_ignore_x(self):
        if self.is_test:
            pass
        else:
            self.check_grad(
                ['Y'], 'Out', max_relative_error=1e-3, no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        if self.is_test:
            pass
        else:
            self.check_grad(
                ['X'], 'Out', max_relative_error=1e-3, no_grad_set=set('Y'))

    def init_x_test_case(self):
        length = len(self.shape_X)
        axis = np.arange(length - 3)
        np.random.shuffle(axis)
        axis = np.append(axis, [length - 2, length - 3, length - 1])
        self.X = self.X.transpose(axis)
        self.axis_X = np.arange(length)
        for index in range(length):
            for item in range(length):
                if axis[item] == index:
                    self.axis_X[index] = item
        shape = np.array(self.shape_X)
        shape = list(shape[axis])
        reshape_X = np.zeros(length, dtype=np.int)
        for index in range(length - 2, length):
            reshape_X[index] = shape[index]
        self.last_dim[0] = reshape_X[length - 1]
        shape[length - 2] *= shape[length - 1]
        shape = shape[0:length - 1]
        self.X = self.X.reshape(shape)

    def init_y_test_case(self):
        length = len(self.shape_Y)
        axis = np.arange(length - 3)
        np.random.shuffle(axis)
        axis = np.append(axis, [length - 2, length - 3, length - 1])
        self.Y = self.Y.transpose(axis)
        self.axis_Y = np.arange(length)
        for index in range(length):
            for item in range(length):
                if axis[item] == index:
                    self.axis_Y[index] = item
        shape = np.array(self.shape_Y)
        shape = list(shape[axis])
        reshape_Y = np.zeros(length, dtype=np.int)
        for index in range(length - 2, len(shape)):
            reshape_Y[index] = shape[index]
        self.last_dim[1] = reshape_Y[length - 1]
        shape[length - 2] *= shape[length - 1]
        shape = shape[0:length - 1]
        self.Y = self.Y.reshape(shape)

    def init_out_test_case(self):
        length = len(self.Out.shape)
        self.axis_Out = [0, 2, 1, 3]
        self.Out = self.Out.transpose(self.axis_Out)
        shape = np.array(self.Out.shape)
        reshape_Out = np.zeros(length - 1, dtype=np.int)
        reshape_Out[length - 2] = shape[length - 2] * shape[length - 1]
        self.last_dim[2] = reshape_Out[length - 2]
        shape[length - 2] *= shape[length - 1]
        shape = shape[0:length - 1]
        self.Out = self.Out.reshape(shape)


# Generate test cases for all possibilities
def inject_test(dim_x, dim_y, trans_x, trans_y, input_mode, output_mode,
                is_test):
    test_name = (
        'TestFusedMatmulReshapeTransposeOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}_input_mode_{}_output_mode_{}_is_test_{}'.
        format(dim_x, dim_y, trans_x, trans_y, input_mode, output_mode,
               is_test))
    shape_x, shape_y = generate_compatible_shapes(dim_x, dim_y, trans_x,
                                                  trans_y)
    if (dim_x != 4 or dim_y != 4) and is_test:
        return
    globals()[test_name] = type(test_name, (Generator, OpTest), {
        'shape_X': shape_x,
        'shape_Y': shape_y,
        'transpose_X': trans_x,
        'transpose_Y': trans_y,
        'input_mode': input_mode,
        'output_mode': output_mode,
        'is_test': is_test,
    })


for dim_X in (1, 2, 3, 4):
    for dim_Y in (1, 2, 3, 4):
        for transose_x in (False, True):
            for transose_y in (False, True):
                for input_mode in (1, 2, 3):
                    for output_mode in (False, True):
                        for is_test in (False, True):
                            inject_test(dim_X, dim_Y, transose_x, transose_y,
                                        input_mode, output_mode, is_test)


# Test case n-dim
def generate_compatible_shapes(dim, transpose_X, transpose_Y):
    M = 2
    N = 4
    K = 3
    shape_X = [2 for _ in range(dim - 2)]
    shape_Y = [2 for _ in range(dim - 2)]

    if transpose_X:
        shape_X += [K, M]
    else:
        shape_X += [M, K]

    if transpose_Y:
        shape_Y += [N, K]
    else:
        shape_Y += [K, N]

    return shape_X, shape_Y


# # Test case n-dim
for dim in [4]:
    for transpose_X in [False, True]:
        for transpose_Y in [False, True]:
            for input_mode in (1, 2, 3):
                for output_mode in (False, True):
                    for is_test in (False, True):
                        test_name = (
                            'TestFusedMatmulReshapeTransposeOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}_input_mode_{}_output_mode_{}_is_test_{}'.
                            format(dim, dim, transpose_X, transpose_Y,
                                   input_mode, output_mode, is_test))
                        shape_X, shape_Y = generate_compatible_shapes(
                            dim, transpose_X, transpose_Y)
                        globals()[test_name] = type(
                            test_name, (Generator, OpTest), {
                                'shape_X': shape_X,
                                'shape_Y': shape_Y,
                                'transpose_X': transpose_X,
                                'transpose_Y': transpose_Y,
                                'input_mode': input_mode,
                                'output_mode': output_mode,
                                'is_test': is_test,
                            })

if __name__ == "__main__":
    unittest.main()
