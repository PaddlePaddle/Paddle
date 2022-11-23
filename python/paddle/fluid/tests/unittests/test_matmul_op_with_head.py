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


def generate_compatible_shapes_mul_head(dim_X, dim_Y, transpose_X, transpose_Y):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 24
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


def matmul_head(X, Y, head_number=1):
    x = []
    y = []
    z = []
    sub_x_width = X.shape[-1] // head_number
    sub_y_height = Y.shape[-2] // head_number
    if np.ndim(X) == 2:
        for i in range(0, head_number):
            x.append(X[:, i * sub_x_width:i * sub_x_width + sub_x_width])
            y.append(Y[i * sub_y_height:i * sub_y_height + sub_y_height, :])
        for i in range(0, head_number):
            z.append(np.matmul(x[i], y[i]))
        Z = np.concatenate((z), axis=1)

    elif np.ndim(X) == 3:
        for i in range(0, head_number):
            x.append(X[:, :, i * sub_x_width:i * sub_x_width + sub_x_width])
            y.append(Y[:, i * sub_y_height:i * sub_y_height + sub_y_height, :])
        for i in range(0, head_number):
            z.append(np.matmul(x[i], y[i]))
        Z = np.concatenate((z), axis=2)
    else:
        print("ERROR: Not supported dimension")

    return Z


def transpose_mat(X):
    if X.ndim >= 2:
        dim = np.arange(X.ndim)
        dim[[-1, -2]] = dim[[-2, -1]]
        X = np.transpose(X, tuple(dim))

    return X


def reference_matmul_mul_head(X,
                              Y,
                              head_number=1,
                              transpose_X=False,
                              transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        X = transpose_mat(X)
    if transpose_Y:
        Y = transpose_mat(Y)

    Out = matmul_head(X, Y, head_number)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


# Generator for multiple head
class GeneratorMulHead(object):

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random(self.shape_X).astype("float32")
        Y = np.random.random(self.shape_Y).astype("float32")
        Out = reference_matmul_mul_head(X, Y, 4, self.transpose_X,
                                        self.transpose_Y)

        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y,
            'head_number': self.head_number
        }
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()


def inject_test_multiple_head(dim_x, dim_y, trans_x, trans_y, head_number):
    test_name = (
        'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}_head_{}'.format(
            dim_x, dim_y, trans_x, trans_y, head_number))
    shape_x, shape_y = generate_compatible_shapes_mul_head(
        dim_x, dim_y, trans_x, trans_y)
    globals()[test_name] = type(
        test_name, (GeneratorMulHead, OpTest), {
            'shape_X': shape_x,
            'shape_Y': shape_y,
            'transpose_X': trans_x,
            'transpose_Y': trans_y,
            'head_number': head_number
        })


def matmul_head2(X, Y, head_number=1):
    x = []
    y = []
    z = []
    sub_x_width = X.shape[-1] // head_number
    sub_y_width = Y.shape[-1] // head_number
    assert (sub_x_width == Y.shape[-2]
            ), "Error: incompatible head number or matrix size!"
    if np.ndim(X) == 2:
        for i in range(0, head_number):
            x.append(X[:, i * sub_x_width:i * sub_x_width + sub_x_width])
            y.append(Y[:, i * sub_y_width:i * sub_y_width + sub_y_width])
        for i in range(0, head_number):
            z.append(np.matmul(x[i], y[i]))
        Z = np.concatenate((z), axis=1)

    elif np.ndim(X) == 3:
        for i in range(0, head_number):
            x.append(X[:, :, i * sub_x_width:i * sub_x_width + sub_x_width])
            y.append(Y[:, :, i * sub_y_width:i * sub_y_width + sub_y_width])
        for i in range(0, head_number):
            z.append(np.matmul(x[i], y[i]))
        Z = np.concatenate((z), axis=2)
    else:
        assert False, "ERROR: Not supported dimension!"
    return Z


def reference_matmul_mul_head2(X,
                               Y,
                               head_number=1,
                               transpose_X=False,
                               transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        X = transpose_mat(X)
    if transpose_Y:
        Y = transpose_mat(Y)

    Out = matmul_head2(X, Y, head_number)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


def generate_compatible_shapes_mul_head2(dim_X, dim_Y, transpose_X,
                                         transpose_Y):
    BATCH_SIZE = 2
    # Assume head number H is 4. We need make sure K1/H = M2
    M1 = 3
    K1 = 8
    M2 = 2
    K2 = 16

    if dim_X >= 2:
        if transpose_X:
            shape_X = [K1, M1]
        else:
            shape_X = [M1, K1]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [K2, M2]
        else:
            shape_Y = [M2, K2]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


# Generator for multiple head, case 2 when width of X is not same as height of Y
class GeneratorMulHead2(object):

    def setUp(self):
        self.op_type = "matmul"

        X = np.zeros(self.shape_X)
        Y = np.zeros(self.shape_Y)
        if len(self.shape_X) == 2:
            X = np.arange(0,
                          self.shape_X[-1] * self.shape_X[-2],
                          dtype=np.float32).reshape(self.shape_X)
            Y = np.arange(0,
                          self.shape_Y[-1] * self.shape_Y[-2],
                          dtype=np.float32).reshape(self.shape_Y)
        else:
            for i in range(0, len(self.shape_X) - 1):
                X[i, :, :] = np.arange(0,
                                       self.shape_X[-1] * self.shape_X[-2],
                                       dtype=np.float32).reshape(
                                           list(self.shape_X)[-2:])
                Y[i, :, :] = np.arange(0,
                                       self.shape_Y[-1] * self.shape_Y[-2],
                                       dtype=np.float32).reshape(
                                           list(self.shape_Y)[-2:])

        Out = reference_matmul_mul_head2(X, Y, 4, self.transpose_X,
                                         self.transpose_Y)

        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y,
            'head_number': self.head_number
        }
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()


def inject_test_multiple_head2(dim_x, dim_y, trans_x, trans_y, head_number):
    test_name = (
        'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}_head2_{}'.format(
            dim_x, dim_y, trans_x, trans_y, head_number))
    shape_x, shape_y = generate_compatible_shapes_mul_head2(
        dim_x, dim_y, trans_x, trans_y)
    globals()[test_name] = type(
        test_name, (GeneratorMulHead2, OpTest), {
            'shape_X': shape_x,
            'shape_Y': shape_y,
            'transpose_X': trans_x,
            'transpose_Y': trans_y,
            'head_number': head_number
        })


#test case for multiple head
for dim in (2, 3):
    for transose_x in (False, True):
        for transose_y in (False, True):
            inject_test_multiple_head(dim, dim, transose_x, transose_y, 4)

#test case for multiple head when X.width != Y.height
for dim in (2, 3):
    for transose_x in (False, True):
        for transose_y in (False, True):
            inject_test_multiple_head2(dim, dim, transose_x, transose_y, 4)

if __name__ == "__main__":
    unittest.main()
