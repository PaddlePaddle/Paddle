# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.nn.functional as F
import unittest
import numpy as np
import six
import paddle
from op_test import OpTest
from paddle.fluid.layers import core


def fill_diagonal_ndarray(x, value, offset=0, dim1=0, dim2=1):
    """Fill value into the diagonal of x that offset is ${offset} and the coordinate system is (dim1, dim2)."""
    strides = x.strides
    shape = x.shape
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
    assert 0 <= dim1 < dim2 <= 2
    assert len(x.shape) == 3

    dim_sum = dim1 + dim2
    dim3 = len(x.shape) - dim_sum
    if offset >= 0:
        diagdim = min(shape[dim1], shape[dim2] - offset)
        diagonal = np.lib.stride_tricks.as_strided(
            x[:, offset:] if dim_sum == 1 else x[:, :, offset:],
            shape=(shape[dim3], diagdim),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))
    else:
        diagdim = min(shape[dim2], shape[dim1] + offset)
        diagonal = np.lib.stride_tricks.as_strided(
            x[-offset:, :] if dim_sum in [1, 2] else x[:, -offset:],
            shape=(shape[dim3], diagdim),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))

    diagonal[...] = value
    return x


def fill_gt(x, y, offset, dim1, dim2):
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset
    xshape = x.shape
    yshape = y.shape
    if len(xshape) != 3:
        perm_list = []
        unperm_list = [0] * len(xshape)
        idx = 0

        for i in range(len(xshape)):
            if i != dim1 and i != dim2:
                perm_list.append(i)
                unperm_list[i] = idx
                idx += 1
        perm_list += [dim1, dim2]
        unperm_list[dim1] = idx
        unperm_list[dim2] = idx + 1

        x = np.transpose(x, perm_list)
        y = y.reshape(-1, yshape[-1])
        nxshape = x.shape
        x = x.reshape((-1, xshape[dim1], xshape[dim2]))
    out = fill_diagonal_ndarray(x, y, offset, 1, 2)

    if len(xshape) != 3:
        out = out.reshape(nxshape)
        out = np.transpose(out, unperm_list)
    return out


class TensorFillDiagTensor_Test(OpTest):
    def setUp(self):
        self.op_type = "fill_diagonal_tensor"
        self.init_kernel_type()
        x = np.random.random((10, 10)).astype(self.dtype)
        y = np.random.random((10, )).astype(self.dtype)
        dim1 = 0
        dim2 = 1
        offset = 0
        out = fill_gt(x, y, offset, dim1, dim2)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {'Out': out}
        self.attrs = {"dim1": dim1, "dim2": dim2, "offset": offset}

    def init_kernel_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TensorFillDiagTensor_Test2(TensorFillDiagTensor_Test):
    def setUp(self):
        self.op_type = "fill_diagonal_tensor"
        self.init_kernel_type()
        x = np.random.random((2, 20, 25)).astype(self.dtype)
        y = np.random.random((2, 20)).astype(self.dtype)
        dim1 = 2
        dim2 = 1
        offset = -3
        out = fill_gt(x, y, offset, dim1, dim2)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {'Out': out}
        self.attrs = {"dim1": dim1, "dim2": dim2, "offset": offset}

    def init_kernel_type(self):
        self.dtype = np.float32


class TensorFillDiagTensor_Test3(TensorFillDiagTensor_Test):
    def setUp(self):
        self.op_type = "fill_diagonal_tensor"
        self.init_kernel_type()
        x = np.random.random((2, 20, 20, 3)).astype(self.dtype)
        y = np.random.random((2, 3, 18)).astype(self.dtype)
        dim1 = 1
        dim2 = 2
        offset = 2
        out = fill_gt(x, y, offset, dim1, dim2)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {'Out': out}
        self.attrs = {"dim1": dim1, "dim2": dim2, "offset": offset}

    def init_kernel_type(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
