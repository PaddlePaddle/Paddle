#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework

paddle.enable_static()


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x


def ref_sparse_attention(q, k, v, offset, columns):
    row = q.shape[0]
    col = q.shape[1]
    mat = np.zeros((row, col))
    # init mat from CSR format
    for cur_row in range(row):
        start_ptr = int(offset[cur_row])
        end_ptr = int(offset[cur_row + 1])
        for ptr in range(start_ptr, end_ptr):
            cur_col = int(columns[ptr])
            mat[cur_row][cur_col] = 1
    # sdd
    a = np.dot(q, k) * mat
    for i in range(row):
        for j in range(col):
            if mat[i][j] == 0:
                a[i][j] = float('-inf')
    # softmax
    b = softmax(a)
    # dsd
    result = np.dot(b, v)
    return result


def init_csr_format(offset, columns, rows, blocksize):
    mat = np.zeros((rows, rows))
    for i in range(0, rows, blocksize):
        for x in range(blocksize):
            for y in range(blocksize):
                p_x = i + x
                p_y = i + y
                if (p_x < rows) and (p_y < rows):
                    mat[p_x][p_y] = 1

    p_offset = 0
    p_column = 0
    count = 0
    for i in range(rows):
        for j in range(rows):
            if mat[i][j] != 0:
                count += 1
                columns[p_column] = j
                p_column += 1
        p_offset += 1
        offset[p_offset] = count
    return offset, columns


class TestSparseAttentionOp(OpTest):
    def config(self):
        self.q_shape = (1, 1, 64, 64)
        self.k_shape = (1, 1, 64, 64)
        self.v_shape = (1, 1, 64, 64)
        self.blocksize = 16

    def init_kernel_type(self):
        self.dtype = "float32"

    def setUp(self):
        # init tensor
        self.init_kernel_type()
        self.config()
        self.op_type = "sparse_attention"
        self.place = paddle.CUDAPlace(0)
        nq = np.random.random(self.q_shape).astype(self.dtype)
        nk = np.random.random(self.k_shape).astype(self.dtype)
        nv = np.random.random(self.v_shape).astype(self.dtype)
        self.q = nq
        self.k = nk
        self.v = nv
        nq = nq.squeeze()
        nk = nk.squeeze()
        nv = nv.squeeze()

        # init CSR format in numpy
        row = nq.shape[0]
        block_num = row / self.blocksize
        block_last = row % self.blocksize
        nnz_num = block_num * self.blocksize * self.blocksize + block_last * block_last
        noffset = np.zeros(row + 1)
        ncolumns = np.zeros(int(nnz_num))
        noffset, ncolumns = init_csr_format(noffset, ncolumns, row,
                                            self.blocksize)

        # init CSR format in paddle Tensor
        self.offset = noffset.astype('int32')
        self.columns = ncolumns.astype('int32')

        result = ref_sparse_attention(nq, nk, nv, noffset, ncolumns)
        result = result.astype(self.dtype)
        result = np.expand_dims(np.expand_dims(result, 0), 0)

        self.inputs = {
            'X': [('Q', self.q), ('K', self.k), ('V', self.v),
                  ('offset', self.offset), ('columns', self.columns)]
        }
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
