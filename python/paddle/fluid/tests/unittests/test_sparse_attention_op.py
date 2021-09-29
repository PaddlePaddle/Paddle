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
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle
import os
import re
import platform


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def get_linux_platform():
    if platform.system().lower() == 'windows':
        return 0
    elif platform.system().lower() == 'linux':
        return 1
    else:
        return -1


def get_suitable_env():
    if get_cuda_version() >= 11020 and get_linux_platform() == 1:
        return True
    else:
        return False


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x


def get_csr_value(mat, layout, nnz):
    row, col = mat.shape[0], mat.shape[1]
    value = np.zeros(nnz)
    ptr = 0
    for i in range(row):
        for j in range(col):
            if layout[i][j] == 1:
                value[ptr] = mat[i][j]
                ptr += 1
    return value


def ref_sparse_attention(q, k, v, offset, columns):
    row, col, nnz = q.shape[0], q.shape[1], columns.shape[0]
    mat = np.zeros((row, row))
    for cur_row in range(row):
        start_ptr = int(offset[cur_row])
        end_ptr = int(offset[cur_row + 1])
        for ptr in range(start_ptr, end_ptr):
            cur_col = int(columns[ptr])
            mat[cur_row][cur_col] = 1
    a = np.dot(q, k.T) * mat
    a_value = get_csr_value(a, mat, nnz)
    scaling = float(col)**-0.5
    a = scaling * a
    for i in range(row):
        for j in range(row):
            if mat[i][j] == 0:
                a[i][j] = float('-inf')
    b = softmax(a)
    b_value = get_csr_value(b, mat, nnz)
    result = np.dot(b, v)
    return result, a_value, b_value


def ref_batch_sparse_attention(q, k, v, offset, columns):
    batch_size, num_heads, row, col = q.shape
    nnz = columns.shape[2]
    result = np.zeros((batch_size, num_heads, row, col))
    result_sdd = np.zeros((batch_size, num_heads, nnz))
    result_softmax = np.zeros((batch_size, num_heads, nnz))
    for i in range(batch_size):
        for j in range(num_heads):
            cur_q, cur_k, cur_v, = q[i][j], k[i][j], v[i][j]
            cur_offset, cur_columns = offset[i][j], columns[i][j]
            cur_result, cur_sdd, cur_softmax = ref_sparse_attention(
                cur_q, cur_k, cur_v, cur_offset, cur_columns)
            result[i][j] = cur_result
            result_sdd[i][j], result_softmax[i][j] = cur_sdd, cur_softmax
    return result, result_sdd, result_softmax


def init_csr_format(batch_size, num_heads, rows, blocksize):
    block_num, block_last = rows / blocksize, rows % blocksize
    nnz_num = block_num * blocksize * blocksize + block_last * block_last
    offset = np.zeros(rows + 1)
    columns = np.zeros(int(nnz_num))
    mat = np.zeros((rows, rows))
    for i in range(0, rows, blocksize):
        for x in range(blocksize):
            for y in range(blocksize):
                p_x, p_y = i + x, i + y
                if (p_x < rows) and (p_y < rows):
                    mat[p_x][p_y] = 1
    p_offset, p_column, count = 0, 0, 0
    for i in range(rows):
        for j in range(rows):
            if mat[i][j] != 0:
                count += 1
                columns[p_column] = j
                p_column += 1
        p_offset += 1
        offset[p_offset] = count
    offset = np.expand_dims(np.expand_dims(offset, 0), 0)
    offset = offset.repeat(num_heads, axis=1)
    offset = offset.repeat(batch_size, axis=0)
    columns = np.expand_dims(np.expand_dims(columns, 0), 0)
    columns = columns.repeat(num_heads, axis=1)
    columns = columns.repeat(batch_size, axis=0)
    return offset, columns


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_suitable_env() == False,
    "core is not compiled with CUDA and cuda version need >= 11.2 in windows")
class TestSparseAttentionOp(OpTest):
    def config(self):
        self.shape = (1, 1, 16, 8)
        self.blocksize = 2
        self.dtype = "float64"

    def setUp(self):
        paddle.enable_static()
        self.config()
        self.op_type = "sparse_attention"
        self.place = paddle.CUDAPlace(0)
        self.q = np.random.random(self.shape).astype(self.dtype)
        self.k = np.random.random(self.shape).astype(self.dtype)
        self.v = np.random.random(self.shape).astype(self.dtype)
        offset, columns = init_csr_format(self.shape[0], self.shape[1],
                                          self.shape[2], self.blocksize)
        self.offset = offset.astype('int32')
        self.columns = columns.astype('int32')

        result, result_sdd, result_softmax = ref_batch_sparse_attention(
            self.q, self.k, self.v, self.offset, self.columns)

        self.inputs = {
            'Q': self.q,
            'K': self.k,
            'V': self.v,
            'Offset': self.offset,
            'Columns': self.columns
        }
        self.outputs = {
            'Out': result.astype(self.dtype),
            'SparseDotSdd': result_sdd.astype(self.dtype),
            'Softmax': result_softmax.astype(self.dtype)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['Q'], 'Out')
        self.check_grad_with_place(self.place, ['K'], 'Out')
        self.check_grad_with_place(self.place, ['V'], 'Out')


class TestSparseAttentionOpFp32Test(TestSparseAttentionOp):
    def config(self):
        self.shape = (1, 1, 8, 16)
        self.blocksize = 2
        self.dtype = "float32"


class TestSparseAttentionOpShapeTest(TestSparseAttentionOp):
    def config(self):
        self.shape = (2, 2, 32, 8)
        self.blocksize = 8
        self.dtype = "float64"


if __name__ == '__main__':
    unittest.main()
