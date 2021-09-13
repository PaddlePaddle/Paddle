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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.static import Program, program_guard
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.nn.functional as F
import os
import re


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


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x


def get_csr_value(mat, layout, nnz):
    row = mat.shape[0]
    col = mat.shape[1]
    value = np.zeros(nnz)
    ptr = 0
    for i in range(row):
        for j in range(col):
            if layout[i][j] == 1:
                value[ptr] = mat[i][j]
                ptr += 1
    return value


def ref_sparse_attention(q, k, v, offset, columns):
    row = q.shape[0]
    col = q.shape[1]
    mat = np.zeros((row, row))
    # init mat from CSR format
    for cur_row in range(row):
        start_ptr = int(offset[cur_row])
        end_ptr = int(offset[cur_row + 1])
        for ptr in range(start_ptr, end_ptr):
            cur_col = int(columns[ptr])
            mat[cur_row][cur_col] = 1
    # sdd
    a = np.dot(q, k.T) * mat

    # Get nnz of a
    nnz = columns.shape[0]
    a_value = get_csr_value(a, mat, nnz)

    # scale
    head_dim = col
    scaling = float(head_dim)**-0.5
    a = scaling * a

    for i in range(row):
        for j in range(row):
            if mat[i][j] == 0:
                a[i][j] = float('-inf')

    # softmax
    b = softmax(a)
    b_value = get_csr_value(b, mat, nnz)
    # dsd
    result = np.dot(b, v)

    return result, a_value, b_value


def init_csr_format(rows, blocksize):
    block_num = rows / blocksize
    block_last = rows % blocksize
    nnz_num = block_num * blocksize * blocksize + block_last * block_last
    offset = np.zeros(rows + 1)
    columns = np.zeros(int(nnz_num))
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


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than 11.2")
class TestSparseAttentionOp(OpTest):
    def config(self):
        self.q_shape = (1, 1, 8, 8)
        self.k_shape = (1, 1, 8, 8)
        self.v_shape = (1, 1, 8, 8)
        self.blocksize = 2

    def init_kernel_type(self):
        self.dtype = "float64"

    def setUp(self):
        # init tensor
        paddle.enable_static()
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
        noffset, ncolumns = init_csr_format(row, self.blocksize)

        # init CSR format in paddle Tensor
        self.offset = noffset.astype('int32')
        self.columns = ncolumns.astype('int32')

        result, a, b = ref_sparse_attention(nq, nk, nv, noffset, ncolumns)
        result = result.astype(self.dtype)
        result = np.expand_dims(np.expand_dims(result, 0), 0)

        self.inputs = {
            'Q': self.q,
            'K': self.k,
            'V': self.v,
            'offset': self.offset,
            'columns': self.columns
        }
        self.outputs = {'Out': result, 'ResultSdd': a, 'ResultSoftmax': b}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['Q'], 'Out')
        self.check_grad_with_place(self.place, ['K'], 'Out')
        self.check_grad_with_place(self.place, ['V'], 'Out')


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than 11.2")
class TestSparseAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 1, 8, 8)
        self.blocksize = 2
        self.dtype = 'float64'

    # def check_static_result(self, place):
    #     with paddle.static.program_guard(Program(), Program()):
    #         paddle.enable_static()
    #         q = paddle.static.data(name="Q", shape=[1, 1, 16, 16], dtype="float32")
    #         k = paddle.static.data(name="K", shape=[1, 1, 16, 16], dtype="float32")
    #         v = paddle.static.data(name="V", shape=[1, 1, 16, 16], dtype="float32")

    #         result = paddle.fluid.core.sparse_attention([q,k,v,])

    #         x_np = np.random.random([4, 3]).astype("float32")
    #         y_np = np.random.random([3, 4]).astype("float32")

    #         exe = fluid.Executor(self.place)
    #         fetches = exe.run(fluid.default_main_program(),
    #                           feed={"input_x": x_np,
    #                                 "input_y": y_np},
    #                           fetch_list=[result])

    def test_dygraph(self):
        paddle.disable_static()
        rows = self.shape[2]
        offset, columns = init_csr_format(rows, self.blocksize)
        offset = offset.astype('int32')
        columns = columns.astype('int32')
        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        paddle_query = paddle.to_tensor(query, place=self.place)
        paddle_key = paddle.to_tensor(key, place=self.place)
        paddle_value = paddle.to_tensor(value, place=self.place)
        paddle_offset = paddle.to_tensor(offset, place=self.place)
        paddle_colunmns = paddle.to_tensor(columns, place=self.place)

        paddle_result, tmp_sdd, tmp_softmax = paddle.fluid.core.sparse_attention(
            paddle_query, paddle_key, paddle_value, paddle_offset,
            paddle_colunmns)

        query = query.squeeze()
        key = key.squeeze()
        value = value.squeeze()

        numpy_result, tmp_a, tmp_b = ref_sparse_attention(query, key, value,
                                                          offset, columns)
        numpy_result = numpy_result.astype(self.dtype)
        numpy_result = np.expand_dims(np.expand_dims(numpy_result, 0), 0)

        self.assertTrue(
            np.allclose(
                paddle_result.numpy(), numpy_result, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
