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

import copy
import os
import re
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


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


def masked_fill(x):
    row, col = x.shape[0], x.shape[1]
    for i in range(row):
        for j in range(col):
            if x[i][j] == 0:
                x[i][j] = float('-inf')
    return x


def init_mask(x):
    row, col = x.shape[0], x.shape[1]
    for i in range(row):
        for j in range(col):
            if x[i][j] == 0 and (j < 0.8 * col):
                x[i][j] = 1
    return x


def softmax(x, kp_mask=None, attn_mask=None, bsz=None):
    if kp_mask is None and attn_mask is None:
        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x
    else:
        # kp_mask
        current_kp_mask = kp_mask[bsz]
        row = current_kp_mask.shape[0]
        current_kp_mask = np.expand_dims(current_kp_mask, 0).repeat(row, axis=0)
        # attn_mask
        current_attn_mask = copy.deepcopy(attn_mask)
        current_attn_mask = masked_fill(current_attn_mask)
        current_kp_mask = masked_fill(current_kp_mask)
        x = x + current_kp_mask
        x = x + current_attn_mask
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


def ref_sparse_attention(
    q, k, v, offset, columns, kp_mask=None, attn_mask=None, bsz=None
):
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
    scaling = float(col) ** -0.5
    a = scaling * a
    for i in range(row):
        for j in range(row):
            if mat[i][j] == 0:
                a[i][j] = float('-inf')
    # softmax
    if kp_mask is None and attn_mask is None:
        b = softmax(a)
    else:
        b = softmax(a, kp_mask=kp_mask, attn_mask=attn_mask, bsz=bsz)
    b_value = get_csr_value(b, mat, nnz)
    result = np.dot(b, v)
    return result, a_value, b_value


def ref_batch_sparse_attention(
    q, k, v, offset, columns, kp_mask=None, attn_mask=None
):
    batch_size, num_heads, row, col = q.shape
    nnz = columns.shape[2]
    result = np.zeros((batch_size, num_heads, row, col))
    result_sdd = np.zeros((batch_size, num_heads, nnz))
    result_softmax = np.zeros((batch_size, num_heads, nnz))
    for i in range(batch_size):
        for j in range(num_heads):
            (
                cur_q,
                cur_k,
                cur_v,
            ) = (
                q[i][j],
                k[i][j],
                v[i][j],
            )
            cur_offset, cur_columns = offset[i][j], columns[i][j]
            if kp_mask is None and attn_mask is None:
                cur_result, cur_sdd, cur_softmax = ref_sparse_attention(
                    cur_q, cur_k, cur_v, cur_offset, cur_columns
                )
            else:
                cur_result, cur_sdd, cur_softmax = ref_sparse_attention(
                    cur_q,
                    cur_k,
                    cur_v,
                    cur_offset,
                    cur_columns,
                    kp_mask=kp_mask,
                    attn_mask=attn_mask,
                    bsz=i,
                )
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


def api_wrapper(
    q, k, v, offset, columns, key_padding_mask=None, attn_mask=None
):
    return paddle._C_ops.sparse_attention(
        q, k, v, offset, columns, key_padding_mask, attn_mask
    )


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestSparseAttentionOp(OpTest):
    def config(self):
        self.shape = (1, 1, 16, 16)
        self.blocksize = 4
        self.dtype = "float64"
        self.use_mask = True

    def setUp(self):
        paddle.enable_static()
        self.config()
        self.op_type = "sparse_attention"
        self.python_api = api_wrapper
        self.python_out_sig = ['Out']
        self.place = paddle.CUDAPlace(0)
        self.q = np.random.random(self.shape).astype(self.dtype)
        self.k = np.random.random(self.shape).astype(self.dtype)
        self.v = np.random.random(self.shape).astype(self.dtype)
        # init CSR tensor
        offset, columns = init_csr_format(
            self.shape[0], self.shape[1], self.shape[2], self.blocksize
        )
        self.offset = offset.astype('int32')
        self.columns = columns.astype('int32')
        # init mask tensor
        key_padding_mask_shape = (self.shape[0], self.shape[2])
        attn_mask_shape = (self.shape[2], self.shape[2])
        key_padding_mask = np.random.randint(0, 2, size=key_padding_mask_shape)
        attn_mask = np.random.randint(0, 2, size=attn_mask_shape)
        key_padding_mask = init_mask(key_padding_mask)
        attn_mask = init_mask(attn_mask)

        self.key_padding_mask = key_padding_mask.astype(self.dtype)
        self.attn_mask = attn_mask.astype(self.dtype)
        if self.use_mask:
            result, result_sdd, result_softmax = ref_batch_sparse_attention(
                self.q,
                self.k,
                self.v,
                self.offset,
                self.columns,
                kp_mask=self.key_padding_mask,
                attn_mask=self.attn_mask,
            )
        else:
            result, result_sdd, result_softmax = ref_batch_sparse_attention(
                self.q, self.k, self.v, self.offset, self.columns
            )

        if self.use_mask:
            self.inputs = {
                'Q': self.q,
                'K': self.k,
                'V': self.v,
                'Offset': self.offset,
                'Columns': self.columns,
                'KeyPaddingMask': self.key_padding_mask,
                'AttnMask': self.attn_mask,
            }
        else:
            self.inputs = {
                'Q': self.q,
                'K': self.k,
                'V': self.v,
                'Offset': self.offset,
                'Columns': self.columns,
            }
        self.outputs = {
            'Out': result.astype(self.dtype),
            'SparseDotSdd': result_sdd.astype(self.dtype),
            'Softmax': result_softmax.astype(self.dtype),
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
        self.use_mask = False


class TestSparseAttentionOpShapeTest(TestSparseAttentionOp):
    def config(self):
        self.shape = (2, 2, 32, 8)
        self.blocksize = 8
        self.dtype = "float64"
        self.use_mask = False


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.3",
)
class TestSparseAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 1, 8, 4)
        self.blocksize = 2
        self.dtype = 'float64'
        self.use_mask = True

    def test_static_graph(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            Q = paddle.static.data(name="Q", shape=self.shape, dtype=self.dtype)
            K = paddle.static.data(name="K", shape=self.shape, dtype=self.dtype)
            V = paddle.static.data(name="V", shape=self.shape, dtype=self.dtype)

            batch_size, num_heads, rows = (
                self.shape[0],
                self.shape[1],
                self.shape[2],
            )
            block_num = rows / self.blocksize
            block_last = rows % self.blocksize
            sparse_nnz_num = (
                block_num * self.blocksize * self.blocksize
                + block_last * block_last
            )
            offset_shape = (batch_size, num_heads, rows + 1)
            columns_shape = (batch_size, num_heads, int(sparse_nnz_num))

            offset = paddle.static.data(
                name="Offset", shape=offset_shape, dtype="int32"
            )
            columns = paddle.static.data(
                name="Columns", shape=columns_shape, dtype="int32"
            )
            key_padding_mask_shape = (self.shape[0], self.shape[2])
            attn_mask_shape = (self.shape[2], self.shape[2])
            if self.use_mask:
                key_padding_mask = paddle.static.data(
                    name="KeyPaddingMask",
                    shape=key_padding_mask_shape,
                    dtype=self.dtype,
                )
                attn_mask = paddle.static.data(
                    name="AttnMask", shape=attn_mask_shape, dtype=self.dtype
                )
                Out = F.sparse_attention(
                    Q,
                    K,
                    V,
                    offset,
                    columns,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            else:
                Out = F.sparse_attention(Q, K, V, offset, columns)

            Q_np = np.random.random(self.shape).astype(self.dtype)
            K_np = np.random.random(self.shape).astype(self.dtype)
            V_np = np.random.random(self.shape).astype(self.dtype)
            offset_np, columns_np = init_csr_format(
                self.shape[0], self.shape[1], self.shape[2], self.blocksize
            )
            offset_np = offset_np.astype('int32')
            columns_np = columns_np.astype('int32')

            # init mask tensor
            key_padding_mask_np = np.random.randint(
                0, 2, size=key_padding_mask_shape
            )
            attn_mask_np = np.random.randint(0, 2, size=attn_mask_shape)
            key_padding_mask_np = init_mask(key_padding_mask_np)
            attn_mask_np = init_mask(attn_mask_np)
            key_padding_mask_np = key_padding_mask_np.astype(self.dtype)
            attn_mask_np = attn_mask_np.astype(self.dtype)

            exe = base.Executor(self.place)
            if self.use_mask:
                fetches_result = exe.run(
                    feed={
                        "Q": Q_np,
                        "K": K_np,
                        "V": V_np,
                        "Offset": offset_np,
                        "Columns": columns_np,
                        'KeyPaddingMask': key_padding_mask_np,
                        'AttnMask': attn_mask_np,
                    },
                    fetch_list=[Out],
                )
                expected_result, __, __ = ref_batch_sparse_attention(
                    Q_np,
                    K_np,
                    V_np,
                    offset_np,
                    columns_np,
                    kp_mask=key_padding_mask_np,
                    attn_mask=attn_mask_np,
                )
            else:
                fetches_result = exe.run(
                    feed={
                        "Q": Q_np,
                        "K": K_np,
                        "V": V_np,
                        "Offset": offset_np,
                        "Columns": columns_np,
                    },
                    fetch_list=[Out],
                )
                expected_result, __, __ = ref_batch_sparse_attention(
                    Q_np, K_np, V_np, offset_np, columns_np
                )

            np.testing.assert_allclose(
                fetches_result[0], expected_result, rtol=1e-05, atol=1e-05
            )

    def test_dygraph(self):
        paddle.disable_static()
        offset, columns = init_csr_format(
            self.shape[0], self.shape[1], self.shape[2], self.blocksize
        )
        offset = offset.astype('int32')
        columns = columns.astype('int32')
        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)
        # init mask tensor
        key_padding_mask_shape = (self.shape[0], self.shape[2])
        attn_mask_shape = (self.shape[2], self.shape[2])
        key_padding_mask = np.random.randint(0, 2, size=key_padding_mask_shape)
        attn_mask = np.random.randint(0, 2, size=attn_mask_shape)
        key_padding_mask = init_mask(key_padding_mask)
        attn_mask = init_mask(attn_mask)
        key_padding_mask = key_padding_mask.astype(self.dtype)
        attn_mask = attn_mask.astype(self.dtype)

        paddle_query = paddle.to_tensor(query, place=self.place)
        paddle_key = paddle.to_tensor(key, place=self.place)
        paddle_value = paddle.to_tensor(value, place=self.place)
        paddle_offset = paddle.to_tensor(offset, place=self.place)
        paddle_columns = paddle.to_tensor(columns, place=self.place)
        paddle_kp_mask = paddle.to_tensor(key_padding_mask, place=self.place)
        paddle_attn_mask = paddle.to_tensor(attn_mask, place=self.place)

        if self.use_mask:
            paddle_result = F.sparse_attention(
                paddle_query,
                paddle_key,
                paddle_value,
                paddle_offset,
                paddle_columns,
                key_padding_mask=paddle_kp_mask,
                attn_mask=paddle_attn_mask,
            )

            numpy_result, __, __ = ref_batch_sparse_attention(
                query,
                key,
                value,
                offset,
                columns,
                kp_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            numpy_result = numpy_result.astype(self.dtype)
        else:
            paddle_result = F.sparse_attention(
                paddle_query,
                paddle_key,
                paddle_value,
                paddle_offset,
                paddle_columns,
            )

            numpy_result, __, __ = ref_batch_sparse_attention(
                query, key, value, offset, columns
            )
            numpy_result = numpy_result.astype(self.dtype)

        np.testing.assert_allclose(
            paddle_result.numpy(), numpy_result, rtol=1e-05, atol=1e-05
        )


class TestSparseAttentionAPITestFloat(TestSparseAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 2, 8, 4)
        self.blocksize = 2
        self.dtype = 'float32'
        self.use_mask = False


class TestSparseAttentionAPITestShape1(TestSparseAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 2, 64, 32)
        self.blocksize = 2
        self.dtype = 'float64'
        self.use_mask = False


class TestSparseAttentionAPITestShape2(TestSparseAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 1, 64, 32)
        self.blocksize = 2
        self.dtype = 'float64'
        self.use_mask = False


class TestSparseAttentionAPITestShape3(TestSparseAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (4, 4, 128, 32)
        self.blocksize = 8
        self.dtype = 'float64'
        self.use_mask = False


class TestSparseAttentionAPITestShape4(TestSparseAttentionAPI):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (3, 3, 35, 15)
        self.blocksize = 3
        self.dtype = 'float64'
        self.use_mask = False


if __name__ == '__main__':
    unittest.main()
