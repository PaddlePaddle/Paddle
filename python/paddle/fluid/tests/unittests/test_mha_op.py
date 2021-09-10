# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core


def _get_attn_output(q, k, v, wq, wk, wv, wo, seq_len, nheads, vec_size,
                     sm_scaler):

    proj_size = vec_size // nheads

    q_bar = q.reshape((-1, seq_len, vec_size))
    k_bar = k.reshape((-1, seq_len, vec_size))
    v_bar = v.reshape((-1, seq_len, vec_size))

    q_bar = np.matmul(q, wq).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))
    k_bar = np.matmul(k, wk).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))
    v_bar = np.matmul(v, wv).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))

    beta = np.matmul(q_bar, k_bar.transpose((0, 1, 3, 2))) * sm_scaler
    alpha = _softmax(beta)

    h_bar = np.matmul(alpha, v_bar).transpose((0, 2, 1, 3)).reshape(
        (-1, seq_len, vec_size))
    out = np.matmul(h_bar, wo)
    return out.reshape((-1, 1, seq_len, vec_size))


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=3, keepdims=True))
    return e_x / e_x.sum(axis=3, keepdims=True)


def _generate_data(batch_size, seq_len, vec_size, dtype):
    Q = np.random.random((batch_size, 1, seq_len, vec_size)).astype(dtype)
    K = np.random.random((batch_size, 1, seq_len, vec_size)).astype(dtype)
    V = np.random.random((batch_size, 1, seq_len, vec_size)).astype(dtype)
    W = np.random.random((4 * vec_size * vec_size, )).astype(dtype)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:].reshape((vec_size, vec_size))

    q_seq_arr = np.full((batch_size, ), seq_len, dtype=np.int32)
    k_seq_arr = np.full((batch_size, ), seq_len, dtype=np.int32)

    lo_win = np.zeros((seq_len, ), dtype=int)
    hi_win = np.full((seq_len, ), seq_len, dtype=int)
    return (Q, K, V, W, WQ, WK, WV, WO, q_seq_arr, k_seq_arr, lo_win, hi_win)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.dtype = np.float16
        self.init_dtype_type()

        batch_size = 2
        nheads = 4
        seq_len = 8
        vec_size = 8
        proj_size = vec_size // nheads

        Q, K, V, W, WQ, WK, WV, WO, \
        q_seq_arr, k_seq_arr, \
        lo_win, hi_win = _generate_data(batch_size, seq_len, vec_size, self.dtype)

        self.inputs = {
            'Q': Q,
            'K': K,
            'V': V,
            'W': W,
            'QO_Seqlen': q_seq_arr,
            'KV_Seqlen': k_seq_arr
        }

        self.attrs = {
            'attn_low_windows': lo_win,
            'attn_high_windows': hi_win,
            'attn_dropout_rate': 0.0,
            'attn_heads': nheads,
            'attn_sm_scaler': 1.0,
            'attn_vec_size': vec_size,
            'attn_q_proj_size': proj_size,
            'attn_k_proj_size': proj_size,
            'attn_v_proj_size': proj_size,
            'attn_o_proj_size': vec_size,
            'attn_max_qo_seq_len': seq_len,
            'attn_max_kv_seq_len': seq_len,
            'attn_beam_size': 1
        }
        O = _get_attn_output(Q, K, V, WQ, WK, WV, WO, seq_len, nheads, vec_size,
                             self.attrs["attn_sm_scaler"])
        self.outputs = {'O': O}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=1e-1)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['Q', 'K', 'V', 'W'], 'O', max_relative_error=1.0)


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP32(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.dtype = np.single
        self.init_dtype_type()

        batch_size = 2
        nheads = 4
        seq_len = 8
        vec_size = 8
        proj_size = vec_size // nheads

        Q, K, V, W, WQ, WK, WV, WO, \
        q_seq_arr, k_seq_arr, \
        lo_win, hi_win = _generate_data(batch_size, seq_len, vec_size, self.dtype)

        self.inputs = {
            'Q': Q,
            'K': K,
            'V': V,
            'W': W,
            'QO_Seqlen': q_seq_arr,
            'KV_Seqlen': k_seq_arr
        }

        self.attrs = {
            'attn_low_windows': lo_win,
            'attn_high_windows': hi_win,
            'attn_dropout_rate': 0.0,
            'attn_heads': nheads,
            'attn_sm_scaler': 1.0,
            'attn_vec_size': vec_size,
            'attn_q_proj_size': proj_size,
            'attn_k_proj_size': proj_size,
            'attn_v_proj_size': proj_size,
            'attn_o_proj_size': vec_size,
            'attn_max_qo_seq_len': seq_len,
            'attn_max_kv_seq_len': seq_len,
            'attn_beam_size': 1
        }
        O = _get_attn_output(Q, K, V, WQ, WK, WV, WO, seq_len, nheads, vec_size,
                             self.attrs["attn_sm_scaler"])
        self.outputs = {'O': O}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=2e-1)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['Q', 'K', 'V', 'W'], 'O', max_relative_error=1.0)


# @unittest.skipIf(not core.is_compiled_with_cuda(),
#                  "core is not compiled with CUDA")
# class TestMHAOpFP64(OpTest):
#     def setUp(self):
#         self.op_type = "mha"
#         self.dtype = np.double
#         self.init_dtype_type()

#         batch_size = 2
#         nheads = 2
#         seq_len = 3
#         vec_size = 4
#         proj_size = vec_size // nheads

#         Q, K, V, W, WQ, WK, WV, WO, \
#         q_seq_arr, k_seq_arr, \
#         lo_win, hi_win = _generate_data(batch_size, seq_len, vec_size, self.dtype)

#         self.inputs = {
#             'Q': Q,
#             'K': K,
#             'V': V,
#             'W': W,
#             'QO_Seqlen': q_seq_arr,
#             'KV_Seqlen': k_seq_arr
#         }

#         self.attrs = {
#             'attn_low_windows': lo_win,
#             'attn_high_windows': hi_win,
#             'attn_dropout_rate': 0.0,
#             'attn_heads': nheads,
#             'attn_sm_scaler': 1.0,
#             'attn_vec_size': vec_size,
#             'attn_q_proj_size': proj_size,
#             'attn_k_proj_size': proj_size,
#             'attn_v_proj_size': proj_size,
#             'attn_o_proj_size': vec_size,
#             'attn_max_qo_seq_len': seq_len,
#             'attn_max_kv_seq_len': seq_len,
#             'attn_beam_size': 1
#         }
#         O = _get_attn_output(Q, K, V, WQ, WK, WV, WO, seq_len, nheads, vec_size,
#                              self.attrs["attn_sm_scaler"])
#         self.outputs = {'O': O}

#     def init_dtype_type(self):
#         pass

#     def test_check_output(self):
#         place = core.CUDAPlace(0)
#         self.check_output_with_place(place, atol=2e-1)

#     def test_check_grad_normal(self):
#         place = core.CUDAPlace(0)
#         self.check_grad_with_place(
#             place, ['Q', 'K', 'V', 'W'], 'O')

if __name__ == "__main__":
    unittest.main()
