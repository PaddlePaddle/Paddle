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
import paddle
import paddle.fluid.core as core
import sys
sys.path.append("..")
from op_test import OpTest
import paddle.fluid as fluid


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOp(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.dtype = np.float16
        self.init_dtype_type()

        batch_size = 2
        nheads = 4
        seq_len = 4
        vec_size = 8
        proj_size = vec_size // nheads


        Q = np.random.random((batch_size, 1, seq_len, vec_size)).astype(self.dtype)
        K = np.random.random((batch_size, 1, seq_len, vec_size)).astype(self.dtype)
        V = np.random.random((batch_size, 1, seq_len, vec_size)).astype(self.dtype)
        W = np.random.random((4*vec_size*vec_size,)).astype(self.dtype)

        # WQ = W[0:64].reshape((vec_size, nheads, proj_size)).transpose(1, 2, 0)
        # WK = W[64:128].reshape((vec_size, nheads, proj_size)).transpose(1, 2, 0)
        # WV = W[128:192].reshape((vec_size, nheads, proj_size)).transpose(1, 2, 0)
        # WO = W[192:].reshape((nheads, proj_size, vec_size))

        WQ = W[0:64].reshape((vec_size, vec_size))
        WK = W[64:128].reshape((vec_size, vec_size))
        WV = W[128:192].reshape((vec_size, vec_size))
        WO = W[192:].reshape((vec_size, vec_size))

        q_seq_arr = np.full((batch_size, ), seq_len)
        k_seq_arr = np.full((batch_size, ), seq_len)

        lo_win = np.zeros((seq_len, ), dtype=int)
        hi_win = np.full((seq_len, ), seq_len, dtype=int)

        self.inputs = {'Q': Q, 'K': K, 'V': V, 'W':W}
        self.attrs = {
            'Q_seq_size_arr': q_seq_arr,
            'K_seq_size_arr':k_seq_arr,
            'attn_low_windows': lo_win,
            'attn_high_windows':hi_win,
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
        O = self._get_attn_output(Q, K, V, WQ, WK, WV, WO, seq_len)
        self.outputs = {'O': O}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=2e-1)

    def test_check_grad_normal(self):
        pass
        # place = core.CUDAPlace(0)
        # if core.is_float16_supported(place):
        #     self.check_grad_with_place(
        #         place, ['X', 'Y'], 'Out', max_relative_error=1.0)

    def _get_attn_output(self, q, k, v, wq, wk, wv, wo, seq_len):

        nheads = self.attrs["attn_heads"]
        vec_size = self.attrs["attn_vec_size"]
        proj_size = vec_size // nheads
        sm_scaler = self.attrs["attn_sm_scaler"]
        
        q_bar = q.reshape((-1, seq_len, vec_size))
        k_bar = k.reshape((-1, seq_len, vec_size))
        v_bar = v.reshape((-1, seq_len, vec_size))

        q_bar = np.matmul(q, wq).reshape((-1, seq_len, nheads, proj_size)).transpose(0, 2, 1, 3)
        k_bar = np.matmul(k, wk).reshape((-1, seq_len, nheads, proj_size)).transpose(0, 2, 1, 3)
        v_bar = np.matmul(v, wv).reshape((-1, seq_len, nheads, proj_size)).transpose(0, 2, 1, 3)

        beta = np.matmul(q_bar, k_bar.transpose(0, 1, 3, 2)) * sm_scaler
        alpha = self._softmax(beta)

        h_bar = np.matmul(alpha, v_bar).transpose(0, 2, 1, 3).reshape((-1, seq_len, vec_size))
        out = np.matmul(h_bar, wo)
        return out

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
        # nheads = self.attrs["attn_heads"]
        # for i in range(nheads):
        #     q_bar = np.dot(wq[i,:,:], q)
        #     k_bar = np.dot(wk[i,:,:], k)
        #     v_bar = np.dot(wv[i,:,:], v)

        #     beta = scaler * np.dot(k_bar.transpose(), q_bar)
        #     alpha = softmax(beta)

        #     h_bar = np.dot(v_bar, alpha)
        #     h = np.dot(wo[i,:,:], h_bar)
        #     out = np.add(out, h)

        # return out

    # def test_check_grad_ingore_x(self):
    #     place = core.CUDAPlace(0)
    #     if core.is_float16_supported(place):
    #         self.check_grad_with_place(
    #             place, ['Y'],
    #             'Out',
    #             max_relative_error=1.0,
    #             no_grad_set=set("X"))

    # def test_check_grad_ingore_y(self):
    #     place = core.CUDAPlace(0)
    #     if core.is_float16_supported(place):
    #         self.check_grad_with_place(
    #             place, ['X'],
    #             'Out',
    #             max_relative_error=1.0,
    #             no_grad_set=set('Y'))


if __name__ == "__main__":
    unittest.main()
