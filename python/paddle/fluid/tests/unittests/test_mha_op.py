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
import paddle


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def _get_attn_output(q, k, v, wq, wk, wv, wo,
                               bq, bk, bv, bo, attn_mask,
                               seq_len, nheads, vec_size, sm_scaler):

    origin_dtype = q.dtype
    np_compute_dtype = np.double if origin_dtype == np.double else np.single
    proj_size = vec_size // nheads

    q_bar = q.reshape((-1, seq_len, vec_size)).astype(np_compute_dtype)
    k_bar = k.reshape((-1, seq_len, vec_size)).astype(np_compute_dtype)
    v_bar = v.reshape((-1, seq_len, vec_size)).astype(np_compute_dtype)

    wq = wq.astype(np_compute_dtype)
    wk = wk.astype(np_compute_dtype)
    wv = wv.astype(np_compute_dtype)
    wo = wo.astype(np_compute_dtype)

    q_bar = (np.matmul(q_bar, wq) + bq).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))
    k_bar = (np.matmul(k_bar, wk) + bk).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))
    v_bar = (np.matmul(v_bar, wv) + bv).reshape(
        (-1, seq_len, nheads, proj_size)).transpose((0, 2, 1, 3))

    beta = np.matmul(q_bar, k_bar.transpose((0, 1, 3, 2))) * sm_scaler
    beta = beta + ((attn_mask-1.0) * 1e9)
    alpha = _softmax(beta)

    h_bar = np.matmul(alpha, v_bar).transpose((0, 2, 1, 3)).reshape(
        (-1, seq_len, vec_size))
    out = np.matmul(h_bar, wo) + bo
    return out.reshape((-1, seq_len, 1, vec_size)).astype(origin_dtype)


def _generate_data(batch_size, max_seq_len, vec_size, dtype):
    Q = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    K = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    V = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    W = np.random.uniform(low=-0.03, high=0.03, size=(4 * vec_size * vec_size)).astype(dtype)
    B =  np.random.uniform(low=-0.01, high=0.01, size=(4 * vec_size,)).astype(dtype)
    W = np.concatenate((W, B), dtype=dtype)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:4 * stride].reshape((vec_size, vec_size))

    bias_start = 4 * stride
    BQ = W[bias_start: bias_start + vec_size]
    BK = W[bias_start + vec_size:bias_start + 2*vec_size]
    BV = W[bias_start + 2*vec_size: bias_start + 3*vec_size]
    BO = W[bias_start + 3*vec_size: bias_start + 4*vec_size]

    return (Q, K, V, W, WQ, WK, WV, WO, BQ, BK, BV, BO)


def _generate_varlen_data(seq_lens, vec_size, dtype):
    """
    seq_lens (list): the desired seq_lens
    """

    assert len(seq_lens) > 0, "batch size should be greater than 0"

    Qs = [(np.random.random((1, seq_len, 1, vec_size)) - .5).astype(dtype)
          for seq_len in seq_lens]
    Ks = [(np.random.random((1, seq_len, 1, vec_size)) - .5).astype(dtype)
          for seq_len in seq_lens]
    Vs = [(np.random.random((1, seq_len, 1, vec_size)) - .5).astype(dtype)
          for seq_len in seq_lens]

    Q = np.concatenate(Qs, axis=1)
    K = np.concatenate(Ks, axis=1)
    V = np.concatenate(Vs, axis=1)
    W = np.random.uniform(low=-0.03, high=0.03, size=(4 * vec_size * vec_size)).astype(dtype)
    B =  np.random.uniform(low=-0.01, high=0.01, size=(4 * vec_size,)).astype(dtype)
    W = np.concatenate((W, B), dtype=dtype)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:4 * stride].reshape((vec_size, vec_size))

    bias_start = 4 * stride
    BQ = W[bias_start: bias_start + vec_size]
    BK = W[bias_start + vec_size:bias_start + 2*vec_size]
    BV = W[bias_start + 2*vec_size: bias_start + 3*vec_size]
    BO = W[bias_start + 3*vec_size: bias_start + 4*vec_size]
    return (Q, K, V, W, WQ, WK, WV, WO, BQ, BK, BV, BO)


def _generate_seq_len(batch, min_seq_len, max_seq_len, is_pad=True):
    seq_len = np.random.randint(
        low=min_seq_len, high=max_seq_len + 1, size=(batch, ), dtype=np.int32)
    if is_pad:
        # if pad, then nothing to do
        lo_win = np.zeros((max_seq_len, ), dtype=np.int32)
        hi_win = np.full(
            (max_seq_len, ), max_seq_len, dtype=np.int32)  # set a large number
    else:
        # if not pad, we should set the low, high windows inside a batch for each sequence
        cumsum = np.cumsum(seq_len, dtype=np.int32)
        lo_win = np.insert(cumsum[:-1], 0, 0)  # compute for each sequence
        lo_win = np.repeat(lo_win, seq_len)  # set for each token
        hi_win = cumsum
        hi_win = np.repeat(hi_win, seq_len)
    return seq_len, seq_len, lo_win, hi_win


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 4
        seq_len = 4
        vec_size = 8
        proj_size = vec_size // nheads

        Q, K, V, W, WQ, WK, WV, WO, BQ, BK, BV, BO = \
            _generate_data(batch_size, seq_len, vec_size, self.dtype)
        qo_slen, kv_slen, lo_win, hi_win = _generate_seq_len(
            batch_size, min_seq_len=seq_len, max_seq_len=seq_len)
        attn_mask = np.ones((batch_size, nheads, seq_len, seq_len))

        self.inputs = {
            'Q': Q,
            'K': K,
            'V': V,
            'W': W,
            'QO_KV_Seqlen': np.concatenate((qo_slen,kv_slen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'attn_dropout_rate': 0.,
            'attn_heads': nheads,
            'attn_sm_scaler': 1.,
            'attn_vec_size': vec_size,
            'attn_q_proj_size': proj_size,
            'attn_k_proj_size': proj_size,
            'attn_v_proj_size': proj_size,
            'attn_o_proj_size': vec_size,
            'attn_max_qo_seq_len': seq_len,
            'attn_max_kv_seq_len': seq_len,
            'attn_beam_size': 1
        }

        O = _get_attn_output(Q, K, V, WQ, WK, WV, WO, BQ, BK, BV, BO, attn_mask,
                             seq_len, nheads, vec_size, self.attrs["attn_sm_scaler"])
        self.outputs = {'O': O}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1.0

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)
        print(f'MHA {self.dtype} fwd passed.')

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['Q', 'K', 'V', 'W'],
            'O',
            max_relative_error=self.grad_rtol)
        print(f'MHA {self.dtype} bwd passed.')


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP32(TestMHAOpFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 0.5


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP64(TestMHAOpFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpPadVarLenFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 1
        nheads = 4
        max_seq_len = 4
        vec_size = 8
        proj_size = vec_size // nheads

        Q, K, V, W, WQ, WK, WV, WO, BQ, BK, BV, BO = \
            _generate_data(batch_size, max_seq_len, vec_size, self.dtype)
        qo_slen, kv_slen, lo_win, hi_win = _generate_seq_len(
            batch_size, min_seq_len=1, max_seq_len=max_seq_len)
        attn_mask = np.ones((batch_size, nheads, max_seq_len, max_seq_len))
        for sid, slen in enumerate(kv_slen):
            attn_mask[sid, :, :, slen:] = 0.0

        self.inputs = {
            'Q': Q,
            'K': K,
            'V': V,
            'W': W,
            'QO_KV_Seqlen': np.concatenate((qo_slen,kv_slen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'attn_dropout_rate': 0.,
            'attn_heads': nheads,
            'attn_sm_scaler': 1.,
            'attn_vec_size': vec_size,
            'attn_q_proj_size': proj_size,
            'attn_k_proj_size': proj_size,
            'attn_v_proj_size': proj_size,
            'attn_o_proj_size': vec_size,
            'attn_max_qo_seq_len': max_seq_len,
            'attn_max_kv_seq_len': max_seq_len,
            'attn_beam_size': 1
        }

        O = _get_attn_output(Q, K, V, WQ, WK, WV, WO, BQ, BK, BV, BO, attn_mask,
                             max_seq_len, nheads, vec_size, self.attrs["attn_sm_scaler"])

        # The output of padding part does not need to care.
        # Here we set output projection's bias to make reference be the same as cuDNN's output.
        for sid, slen in enumerate(kv_slen):
            O[sid, slen:, :] = BO

        self.outputs = {'O': O}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1.0

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)
        print(f'MHA padded varlen {self.dtype} fwd passed.')

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['Q', 'K', 'V', 'W'],
            'O',
            max_relative_error=self.grad_rtol)
        print(f'MHA padded varlen {self.dtype} bwd passed.')


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpPadVarLenFP32(TestMHAOpPadVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 0.5


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpPadVarLenFP64(TestMHAOpPadVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpVarLenFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 4
        max_seq_len = 4
        vec_size = 8
        proj_size = vec_size // nheads
        
        qo_slen, kv_slen, lo_win, hi_win = _generate_seq_len(
            batch_size, min_seq_len=1, max_seq_len=max_seq_len, is_pad=False)
        Q, K, V, W, WQ, WK, WV, WO, BQ, BK, BV, BO = \
            _generate_varlen_data(qo_slen, vec_size, self.dtype)

        qo_slens = np.sum(qo_slen, dtype=np.int32).reshape(1, )
        kv_slens = np.sum(kv_slen, dtype=np.int32).reshape(1, )

        self.inputs = {
            'Q': Q,
            'K': K,
            'V': V,
            'W': W,
            'QO_KV_Seqlen': np.concatenate((qo_slens, kv_slens)),
            'low_high_windows': np.concatenate((lo_win, hi_win))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'attn_dropout_rate': 0.,
            'attn_heads': nheads,
            'attn_sm_scaler': 1.,
            'attn_vec_size': vec_size,
            'attn_q_proj_size': proj_size,
            'attn_k_proj_size': proj_size,
            'attn_v_proj_size': proj_size,
            'attn_o_proj_size': vec_size,
            'attn_max_qo_seq_len': np.sum(qo_slen, dtype=np.int32),
            'attn_max_kv_seq_len': np.sum(kv_slen, dtype=np.int32),
            'attn_beam_size': 1
        }

        offset = np.insert(np.cumsum(qo_slen), 0, 0)
        O = None
        for sid, slen in enumerate(qo_slen):
            sub_o = _get_attn_output(
                                     Q[0, offset[sid]:offset[sid + 1], :, :],
                                     K[0, offset[sid]:offset[sid + 1], :, :],
                                     V[0, offset[sid]:offset[sid + 1], :, :],
                                     WQ, WK, WV, WO, BQ, BK, BV, BO, np.ones((1, nheads, slen, slen)),
                                     slen, nheads, vec_size,
                                     self.attrs["attn_sm_scaler"])
            if O is not None:
                O = np.concatenate((O, sub_o), axis=1)
            else:
                O = sub_o
        self.outputs = {'O': O}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1.0

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)
        print(f'MHA varlen {self.dtype} fwd passed.')

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['Q', 'K', 'V', 'W'],
            'O',
            max_relative_error=self.grad_rtol)
        print(f'MHA varlen {self.dtype} bwd passed.')


class TestMHAOpVarLenFP32(TestMHAOpVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 0.5


class TestMHAOpVarLenFP64(TestMHAOpVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


if __name__ == "__main__":
    unittest.main()
