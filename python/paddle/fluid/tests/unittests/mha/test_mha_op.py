# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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
from utils import generate_weight, generate_data, generate_varlen_data

import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def _get_attn_output(q, k, v, wq, bq, wk, bk, wv, bv, wo, bo, attn_mask, seqlen,
                     nheads, vec_size, sm_scaler):

    origin_dtype = q.dtype
    np_compute_dtype = np.double if origin_dtype == np.double else np.single
    proj_size = vec_size // nheads

    q_bar = q.reshape((-1, seqlen, vec_size)).astype(np_compute_dtype)
    k_bar = k.reshape((-1, seqlen, vec_size)).astype(np_compute_dtype)
    v_bar = v.reshape((-1, seqlen, vec_size)).astype(np_compute_dtype)

    wq = wq.astype(np_compute_dtype)
    wk = wk.astype(np_compute_dtype)
    wv = wv.astype(np_compute_dtype)
    wo = wo.astype(np_compute_dtype)

    q_bar = (np.matmul(q_bar, wq) + bq).reshape(
        (-1, seqlen, nheads, proj_size)).transpose((0, 2, 1, 3))
    k_bar = (np.matmul(k_bar, wk) + bk).reshape(
        (-1, seqlen, nheads, proj_size)).transpose((0, 2, 1, 3))
    v_bar = (np.matmul(v_bar, wv) + bv).reshape(
        (-1, seqlen, nheads, proj_size)).transpose((0, 2, 1, 3))

    beta = np.matmul(q_bar, k_bar.transpose((0, 1, 3, 2))) * sm_scaler
    beta = beta + ((attn_mask - 1.0) * 1e9)
    alpha = _softmax(beta)

    h_bar = np.matmul(alpha, v_bar).transpose((0, 2, 1, 3)).reshape(
        (-1, seqlen, vec_size))
    out = np.matmul(h_bar, wo) + bo
    return out.reshape((-1, seqlen, vec_size)).astype(origin_dtype)


def _generate_seqlen(batch, min_seqlen, max_seqlen, is_pad=True):
    seqlen = np.random.randint(
        low=min_seqlen, high=max_seqlen + 1, size=(batch, ), dtype=np.int32)
    if is_pad:
        # if pad, then nothing to do
        low_windows = np.zeros((max_seqlen, ), dtype=np.int32)
        high_windows = np.full(
            (max_seqlen, ), max_seqlen, dtype=np.int32)  # set a large number
    else:
        # if not pad, we should set the low, high windows inside a batch for each sequence
        cumsum = np.cumsum(seqlen, dtype=np.int32)
        low_windows = np.insert(cumsum[:-1], 0, 0)  # compute for each sequence
        low_windows = np.repeat(low_windows, seqlen)  # set for each token
        high_windows = cumsum
        high_windows = np.repeat(high_windows, seqlen)
    return seqlen, seqlen, low_windows, high_windows


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 4
        seqlen = 4
        embed_dim = 8
        proj_size = embed_dim // nheads

        query, key, value = generate_data(batch_size, seqlen, embed_dim,
                                          self.dtype)
        weight,  q_proj_weight, q_proj_bias, \
            k_proj_weight, k_proj_bias, \
            v_proj_weight, v_proj_bias, \
            out_proj_weight, out_proj_bias = generate_weight(embed_dim, self.dtype)

        qo_seqlen, kv_seqlen, _, _ = _generate_seqlen(
            batch_size, min_seqlen=seqlen, max_seqlen=seqlen)
        attn_mask = np.ones((batch_size, nheads, seqlen, seqlen))

        self.inputs = {
            'query': query,
            'key': key,
            'value': value,
            'weight': weight,
            'qo_kv_seqlen': np.concatenate((qo_seqlen, kv_seqlen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'pre_dropout_rate': 0.,
            'num_heads': nheads,
            'softmax_scaler': 1.,
            'embedding_size': embed_dim,
            'query_proj_size': proj_size,
            'key_proj_size': proj_size,
            'value_proj_size': proj_size,
            'output_proj_size': embed_dim,
            'max_qo_seqlen': seqlen,
            'max_kv_seqlen': seqlen
        }

        output = _get_attn_output(query, key, value, q_proj_weight, q_proj_bias,
                                  k_proj_weight, k_proj_bias, v_proj_weight,
                                  v_proj_bias, out_proj_weight, out_proj_bias,
                                  attn_mask, seqlen, nheads, embed_dim,
                                  self.attrs["softmax_scaler"])
        self.outputs = {'output': output}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1e-1

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['query', 'key', 'value', 'weight'],
            'output',
            max_relative_error=self.grad_rtol)


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP32(TestMHAOpFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 1e-4


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP64(TestMHAOpFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpPadVarLenFP16(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 4
        max_seqlen = 4
        embed_dim = 8
        proj_size = embed_dim // nheads

        query, key, value = generate_data(batch_size, max_seqlen, embed_dim,
                                          self.dtype)
        weight,  q_proj_weight, q_proj_bias, \
            k_proj_weight, k_proj_bias, \
            v_proj_weight, v_proj_bias, \
            out_proj_weight, out_proj_bias = generate_weight(embed_dim, self.dtype)

        qo_seqlen, kv_seqlen, _, _ = _generate_seqlen(
            batch_size, min_seqlen=1, max_seqlen=max_seqlen)
        attn_mask = np.ones((batch_size, nheads, max_seqlen, max_seqlen))
        for sid, slen in enumerate(kv_seqlen):
            attn_mask[sid, :, :, slen:] = 0.0

        self.inputs = {
            'query': query,
            'key': key,
            'value': value,
            'weight': weight,
            'qo_kv_seqlen': np.concatenate((qo_seqlen, kv_seqlen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'pre_dropout_rate': 0.,
            'num_heads': nheads,
            'softmax_scaler': 1.,
            'embedding_size': embed_dim,
            'query_proj_size': proj_size,
            'key_proj_size': proj_size,
            'value_proj_size': proj_size,
            'output_proj_size': embed_dim,
            'max_qo_seqlen': max_seqlen,
            'max_kv_seqlen': max_seqlen
        }

        output = _get_attn_output(query, key, value, q_proj_weight, q_proj_bias,
                                  k_proj_weight, k_proj_bias, v_proj_weight,
                                  v_proj_bias, out_proj_weight, out_proj_bias,
                                  attn_mask, max_seqlen, nheads, embed_dim,
                                  self.attrs["softmax_scaler"])

        # The output of padding part does not need to care.
        # Here we set output projection's bias to make reference be the same as cuDNN's output.
        for sid, slen in enumerate(kv_seqlen):
            output[sid, slen:, :] = out_proj_bias

        self.outputs = {'output': output}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1e-1

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['query', 'key', 'value', 'weight'],
            'output',
            max_relative_error=self.grad_rtol)


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpPadVarLenFP32(TestMHAOpPadVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 1e-4


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
        max_seqlen = 4
        embed_dim = 8
        proj_size = embed_dim // nheads

        qo_seqlen, kv_seqlen, low_winows, high_windows = _generate_seqlen(
            batch_size, min_seqlen=1, max_seqlen=max_seqlen, is_pad=False)

        query, key, value = generate_varlen_data(qo_seqlen, embed_dim,
                                                 self.dtype)
        weight,  q_proj_weight, q_proj_bias, \
            k_proj_weight, k_proj_bias, \
            v_proj_weight, v_proj_bias, \
            out_proj_weight, out_proj_bias = generate_weight(embed_dim, self.dtype)

        qo_seqlens = np.sum(qo_seqlen, dtype=np.int32).reshape(1, )
        kv_seqlens = np.sum(kv_seqlen, dtype=np.int32).reshape(1, )

        self.inputs = {
            'query': query,
            'key': key,
            'value': value,
            'weight': weight,
            'qo_kv_seqlen': np.concatenate((qo_seqlens, kv_seqlens)),
            'low_high_windows_host': np.concatenate((low_winows, high_windows))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'pre_dropout_rate': 0.,
            'num_heads': nheads,
            'softmax_scaler': 1.,
            'embedding_size': embed_dim,
            'query_proj_size': proj_size,
            'key_proj_size': proj_size,
            'value_proj_size': proj_size,
            'output_proj_size': embed_dim,
            'max_qo_seqlen': np.sum(qo_seqlen, dtype=np.int32),
            'max_kv_seqlen': np.sum(kv_seqlen, dtype=np.int32)
        }

        offset = np.insert(np.cumsum(qo_seqlen), 0, 0)
        output = None
        for sid, slen in enumerate(qo_seqlen):
            sub_o = _get_attn_output(
                query[0, offset[sid]:offset[sid + 1], :],
                key[0, offset[sid]:offset[sid + 1], :],
                value[0, offset[sid]:offset[sid + 1], :], q_proj_weight,
                q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight,
                v_proj_bias, out_proj_weight, out_proj_bias,
                np.ones((1, nheads, slen, slen)), slen, nheads, embed_dim,
                self.attrs["softmax_scaler"])
            if output is not None:
                output = np.concatenate((output, sub_o), axis=1)
            else:
                output = sub_o
        self.outputs = {'output': output}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1e-1

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['query', 'key', 'value', 'weight'],
            'output',
            max_relative_error=self.grad_rtol)


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpVarLenFP32(TestMHAOpVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 1e-4


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpVarLenFP64(TestMHAOpVarLenFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP16WithResidual(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 4
        seqlen = 4
        embed_dim = 8
        proj_size = embed_dim // nheads

        query, key, value = generate_data(batch_size, seqlen, embed_dim,
                                          self.dtype)
        residual = np.random.uniform(
            low=-0.03, high=0.03, size=query.shape).astype(self.dtype)
        weight,  q_proj_weight, q_proj_bias, \
            k_proj_weight, k_proj_bias, \
            v_proj_weight, v_proj_bias, \
            out_proj_weight, out_proj_bias = generate_weight(embed_dim, self.dtype)

        qo_seqlen, kv_seqlen, _, _ = _generate_seqlen(
            batch_size, min_seqlen=seqlen, max_seqlen=seqlen)
        attn_mask = np.ones((batch_size, nheads, seqlen, seqlen))

        self.inputs = {
            'query': query,
            'key': key,
            'value': value,
            'weight': weight,
            'residual': residual,
            'qo_kv_seqlen': np.concatenate((qo_seqlen, kv_seqlen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'pre_dropout_rate': 0.,
            'num_heads': nheads,
            'softmax_scaler': 1.,
            'embedding_size': embed_dim,
            'query_proj_size': proj_size,
            'key_proj_size': proj_size,
            'value_proj_size': proj_size,
            'output_proj_size': embed_dim,
            'max_qo_seqlen': seqlen,
            'max_kv_seqlen': seqlen
        }

        output = _get_attn_output(query, key, value, q_proj_weight, q_proj_bias,
                                  k_proj_weight, k_proj_bias, v_proj_weight,
                                  v_proj_bias, out_proj_weight, out_proj_bias,
                                  attn_mask, seqlen, nheads, embed_dim,
                                  self.attrs["softmax_scaler"])
        self.outputs = {'output': output + residual}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-2
        self.grad_rtol = 1e-1

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad_normal(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_grad_with_place(
            self.place, ['query', 'key', 'value', 'weight', 'residual'],
            'output',
            max_relative_error=self.grad_rtol)


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP32WithResidual(TestMHAOpFP16WithResidual):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-3
        self.grad_rtol = 1e-4


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpFP64WithResidual(TestMHAOpFP16WithResidual):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6
        self.grad_rtol = 1e-7


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpWithPostDropout00(OpTest):
    def setUp(self):
        self.op_type = "mha"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        batch_size = 4
        nheads = 12
        seqlen = 128
        embed_dim = 768
        proj_size = embed_dim // nheads

        query, key, value = generate_data(batch_size, seqlen, embed_dim,
                                          self.dtype)

        weight, _, _, _, _, _, _, _, _ = generate_weight(embed_dim, self.dtype)

        qo_seqlen, kv_seqlen, _, _ = _generate_seqlen(
            batch_size, min_seqlen=seqlen, max_seqlen=seqlen)

        self.inputs = {
            'query': query,
            'key': key,
            'value': value,
            'weight': weight,
            'qo_kv_seqlen': np.concatenate((qo_seqlen, kv_seqlen))
        }

        self.attrs = {
            'cache_key': str(id(type(self))),
            'pre_dropout_rate': 0.,
            'num_heads': nheads,
            'softmax_scaler': 1.,
            'embedding_size': embed_dim,
            'query_proj_size': proj_size,
            'key_proj_size': proj_size,
            'value_proj_size': proj_size,
            'output_proj_size': embed_dim,
            'max_qo_seqlen': seqlen,
            'max_kv_seqlen': seqlen,
            'post_dropout_rate': self.post_dropout_rate
        }

        self.outputs = {'output': np.array(query, copy=True)}

    def init_dtype_type(self):
        self.dtype = np.float32
        self.post_dropout_rate = 0.0
        self.probability_margin = 0.05

    def test_check_output(self):
        self.check_output_with_place_customized(
            self.post_dropout_output_checker, self.place)

    def post_dropout_output_checker(self, outputs):
        output_flattened = outputs[0].flatten()
        zero_ratio = 1.0 - float(np.nonzero(output_flattened)[0]
                                 .size) / output_flattened.size

        lower_bound = self.post_dropout_rate - self.probability_margin
        upper_bound = self.post_dropout_rate + self.probability_margin
        self.assertTrue(
            (zero_ratio >= lower_bound and zero_ratio <= upper_bound),
            "[TestMHAOpWithPostDropout] The ratio of output zeros is not between "
            "margins, expect in [{:.2f}, {:.2f}], but got {:2f}".format(
                lower_bound, upper_bound, zero_ratio))


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpWithPostDropout05(TestMHAOpWithPostDropout00):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.post_dropout_rate = 0.5
        self.probability_margin = 0.05


@skip_check_grad_ci(reason="Developing")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHAOpWithPostDropout10(TestMHAOpWithPostDropout00):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.post_dropout_rate = 1.0
        self.probability_margin = 0.05


if __name__ == "__main__":
    unittest.main()
