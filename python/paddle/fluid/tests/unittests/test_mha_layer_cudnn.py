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
import paddle.fluid.core as core

import paddle
from paddle.nn import MultiHeadAttention, CUDNNMultiHeadAttention
from paddle.nn.layer import CUDNNSeqDataInfo


def is_equal_atol(a, b, atol):
    a = a.flatten()
    b = b.flatten()

    a_b_diff = np.abs(a - b)
    is_error = np.sum(a_b_diff > atol)
    if is_error:
        return False
    return True


def is_equal_rtol(a, b, rtol):
    a = a.flatten()
    b = b.flatten()

    a_abs = np.abs(a)
    rel_err = max(np.abs(a - b) / a_abs)
    if rel_err > rtol:
        return False
    return True


def _generate_data(batch_size, max_seq_len, vec_size, dtype):
    Q = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    K = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    V = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    W = (np.random.random((4 * vec_size * vec_size, )) - .5).astype(dtype)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:].reshape((vec_size, vec_size))

    return (Q, K, V, W, WQ, WK, WV, WO)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMHALayer(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        nheads = 4
        seq_len = 4
        vec_size = 8

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = _generate_data(batch_size, seq_len, vec_size, self.dtype)

        self.ref_mha = MultiHeadAttention(vec_size, nheads)
        self.ref_mha.q_proj.weight.set_value(self.WQ)
        self.ref_mha.k_proj.weight.set_value(self.WK)
        self.ref_mha.v_proj.weight.set_value(self.WV)
        self.ref_mha.out_proj.weight.set_value(self.WO)

        self.cudnn_mha = CUDNNMultiHeadAttention(vec_size, nheads)
        self.cudnn_mha.weight.set_value(self.W)

        self.q_tensor = paddle.to_tensor(
            self.Q, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor = paddle.to_tensor(
            self.K, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor = paddle.to_tensor(
            self.V, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.q_3dim_tensor = paddle.to_tensor(
            self.Q.reshape((batch_size, seq_len, vec_size)),
            dtype=self.dtype,
            place=self.place,
            stop_gradient=False)
        self.k_3dim_tensor = paddle.to_tensor(
            self.K.reshape((batch_size, seq_len, vec_size)),
            dtype=self.dtype,
            place=self.place,
            stop_gradient=False)
        self.v_3dim_tensor = paddle.to_tensor(
            self.V.reshape((batch_size, seq_len, vec_size)),
            dtype=self.dtype,
            place=self.place,
            stop_gradient=False)

        attn_mask = np.ones((batch_size, seq_len))
        self.seq_data = CUDNNSeqDataInfo(attn_mask, self.place)
        self.attn_tensor = paddle.to_tensor(
            np.ones((batch_size, nheads, seq_len, seq_len)),
            dtype=np.bool,
            place=self.place)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-3
        self.rtol = 1e-2

    def test_fwd_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return

        ref_output = self.ref_mha(self.q_3dim_tensor, self.k_3dim_tensor,
                                  self.v_3dim_tensor, self.attn_tensor)
        cudnn_output = self.cudnn_mha(self.q_tensor, self.k_tensor,
                                      self.v_tensor, self.seq_data)
        self.assertEqual(
            is_equal_atol(ref_output.numpy(), cudnn_output.numpy(), self.atol),
            True)

    def test_full_grads(self):
        self.q_tensor.stop_gradient = False
        self.k_tensor.stop_gradient = False
        self.v_tensor.stop_gradient = False
        self.q_3dim_tensor.stop_gradient = False
        self.k_3dim_tensor.stop_gradient = False
        self.v_3dim_tensor.stop_gradient = False

        self._cehck_grads()

    def test_weight_grads_only(self):
        self.q_tensor.stop_gradient = True
        self.k_tensor.stop_gradient = True
        self.v_tensor.stop_gradient = True
        self.q_3dim_tensor.stop_gradient = True
        self.k_3dim_tensor.stop_gradient = True
        self.v_3dim_tensor.stop_gradient = True

        self._cehck_grads(False)

    def _cehck_grads(self, check_data_grads=True):
        ref_output = self.ref_mha(self.q_3dim_tensor, self.k_3dim_tensor,
                                  self.v_3dim_tensor, self.attn_tensor)
        cudnn_output = self.cudnn_mha(self.q_tensor, self.k_tensor,
                                      self.v_tensor, self.seq_data)

        ref_loss = paddle.mean(ref_output)
        cudnn_loss = paddle.mean(cudnn_output)

        paddle.autograd.backward([ref_loss, cudnn_loss])

        ref_weight_grad = self._get_grads_from_ref()
        cudnn_weight_grad = self.cudnn_mha.weight.grad.numpy()
        self.assertEqual(
            is_equal_rtol(ref_weight_grad, cudnn_weight_grad, self.rtol), True)

        if check_data_grads:
            self.assertEqual(
                is_equal_rtol(self.q_3dim_tensor.grad.numpy(),
                              self.q_tensor.grad.numpy(), self.rtol), True)
            self.assertEqual(
                is_equal_rtol(self.k_3dim_tensor.grad.numpy(),
                              self.k_tensor.grad.numpy(), self.rtol), True)
            self.assertEqual(
                is_equal_rtol(self.v_3dim_tensor.grad.numpy(),
                              self.v_tensor.grad.numpy(), self.rtol), True)

    def _get_grads_from_ref(self):
        return np.concatenate(
            (self.ref_mha.q_proj.weight.grad.numpy(),
             self.ref_mha.k_proj.weight.grad.numpy(),
             self.ref_mha.v_proj.weight.grad.numpy(),
             self.ref_mha.out_proj.weight.grad.numpy()),
            axis=0)


if __name__ == "__main__":
    unittest.main()
