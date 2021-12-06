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
from paddle.nn.layer import CUDNNSeqInfoInfer


def compare(ref, res, atol, rtol):
    ref = ref.flatten()
    res = res.flatten()

    tmp_ref = ref.astype(np.float)
    tol = atol + rtol * abs(tmp_ref)

    diff = abs(res - ref)

    indices = np.transpose(np.where(diff > tol))
    if len(indices) == 0:
        return True
    return False


def _generate_data(batch_size, max_seq_len, vec_size, dtype):
    Q = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(dtype)
    K = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(dtype)
    V = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(dtype)
    W = (np.random.random((4 * vec_size * vec_size, )) - .5).astype(np.single)
    W = np.concatenate((W, np.zeros((4 * vec_size, ))), dtype=np.single)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:4 * stride].reshape((vec_size, vec_size))

    return (Q, K, V, W, WQ, WK, WV, WO)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP32CUDNNMHALayer(unittest.TestCase):
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
            self.Q, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_3dim_tensor = paddle.to_tensor(
            self.K, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_3dim_tensor = paddle.to_tensor(
            self.V, dtype=self.dtype, place=self.place, stop_gradient=False)

        attn_mask = paddle.to_tensor(
            np.ones((batch_size, seq_len)), place=self.place)
        seq_infer = CUDNNSeqInfoInfer()
        self.seq_data = seq_infer(attn_mask)

        self.attn_tensor = paddle.to_tensor(
            np.ones((batch_size, nheads, seq_len, seq_len)),
            dtype=np.bool,
            place=self.place)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-4

    def test_fwd_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return

        enable_amp = False
        if self.dtype == np.float16:
            enable_amp = True

        with paddle.amp.auto_cast(enable=enable_amp, custom_white_list={'mha'}):
            ref_output = self.ref_mha(self.q_3dim_tensor, self.k_3dim_tensor,
                                      self.v_3dim_tensor, self.attn_tensor)
            cudnn_output = self.cudnn_mha(self.q_tensor, self.k_tensor,
                                          self.v_tensor, self.seq_data)
        self.assertTrue(
            compare(ref_output.numpy(),
                    cudnn_output.numpy(), self.atol, self.rtol),
            "[Test*CUDNNMHALayer] outputs are miss-matched.")

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

        enable_amp = False
        if self.dtype == np.float16:
            enable_amp = True

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        with paddle.amp.auto_cast(enable=enable_amp, custom_white_list={'mha'}):
            ref_output = self.ref_mha(self.q_3dim_tensor, self.k_3dim_tensor,
                                      self.v_3dim_tensor, self.attn_tensor)
            cudnn_output = self.cudnn_mha(self.q_tensor, self.k_tensor,
                                          self.v_tensor, self.seq_data)

            ref_loss = paddle.mean(ref_output)
            cudnn_loss = paddle.mean(cudnn_output)

            if enable_amp:
                ref_loss = scaler.scale(ref_loss)
                cudnn_loss = scaler.scale(cudnn_loss)
            paddle.autograd.backward([ref_loss, cudnn_loss])

        ref_weight_grad = self._get_grads_from_ref()
        cudnn_weight_grad = self.cudnn_mha.weight.grad.numpy()

        self.assertTrue(
            compare(ref_weight_grad, cudnn_weight_grad, self.atol, self.rtol),
            "[Test*CUDNNMHALayer] weight_grads are miss-matched.")
        if check_data_grads:
            self.assertTrue(
                compare(self.q_3dim_tensor.grad.numpy(),
                        self.q_tensor.grad.numpy(), self.atol, self.rtol),
                "[Test*CUDNNMHALayer] Q_grads are miss-matched.")
            self.assertTrue(
                compare(self.k_3dim_tensor.grad.numpy(),
                        self.k_tensor.grad.numpy(), self.atol, self.rtol),
                "[Test*CUDNNMHALayer] K_grads are miss-matched.")
            self.assertTrue(
                compare(self.v_3dim_tensor.grad.numpy(),
                        self.v_tensor.grad.numpy(), self.atol, self.rtol),
                "[Test*CUDNNMHALayer] V_grads are miss-matched.")

    def _get_grads_from_ref(self):
        return np.concatenate(
            (self.ref_mha.q_proj.weight.grad.numpy().flatten(),
             self.ref_mha.k_proj.weight.grad.numpy().flatten(),
             self.ref_mha.v_proj.weight.grad.numpy().flatten(),
             self.ref_mha.out_proj.weight.grad.numpy().flatten(),
             self.ref_mha.q_proj.bias.grad.numpy().flatten(),
             self.ref_mha.k_proj.bias.grad.numpy().flatten(),
             self.ref_mha.v_proj.bias.grad.numpy().flatten(),
             self.ref_mha.out_proj.bias.grad.numpy().flatten()),
            axis=0)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16CUDNNMHALayer(TestFP32CUDNNMHALayer):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3
        self.rtol = 1e-4


if __name__ == "__main__":
    unittest.main()
