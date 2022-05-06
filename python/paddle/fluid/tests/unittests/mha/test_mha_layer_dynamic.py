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
import paddle.fluid.core as core

import paddle
from paddle.nn import MultiHeadAttention
from utils import compare, generate_weight, generate_data, get_dtype_str
from utils import MultiHeadAttentionRef, skip_unit_test


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestFP32cuDNNMultiHeadAttentionDynamic(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        nheads = 4
        seqlen = 4
        embed_dim = 8

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.query, self.key, self.value = generate_data(batch_size, seqlen,
                                                         embed_dim, self.dtype)
        self.weight,  self.q_proj_weight, self.q_proj_bias, \
            self.k_proj_weight, self.k_proj_bias, \
            self.v_proj_weight, self.v_proj_bias, \
            self.out_proj_weight, self.out_proj_bias = generate_weight(embed_dim, np.single)

        self.ref_mha = MultiHeadAttentionRef(embed_dim, nheads)
        self.ref_mha.q_proj.weight.set_value(self.q_proj_weight)
        self.ref_mha.k_proj.weight.set_value(self.k_proj_weight)
        self.ref_mha.v_proj.weight.set_value(self.v_proj_weight)
        self.ref_mha.out_proj.weight.set_value(self.out_proj_weight)
        self.ref_mha.q_proj.bias.set_value(self.q_proj_bias)
        self.ref_mha.k_proj.bias.set_value(self.k_proj_bias)
        self.ref_mha.v_proj.bias.set_value(self.v_proj_bias)
        self.ref_mha.out_proj.bias.set_value(self.out_proj_bias)

        self.cudnn_mha = MultiHeadAttention(embed_dim, nheads)
        self.cudnn_mha.weight.set_value(self.weight)

        self.q_tensor_cudnn = paddle.to_tensor(
            self.query, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor_cudnn = paddle.to_tensor(
            self.key, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor_cudnn = paddle.to_tensor(
            self.value, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.q_tensor_ref = paddle.to_tensor(
            self.query, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor_ref = paddle.to_tensor(
            self.key, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor_ref = paddle.to_tensor(
            self.value, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.attn_tensor = paddle.to_tensor(
            np.ones((batch_size, nheads, seqlen, seqlen)),
            dtype=np.int32,
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
            ref_output, cudnn_output = self.compute_output()

        self.assertTrue(
            compare(ref_output.numpy(),
                    cudnn_output.numpy(), self.atol, self.rtol),
            "[Test{}MultiHeadAttention - cuDNN] outputs are miss-matched.".
            format(get_dtype_str(self.dtype)))

    def test_full_grads(self):
        self.q_tensor_cudnn.stop_gradient = False
        self.k_tensor_cudnn.stop_gradient = False
        self.v_tensor_cudnn.stop_gradient = False
        self.q_tensor_ref.stop_gradient = False
        self.k_tensor_ref.stop_gradient = False
        self.v_tensor_ref.stop_gradient = False

        self._cehck_grads()

    def test_weight_grads_only(self):
        self.q_tensor_cudnn.stop_gradient = True
        self.k_tensor_cudnn.stop_gradient = True
        self.v_tensor_cudnn.stop_gradient = True
        self.q_tensor_ref.stop_gradient = True
        self.k_tensor_ref.stop_gradient = True
        self.v_tensor_ref.stop_gradient = True

        self._cehck_grads(False)

    def compute_output(self):
        ref_output = self.ref_mha(self.q_tensor_ref, self.k_tensor_ref,
                                  self.v_tensor_ref, self.attn_tensor)
        cudnn_output = self.cudnn_mha(self.q_tensor_cudnn, self.k_tensor_cudnn,
                                      self.v_tensor_cudnn, self.attn_tensor)
        return ref_output, cudnn_output

    def _cehck_grads(self, check_data_grads=True):

        enable_amp = True if self.dtype == np.float16 else False

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        with paddle.amp.auto_cast(enable=enable_amp, custom_white_list={'mha'}):
            ref_output, cudnn_output = self.compute_output()

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
            "[Test{}cuDNNMultiHeadAttention] weight_grads are miss-matched.".
            format(get_dtype_str(self.dtype)))
        if check_data_grads:
            self.assertTrue(
                compare(self.q_tensor_ref.grad.numpy(),
                        self.q_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[Test{}cuDNNMultiHeadAttention] query_grads are miss-matched.".
                format(get_dtype_str(self.dtype)))
            self.assertTrue(
                compare(self.k_tensor_ref.grad.numpy(),
                        self.k_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[Test{}cuDNNMultiHeadAttention] key_grads are miss-matched.".
                format(get_dtype_str(self.dtype)))
            self.assertTrue(
                compare(self.v_tensor_ref.grad.numpy(),
                        self.v_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[Test{}cuDNNMultiHeadAttention] value_grads are miss-matched.".
                format(get_dtype_str(self.dtype)))

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


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestFP16cuDNNMultiHeadAttentionDynamic(
        TestFP32cuDNNMultiHeadAttentionDynamic):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3
        self.rtol = 1e-4


if __name__ == "__main__":
    unittest.main()
