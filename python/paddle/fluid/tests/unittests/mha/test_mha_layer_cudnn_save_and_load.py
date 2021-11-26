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
from paddle.static import InputSpec


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


class CUDNNMHAWithSeqInfer(paddle.nn.Layer):
    def __init__(self, hidden, heads):
        super(CUDNNMHAWithSeqInfer, self).__init__()
        self.seq_info_infer = CUDNNSeqInfoInfer()
        self.cudnn_mha = CUDNNMultiHeadAttention(
            hidden, heads, seq_data_infer=self.seq_info_infer)

    @paddle.jit.to_static(input_spec=[
        InputSpec(
            shape=[None, 4, 8], dtype='float32'), InputSpec(
                shape=[None, 4, 8], dtype='float32'), InputSpec(
                    shape=[None, 4, 8], dtype='float32'), InputSpec(
                        shape=[None, 4], dtype='int32')
    ])
    def forward(self, query, key, value, attn_mask):
        seq_data_info = self.seq_info_infer(attn_mask)
        return self.cudnn_mha(query, key, value, seq_data_info)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNMHALayerJitSaving(unittest.TestCase):
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

        self.cudnn_mha = CUDNNMHAWithSeqInfer(vec_size, nheads)
        self.cudnn_mha.cudnn_mha.weight.set_value(self.W)

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

        self.q_tensor.stop_gradient = False
        self.k_tensor.stop_gradient = False
        self.v_tensor.stop_gradient = False
        self.q_3dim_tensor.stop_gradient = False
        self.k_3dim_tensor.stop_gradient = False
        self.v_3dim_tensor.stop_gradient = False

        self.attn_mask = paddle.to_tensor(
            np.ones(
                (batch_size, seq_len), dtype=np.int32), place=self.place)
        self.attn_tensor = paddle.to_tensor(
            np.ones((batch_size, nheads, seq_len, seq_len)),
            dtype=np.bool,
            place=self.place)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_jit_save_and_load(self):
        path = '/tmp/paddle_mha_jit_save'
        paddle.jit.save(self.cudnn_mha, path)

        loaded_cudnn_mha = paddle.jit.load(path)
        loaded_cudnn_mha.train()

        ref_output = self.ref_mha(self.q_3dim_tensor, self.k_3dim_tensor,
                                  self.v_3dim_tensor, self.attn_tensor)
        cudnn_output = loaded_cudnn_mha(self.q_tensor, self.k_tensor,
                                        self.v_tensor, self.attn_mask)
        self.assertTrue(
            compare(ref_output.numpy(),
                    cudnn_output.numpy(), self.atol, self.rtol),
            "[TestCUDNNMHALayerJitSaving] outputs are miss-matched.")

        ref_loss = paddle.mean(ref_output)
        cudnn_loss = paddle.mean(cudnn_output)

        paddle.autograd.backward([ref_loss, cudnn_loss])

        ref_weight_grad = self._get_grads_from_ref()
        for key in loaded_cudnn_mha._parameters:
            cudnn_weight_grad = loaded_cudnn_mha._parameters[key].grad.numpy()
            break

        self.assertTrue(
            compare(ref_weight_grad, cudnn_weight_grad, self.atol, self.rtol),
            "[TestCUDNNMHALayerJitSaving] weight_grads are miss-matched.")
        self.assertTrue(
            compare(self.q_3dim_tensor.grad.numpy(),
                    self.q_tensor.grad.numpy(), self.atol, self.rtol),
            "[TestCUDNNMHALayerJitSaving] Q_grads are miss-matched.")
        self.assertTrue(
            compare(self.k_3dim_tensor.grad.numpy(),
                    self.k_tensor.grad.numpy(), self.atol, self.rtol),
            "[TestCUDNNMHALayerJitSaving] K_grads are miss-matched.")
        self.assertTrue(
            compare(self.v_3dim_tensor.grad.numpy(),
                    self.v_tensor.grad.numpy(), self.atol, self.rtol),
            "[TestCUDNNMHALayerJitSaving] V_grads are miss-matched.")

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
class TestCUDNNMHALayerSaveInferenceModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seq_len = 4
        self.vec_size = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.ref_exe = paddle.static.Executor(self.place)
        self.cudnn_exe = paddle.static.Executor(self.place)

        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = \
            _generate_data(self.batch_size, self.seq_len, self.vec_size, self.dtype)

        # Paddle's MHA Setup ====================
        self.ref_main_prog = paddle.static.Program()
        self.ref_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.ref_main_prog,
                                         self.ref_startup_prog):
            q_input_3dim = paddle.static.data(
                name="q_input_3dim",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            k_input_3dim = paddle.static.data(
                name="k_input_3dim",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            v_input_3dim = paddle.static.data(
                name="v_input_3dim",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            attn_mask_4dim = paddle.static.data(
                name="attn_mask_4dim",
                shape=[-1, self.nheads, self.seq_len, self.seq_len],
                dtype="int32")

            q_input_3dim.stop_gradient = False
            k_input_3dim.stop_gradient = False
            v_input_3dim.stop_gradient = False

            self.ref_mha = MultiHeadAttention(self.vec_size, self.nheads)
            ref_mha_output = self.ref_mha(q_input_3dim, k_input_3dim,
                                          v_input_3dim, attn_mask_4dim)
            self.ref_mha_loss = paddle.mean(ref_mha_output)

        self.ref_exe.run(self.ref_startup_prog)
        with paddle.static.program_guard(self.ref_main_prog,
                                         self.ref_startup_prog):
            self.ref_mha.q_proj.weight.set_value(self.WQ)
            self.ref_mha.k_proj.weight.set_value(self.WK)
            self.ref_mha.v_proj.weight.set_value(self.WV)
            self.ref_mha.out_proj.weight.set_value(self.WO)

        # cuDNN's MHA Setup ====================
        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            self.q_input = paddle.static.data(
                name="q_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            self.k_input = paddle.static.data(
                name="k_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            self.v_input = paddle.static.data(
                name="v_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            self.attn_mask_input = paddle.static.data(
                name="attn_mask", shape=[-1, self.seq_len], dtype="int32")

            self.q_input.stop_gradient = False
            self.k_input.stop_gradient = False
            self.v_input.stop_gradient = False

            seq_info_infer = CUDNNSeqInfoInfer()
            self.cudnn_mha = CUDNNMultiHeadAttention(self.vec_size, self.nheads)

            seq_info = seq_info_infer(self.attn_mask_input)
            cudnn_mha_output = self.cudnn_mha(self.q_input, self.k_input,
                                              self.v_input, seq_info)
            self.cudnn_mha_loss = paddle.mean(cudnn_mha_output)

        self.cudnn_exe.run(self.cudnn_startup_prog)
        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            self.cudnn_mha.weight.set_value(self.W)

        self.attn_mask_for_ori = np.ones(
            (self.batch_size, self.nheads, self.seq_len, self.seq_len),
            dtype=np.int32)
        self.attn_mask_for_cudnn = np.ones(
            (self.batch_size, self.seq_len), dtype=np.int32)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_jit_save_and_load(self):
        ref_out = self.ref_exe.run(self.ref_main_prog,
                                   feed={
                                       "q_input_3dim": self.Q,
                                       "k_input_3dim": self.K,
                                       "v_input_3dim": self.V,
                                       "attn_mask_4dim": self.attn_mask_for_ori
                                   },
                                   fetch_list=[self.ref_mha_loss.name])

        path = '/tmp/paddle_mha_save_inference'
        paddle.static.save_inference_model(
            path,
            [self.q_input, self.k_input, self.v_input, self.attn_mask_input],
            [self.cudnn_mha_loss],
            self.cudnn_exe,
            program=self.cudnn_main_prog)

        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(path, self.cudnn_exe))
        cudnn_out = self.cudnn_exe.run(inference_program,
                                       feed={
                                           feed_target_names[0]: self.Q,
                                           feed_target_names[1]: self.K,
                                           feed_target_names[2]: self.V,
                                           feed_target_names[3]:
                                           self.attn_mask_for_cudnn
                                       },
                                       fetch_list=fetch_targets)
        self.assertTrue(
            compare(
                np.array(ref_out), np.array(cudnn_out), self.atol, self.rtol),
            "[TestCUDNNMHALayerSaveInferenceModel] outputs are miss-matched.")


if __name__ == "__main__":
    unittest.main()
