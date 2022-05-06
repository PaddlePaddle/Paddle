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
from paddle.static import InputSpec
from paddle.nn import MultiHeadAttention
from utils import compare, generate_weight, generate_data
from utils import MultiHeadAttentionRef, skip_unit_test

_NHEADS = 4
_SEQLEN = 4
_EMBED_DIM = 8


class cuDNNMultiHeadAttention(paddle.nn.Layer):
    def __init__(self, embed_dim, nheads):
        super(cuDNNMultiHeadAttention, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, nheads)

    @paddle.jit.to_static(input_spec=[
        InputSpec(
            shape=[None, _SEQLEN, _EMBED_DIM], dtype='float32'), InputSpec(
                shape=[None, _SEQLEN, _EMBED_DIM], dtype='float32'), InputSpec(
                    shape=[None, _SEQLEN, _EMBED_DIM], dtype='float32'),
        InputSpec(
            shape=[None, _NHEADS, _SEQLEN, _SEQLEN], dtype='int32')
    ])
    def forward(self, query, key, value, attn_mask):
        return self.mha(query, key, value, attn_mask)


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestcuDNNMultiHeadAttentionWithJitToStatic(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        nheads = _NHEADS
        seqlen = _SEQLEN
        embed_dim = _EMBED_DIM

        paddle.disable_static()

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.query, self.key, self.value = generate_data(batch_size, seqlen,
                                                         embed_dim, self.dtype)
        self.weight,  self.q_proj_weight, self.q_proj_bias, \
            self.k_proj_weight, self.k_proj_bias, \
            self.v_proj_weight, self.v_proj_bias, \
            self.out_proj_weight, self.out_proj_bias = generate_weight(embed_dim, self.dtype)

        self.ref_mha = MultiHeadAttentionRef(embed_dim, nheads)
        self.ref_mha.q_proj.weight.set_value(self.q_proj_weight)
        self.ref_mha.k_proj.weight.set_value(self.k_proj_weight)
        self.ref_mha.v_proj.weight.set_value(self.v_proj_weight)
        self.ref_mha.out_proj.weight.set_value(self.out_proj_weight)
        self.ref_mha.q_proj.bias.set_value(self.q_proj_bias)
        self.ref_mha.k_proj.bias.set_value(self.k_proj_bias)
        self.ref_mha.v_proj.bias.set_value(self.v_proj_bias)
        self.ref_mha.out_proj.bias.set_value(self.out_proj_bias)

        self.cudnn_mha = cuDNNMultiHeadAttention(embed_dim, nheads)
        self.cudnn_mha.mha.weight.set_value(self.weight)

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
        ref_output = self.ref_mha(self.q_tensor_ref, self.k_tensor_ref,
                                  self.v_tensor_ref, self.attn_tensor)
        cudnn_output = self.cudnn_mha(self.q_tensor_cudnn, self.k_tensor_cudnn,
                                      self.v_tensor_cudnn, self.attn_tensor)
        self.assertTrue(
            compare(ref_output.numpy(),
                    cudnn_output.numpy(), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitToStatic] outputs are miss-matched."
        )

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

    def _cehck_grads(self, check_data_grads=True):
        ref_output = self.ref_mha(self.q_tensor_ref, self.k_tensor_ref,
                                  self.v_tensor_ref, self.attn_tensor)
        cudnn_output = self.cudnn_mha(self.q_tensor_cudnn, self.k_tensor_cudnn,
                                      self.v_tensor_cudnn, self.attn_tensor)

        ref_loss = paddle.mean(ref_output)
        cudnn_loss = paddle.mean(cudnn_output)

        paddle.autograd.backward([ref_loss, cudnn_loss])

        ref_weight_grad = self._get_grads_from_ref()
        cudnn_weight_grad = self.cudnn_mha.mha.weight.grad.numpy()

        self.assertTrue(
            compare(ref_weight_grad, cudnn_weight_grad, self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitToStatic] weight_grads are miss-matched."
        )
        if check_data_grads:
            self.assertTrue(
                compare(self.q_tensor_ref.grad.numpy(),
                        self.q_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[TestcuDNNMultiHeadAttentionWithJitToStatic] query_grads are miss-matched."
            )
            self.assertTrue(
                compare(self.k_tensor_ref.grad.numpy(),
                        self.k_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[TestcuDNNMultiHeadAttentionWithJitToStatic] key_grads are miss-matched."
            )
            self.assertTrue(
                compare(self.v_tensor_ref.grad.numpy(),
                        self.v_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
                "[TestcuDNNMultiHeadAttentionWithJitToStatic] value_grads are miss-matched."
            )

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
class TestcuDNNMultiHeadAttentionWithJitSave(
        TestcuDNNMultiHeadAttentionWithJitToStatic):
    def setUp(self):
        super(TestcuDNNMultiHeadAttentionWithJitSave, self).setUp()
        self.q_tensor_cudnn.stop_gradient = False
        self.k_tensor_cudnn.stop_gradient = False
        self.v_tensor_cudnn.stop_gradient = False
        self.q_tensor_ref.stop_gradient = False
        self.k_tensor_ref.stop_gradient = False
        self.v_tensor_ref.stop_gradient = False

    def test_jit_save_and_load(self):
        path = '/tmp/paddle_mha_layer_jit_save'
        paddle.fluid.dygraph.jit._clear_save_pre_hooks()
        paddle.jit.save(self.cudnn_mha, path)

        loaded_cudnn_mha = paddle.jit.load(path)
        loaded_cudnn_mha.train()

        ref_output = self.ref_mha(self.q_tensor_ref, self.k_tensor_ref,
                                  self.v_tensor_ref, self.attn_tensor)
        cudnn_output = loaded_cudnn_mha(self.q_tensor_cudnn,
                                        self.k_tensor_cudnn,
                                        self.v_tensor_cudnn, self.attn_tensor)

        self.assertTrue(
            compare(ref_output.numpy(),
                    cudnn_output.numpy(), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitSave] outputs are miss-matched.")

        ref_loss = paddle.mean(ref_output)
        cudnn_loss = paddle.mean(cudnn_output)

        paddle.autograd.backward([ref_loss, cudnn_loss])

        ref_weight_grad = self._get_grads_from_ref()
        for key in loaded_cudnn_mha._parameters:
            cudnn_weight_grad = loaded_cudnn_mha._parameters[key].grad.numpy()
            break
        self.assertTrue(
            compare(ref_weight_grad, cudnn_weight_grad, self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitSave] weight_grads are miss-matched."
        )

        self.assertTrue(
            compare(self.q_tensor_ref.grad.numpy(),
                    self.q_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitSave] query_grads are miss-matched."
        )
        self.assertTrue(
            compare(self.k_tensor_ref.grad.numpy(),
                    self.k_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitSave] key_grads are miss-matched."
        )
        self.assertTrue(
            compare(self.v_tensor_ref.grad.numpy(),
                    self.v_tensor_cudnn.grad.numpy(), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithJitSave] value_grads are miss-matched."
        )


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestcuDNNMultiHeadAttentionSaveInferenceModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seqlen = 4
        self.embed_dim = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.init_dtype_type()

        self.query, self.key, self.value = generate_data(
            self.batch_size, self.seqlen, self.embed_dim, self.dtype)
        self.weight,  self.q_proj_weight, self.q_proj_bias, \
            self.k_proj_weight, self.k_proj_bias, \
            self.v_proj_weight, self.v_proj_bias, \
            self.out_proj_weight, self.out_proj_bias = generate_weight(self.embed_dim, np.single)

        # Paddle's MHA Setup ====================
        self.ref_main_prog = paddle.static.Program()
        self.ref_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.ref_main_prog,
                                         self.ref_startup_prog):
            q_input_ref = paddle.static.data(
                name="q_input_ref",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            k_input_ref = paddle.static.data(
                name="k_input_ref",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            v_input_ref = paddle.static.data(
                name="v_input_ref",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            attn_mask_ref = paddle.static.data(
                name="attn_mask_ref",
                shape=[-1, self.nheads, self.seqlen, self.seqlen],
                dtype="int32")

            q_input_ref.stop_gradient = False
            k_input_ref.stop_gradient = False
            v_input_ref.stop_gradient = False

            self.ref_mha = MultiHeadAttentionRef(self.embed_dim, self.nheads)

            ref_mha_output = self.ref_mha(q_input_ref, k_input_ref, v_input_ref,
                                          attn_mask_ref)
            self.ref_mha_loss = paddle.mean(ref_mha_output)

        self.exe.run(self.ref_startup_prog)
        with paddle.static.program_guard(self.ref_main_prog,
                                         self.ref_startup_prog):
            self.ref_mha.q_proj.weight.set_value(self.q_proj_weight)
            self.ref_mha.k_proj.weight.set_value(self.k_proj_weight)
            self.ref_mha.v_proj.weight.set_value(self.v_proj_weight)
            self.ref_mha.out_proj.weight.set_value(self.out_proj_weight)
            self.ref_mha.q_proj.bias.set_value(self.q_proj_bias)
            self.ref_mha.k_proj.bias.set_value(self.k_proj_bias)
            self.ref_mha.v_proj.bias.set_value(self.v_proj_bias)
            self.ref_mha.out_proj.bias.set_value(self.out_proj_bias)

        # cuDNN's MHA Setup ====================
        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            self.q_input_cudnn = paddle.static.data(
                name="q_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            self.k_input_cudnn = paddle.static.data(
                name="k_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            self.v_input_cudnn = paddle.static.data(
                name="v_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            self.attn_mask_cudnn = paddle.static.data(
                name="attn_mask_cudnn",
                shape=[-1, self.nheads, self.seqlen, self.seqlen],
                dtype="int32")

            self.q_input_cudnn.stop_gradient = False
            self.k_input_cudnn.stop_gradient = False
            self.v_input_cudnn.stop_gradient = False

            self.cudnn_mha = MultiHeadAttention(self.embed_dim, self.nheads)

            cudnn_mha_output = self.cudnn_mha(
                self.q_input_cudnn, self.k_input_cudnn, self.v_input_cudnn,
                self.attn_mask_cudnn)
            self.cudnn_mha_loss = paddle.mean(cudnn_mha_output)

        self.exe.run(self.cudnn_startup_prog)
        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            self.cudnn_mha.weight.set_value(self.weight)

        self.attn_mask = np.ones(
            (self.batch_size, self.nheads, self.seqlen, self.seqlen),
            dtype=np.int32)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-4

    def test_jit_save_and_load(self):
        ref_output = self.exe.run(self.ref_main_prog,
                                  feed={
                                      "q_input_ref": self.query,
                                      "k_input_ref": self.key,
                                      "v_input_ref": self.value,
                                      "attn_mask_ref": self.attn_mask
                                  },
                                  fetch_list=[self.ref_mha_loss.name])

        path = '/tmp/paddle_mha_layer_save_inference_model'
        paddle.static.save_inference_model(
            path, [
                self.q_input_cudnn, self.k_input_cudnn, self.v_input_cudnn,
                self.attn_mask_cudnn
            ], [self.cudnn_mha_loss],
            self.exe,
            program=self.cudnn_main_prog)

        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(path, self.exe))
        cudnn_output = self.exe.run(inference_program,
                                    feed={
                                        feed_target_names[0]: self.query,
                                        feed_target_names[1]: self.key,
                                        feed_target_names[2]: self.value,
                                        feed_target_names[3]: self.attn_mask
                                    },
                                    fetch_list=fetch_targets)
        self.assertTrue(
            compare(
                np.array(ref_output),
                np.array(cudnn_output), self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionSaveInferenceModel] outputs are miss-matched."
        )


if __name__ == "__main__":
    unittest.main()
