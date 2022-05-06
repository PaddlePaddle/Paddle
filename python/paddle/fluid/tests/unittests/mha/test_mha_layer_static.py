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
class TestFP32cuDNNMultiHeadAttentionStatic(unittest.TestCase):
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
            with paddle.static.amp.fp16_guard():
                self.ref_mha_loss = self.compute_ref_loss(
                    q_input_ref, k_input_ref, v_input_ref, attn_mask_ref)

                paddle.static.append_backward(loss=self.ref_mha_loss)

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
            q_input_cudnn = paddle.static.data(
                name="q_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            k_input_cudnn = paddle.static.data(
                name="k_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            v_input_cudnn = paddle.static.data(
                name="v_input_cudnn",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            attn_mask_cudnn = paddle.static.data(
                name="attn_mask_cudnn",
                shape=[-1, self.nheads, self.seqlen, self.seqlen],
                dtype="int32")

            q_input_cudnn.stop_gradient = False
            k_input_cudnn.stop_gradient = False
            v_input_cudnn.stop_gradient = False

            self.cudnn_mha = MultiHeadAttention(self.embed_dim, self.nheads)
            with paddle.static.amp.fp16_guard():
                self.cudnn_mha_loss = self.compute_cudnn_loss(
                    q_input_cudnn, k_input_cudnn, v_input_cudnn,
                    attn_mask_cudnn)

                paddle.static.append_backward(loss=self.cudnn_mha_loss)

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

    def test_fwd_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return

        ref_out = self.exe.run(self.ref_main_prog,
                               feed={
                                   "q_input_ref": self.query,
                                   "k_input_ref": self.key,
                                   "v_input_ref": self.value,
                                   "attn_mask_ref": self.attn_mask
                               },
                               fetch_list=[self.ref_mha_loss.name])

        cudnn_out = self.exe.run(self.cudnn_main_prog,
                                 feed={
                                     "q_input_cudnn": self.query,
                                     "k_input_cudnn": self.key,
                                     "v_input_cudnn": self.value,
                                     "attn_mask_cudnn": self.attn_mask
                                 },
                                 fetch_list=[self.cudnn_mha_loss.name])

        self.assertTrue(
            compare(
                np.array(ref_out), np.array(cudnn_out), self.atol, self.rtol),
            "[Test{}cuDNNMultiHeadAttentionStatic] outputs are miss-matched.".
            format(get_dtype_str(self.dtype)))

    def test_grads(self):

        q_input_ref_grad, k_input_ref_grad, v_input_ref_grad, \
        wq_grad, wk_grad, wv_grad, wo_grad, \
        bq_grad, bk_grad, bv_grad, bo_grad, = self.exe.run(self.ref_main_prog,
                    feed={"q_input_ref": self.query,
                        "k_input_ref": self.key,
                        "v_input_ref": self.value,
                        "attn_mask_ref": self.attn_mask},
                    fetch_list=["q_input_ref@GRAD", "k_input_ref@GRAD", "v_input_ref@GRAD",
                                "{}.w_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.out_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.out_proj.full_name()),])

        q_input_cudnn_grad, k_input_cudnn_grad, v_input_cudnn_grad, \
        weight_cudnn_grad = self.exe.run(self.cudnn_main_prog,
                        feed={"q_input_cudnn": self.query,
                            "k_input_cudnn": self.key,
                            "v_input_cudnn": self.value,
                            "attn_mask_cudnn": self.attn_mask},
                        fetch_list=["q_input_cudnn@GRAD", "k_input_cudnn@GRAD", "v_input_cudnn@GRAD",
                            "{}.w_0@GRAD".format(self.cudnn_mha.full_name())])

        weight_ref_grad = np.concatenate(
            (
                np.array(wq_grad).flatten(),
                np.array(wk_grad).flatten(),
                np.array(wv_grad).flatten(),
                np.array(wo_grad).flatten(),
                np.array(bq_grad).flatten(),
                np.array(bk_grad).flatten(),
                np.array(bv_grad).flatten(),
                np.array(bo_grad).flatten(), ),
            axis=0)

        self.assertTrue(
            compare(weight_ref_grad, weight_cudnn_grad, self.atol, self.rtol),
            "[Test{}cuDNNMultiHeadAttentionStatic] weight_grads are miss-matched.".
            format(get_dtype_str(self.dtype)))
        self.assertTrue(
            compare(q_input_ref_grad, q_input_cudnn_grad, self.atol, self.rtol),
            "[Test{}cuDNNMultiHeadAttentionStatic] query_grads are miss-matched.".
            format(get_dtype_str(self.dtype)))
        self.assertTrue(
            compare(k_input_ref_grad, k_input_cudnn_grad, self.atol, self.rtol),
            "[Test{}cuDNNMultiHeadAttentionStatic] key_grads are miss-matched.".
            format(get_dtype_str(self.dtype)))
        self.assertTrue(
            compare(v_input_ref_grad, v_input_cudnn_grad, self.atol, self.rtol),
            "[Test{}cuDNNMultiHeadAttentionStatic] value_grads are miss-matched.".
            format(get_dtype_str(self.dtype)))

    def compute_ref_loss(self, qeury, key, value, mask):
        ref_mha_output = self.ref_mha(qeury, key, value, mask)
        return paddle.mean(ref_mha_output)

    def compute_cudnn_loss(self, qeury, key, value, mask):
        cudnn_mha_output = self.cudnn_mha(qeury, key, value, mask)
        return paddle.mean(cudnn_mha_output)


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestFP16cuDNNMultiHeadAttentionStatic(
        TestFP32cuDNNMultiHeadAttentionStatic):
    def setUp(self):
        super().setUp()
        ori_fp16_var_list = paddle.static.amp.cast_model_to_fp16(
            self.ref_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, self.ref_main_prog, to_fp16_var_names=ori_fp16_var_list)

        cudnn_fp16_var_list = paddle.static.amp.cast_model_to_fp16(
            self.cudnn_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place,
            self.cudnn_main_prog,
            to_fp16_var_names=cudnn_fp16_var_list)

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3
        self.rtol = 1e-4


if __name__ == '__main__':
    unittest.main()
