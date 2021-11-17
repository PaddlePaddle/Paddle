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
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    K = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    V = (np.random.random(
        (batch_size, max_seq_len, 1, vec_size)) - .5).astype(dtype)
    W = (np.random.random((4 * vec_size * vec_size, )) - .5).astype(np.single)
    W = np.concatenate((W, np.zeros((4*vec_size,))), dtype=np.single)

    stride = vec_size * vec_size
    WQ = W[0:stride].reshape((vec_size, vec_size))
    WK = W[stride:2 * stride].reshape((vec_size, vec_size))
    WV = W[2 * stride:3 * stride].reshape((vec_size, vec_size))
    WO = W[3 * stride:4 * stride].reshape((vec_size, vec_size))

    return (Q, K, V, W, WQ, WK, WV, WO)

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP32CUDNNMHALayerStatic(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seq_len = 4
        self.vec_size = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = _generate_data(self.batch_size, self.seq_len, self.vec_size, self.dtype)

        # Paddle's MHA Setup ====================
        self.ref_main_prog = paddle.static.Program()
        self.ref_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.ref_main_prog, self.ref_startup_prog):
            q_input_3dim = paddle.static.data(name="q_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            k_input_3dim = paddle.static.data(name="k_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            v_input_3dim = paddle.static.data(name="v_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            attn_mask_4dim = paddle.static.data(name="attn_mask_4dim", shape=[-1, self.nheads, self.seq_len, self.seq_len], dtype="int32")

            q_input_3dim.stop_gradient = False
            k_input_3dim.stop_gradient = False
            v_input_3dim.stop_gradient = False

            self.ref_mha = MultiHeadAttention(self.vec_size, self.nheads)
            with paddle.static.amp.fp16_guard():
                ref_mha_output = self.ref_mha(q_input_3dim, k_input_3dim, v_input_3dim, attn_mask_4dim)
                self.ref_mha_loss = paddle.mean(ref_mha_output)

                paddle.static.append_backward(loss=self.ref_mha_loss)

        self.exe.run(self.ref_startup_prog)
        with paddle.static.program_guard(self.ref_main_prog, self.ref_startup_prog):
            self.ref_mha.q_proj.weight.set_value(self.WQ)
            self.ref_mha.k_proj.weight.set_value(self.WK)
            self.ref_mha.v_proj.weight.set_value(self.WV)
            self.ref_mha.out_proj.weight.set_value(self.WO)

        # cuDNN's MHA Setup ====================
        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog, self.cudnn_startup_prog):
            q_input = paddle.static.data(name="q_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            k_input = paddle.static.data(name="k_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            v_input = paddle.static.data(name="v_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            attn_mask_input = paddle.static.data(name="attn_mask", shape=[-1, self.seq_len], dtype="int32")

            q_input.stop_gradient = False
            k_input.stop_gradient = False
            v_input.stop_gradient = False

            seq_info_infer = CUDNNSeqInfoInfer()
            self.cudnn_mha = CUDNNMultiHeadAttention(self.vec_size, self.nheads)
            with paddle.static.amp.fp16_guard():
                seq_info = seq_info_infer(attn_mask_input)
                cudnn_mha_output = self.cudnn_mha(q_input, k_input, v_input, seq_info)
                self.cudnn_mha_loss = paddle.mean(cudnn_mha_output)

                paddle.static.append_backward(loss=self.cudnn_mha_loss)

        self.exe.run(self.cudnn_startup_prog)
        with paddle.static.program_guard(self.cudnn_main_prog, self.cudnn_startup_prog):
            self.cudnn_mha.weight.set_value(self.W)

        self.attn_mask_for_ori = np.ones((self.batch_size, self.nheads, self.seq_len, self.seq_len), dtype=np.int32)
        self.attn_mask_for_cudnn = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_fwd_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return

        ref_out = self.exe.run(self.ref_main_prog,
                        feed={"q_input_3dim": self.Q.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "k_input_3dim": self.K.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "v_input_3dim": self.V.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "attn_mask_4dim": self.attn_mask_for_ori},
                        fetch_list=[self.ref_mha_loss.name])

        cudnn_out = self.exe.run(self.cudnn_main_prog,
                        feed={"q_input": self.Q,
                            "k_input": self.K,
                            "v_input": self.V,
                            "attn_mask": self.attn_mask_for_cudnn},
                        fetch_list=[self.cudnn_mha_loss.name])

        self.assertTrue(compare(np.array(ref_out), np.array(cudnn_out), self.atol, self.rtol),
                        "[Test*CUDNNMHALayer-Static] outputs are miss-matched.")
        print(f'CUDNNMultiHeadAttention Layer [Static] {self.dtype} fwd passed.')


    def test_grads(self):

        q_input_3dim_grad, k_input_3dim_grad, v_input_3dim_grad, \
        wq_grad, wk_grad, wv_grad, wo_grad, \
        bq_grad, bk_grad, bv_grad, bo_grad, = self.exe.run(self.ref_main_prog,
                    feed={"q_input_3dim": self.Q.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "k_input_3dim": self.K.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "v_input_3dim": self.V.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "attn_mask_4dim": self.attn_mask_for_ori},
                    fetch_list=["q_input_3dim@GRAD", "k_input_3dim@GRAD", "v_input_3dim@GRAD",
                                "{}.w_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.out_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.out_proj.full_name()),])

        q_input_grad, k_input_grad, v_input_grad, \
        cdunn_w_grad = self.exe.run(self.cudnn_main_prog,
                        feed={"q_input": self.Q,
                            "k_input": self.K,
                            "v_input": self.V,
                            "attn_mask": self.attn_mask_for_cudnn},
                        fetch_list=["q_input@GRAD", "k_input@GRAD", "v_input@GRAD",
                            "{}.w_0@GRAD".format(self.cudnn_mha.full_name())])

        ref_weight_grad = np.concatenate(
                            (np.array(wq_grad).flatten(), np.array(wk_grad).flatten(), 
                             np.array(wv_grad).flatten(), np.array(wo_grad).flatten(),
                             np.array(bq_grad).flatten(), np.array(bk_grad).flatten(), 
                             np.array(bv_grad).flatten(), np.array(bo_grad).flatten(),),
                            axis=0)

        self.assertTrue(compare(ref_weight_grad, cdunn_w_grad, self.atol, self.rtol),
            "[Test*CUDNNMHALayer-Static] weight_grads are miss-matched.")
        self.assertTrue(
            compare(q_input_3dim_grad, q_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayer-Static] Q_grads are miss-matched.")
        self.assertTrue(
            compare(k_input_3dim_grad, k_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayer-Static] K_grads are miss-matched.")
        self.assertTrue(
            compare(v_input_3dim_grad, v_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayer-Static] V_grads are miss-matched.")

        print(f'CUDNNMultiHeadAttention [Static] Layer {self.dtype} bwd passed.')

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16CUDNNMHALayerStatic(TestFP32CUDNNMHALayerStatic):
    def setUp(self):
        super().setUp()
        ori_fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.ref_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(self.place, self.ref_main_prog, to_fp16_var_names=ori_fp16_var_list)

        cudnn_fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.cudnn_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(self.place, self.cudnn_main_prog, to_fp16_var_names=cudnn_fp16_var_list)

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3
        self.rtol = 1e-2

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP32CUDNNMHALayerStaticWithSeqDataCache(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seq_len = 4
        self.vec_size = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = _generate_data(self.batch_size, self.seq_len, self.vec_size, self.dtype)

        # Paddle's MHA Setup ====================
        self.ref_main_prog = paddle.static.Program()
        self.ref_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.ref_main_prog, self.ref_startup_prog):
            q_input_3dim = paddle.static.data(name="q_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            k_input_3dim = paddle.static.data(name="k_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            v_input_3dim = paddle.static.data(name="v_input_3dim", shape=[-1, self.seq_len, self.vec_size], dtype='float32')
            attn_mask_4dim = paddle.static.data(name="attn_mask_4dim", shape=[-1, self.nheads, self.seq_len, self.seq_len], dtype="int32")

            q_input_3dim.stop_gradient = False
            k_input_3dim.stop_gradient = False
            v_input_3dim.stop_gradient = False

            self.ref_mha = MultiHeadAttention(self.vec_size, self.nheads)
            with paddle.static.amp.fp16_guard():
                ref_mha_output = self.ref_mha(q_input_3dim, k_input_3dim, v_input_3dim, attn_mask_4dim)
                self.ref_mha_loss = paddle.mean(ref_mha_output)

                paddle.static.append_backward(loss=self.ref_mha_loss)

        self.exe.run(self.ref_startup_prog)
        with paddle.static.program_guard(self.ref_main_prog, self.ref_startup_prog):
            self.ref_mha.q_proj.weight.set_value(self.WQ)
            self.ref_mha.k_proj.weight.set_value(self.WK)
            self.ref_mha.v_proj.weight.set_value(self.WV)
            self.ref_mha.out_proj.weight.set_value(self.WO)

        # cuDNN's MHA Setup ====================
        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog, self.cudnn_startup_prog):
            q_input = paddle.static.data(name="q_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            k_input = paddle.static.data(name="k_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            v_input = paddle.static.data(name="v_input", shape=[-1, self.seq_len, 1, self.vec_size], dtype='float32')
            attn_mask_input = paddle.static.data(name="attn_mask", shape=[-1, self.seq_len], dtype="int32")

            q_input.stop_gradient = False
            k_input.stop_gradient = False
            v_input.stop_gradient = False

            seq_info_infer = CUDNNSeqInfoInfer()
            self.cudnn_mha = CUDNNMultiHeadAttention(self.vec_size, self.nheads, seq_data_infer=seq_info_infer)
            with paddle.static.amp.fp16_guard():
                seq_info = seq_info_infer(attn_mask_input)
                cudnn_mha_output = self.cudnn_mha(q_input, k_input, v_input, seq_info)
                self.cudnn_mha_loss = paddle.mean(cudnn_mha_output)

                paddle.static.append_backward(loss=self.cudnn_mha_loss)

        self.exe.run(self.cudnn_startup_prog)
        with paddle.static.program_guard(self.cudnn_main_prog, self.cudnn_startup_prog):
            self.cudnn_mha.weight.set_value(self.W)

        self.attn_mask_for_ori = np.ones((self.batch_size, self.nheads, self.seq_len, self.seq_len), dtype=np.int32)
        self.attn_mask_for_cudnn = np.ones((self.batch_size, self.seq_len), dtype=np.int32)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_fwd_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return

        ref_out = self.exe.run(self.ref_main_prog,
                        feed={"q_input_3dim": self.Q.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "k_input_3dim": self.K.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "v_input_3dim": self.V.reshape((self.batch_size, self.seq_len, self.vec_size)),
                            "attn_mask_4dim": self.attn_mask_for_ori},
                        fetch_list=[self.ref_mha_loss.name])

        cudnn_out = self.exe.run(self.cudnn_main_prog,
                        feed={"q_input": self.Q,
                            "k_input": self.K,
                            "v_input": self.V,
                            "attn_mask": self.attn_mask_for_cudnn},
                        fetch_list=[self.cudnn_mha_loss.name])

        self.assertTrue(compare(np.array(ref_out), np.array(cudnn_out), self.atol, self.rtol),
                "[Test*CUDNNMHALayerWithSeqDataCache-Static] outputs are miss-matched.")
        print(f'CUDNNMHALayerStaticWithSeqDataCache Layer [Static] {self.dtype} fwd passed.')


    def test_grads(self):

        q_input_3dim_grad, k_input_3dim_grad, v_input_3dim_grad, \
        wq_grad, wk_grad, wv_grad, wo_grad, \
        bq_grad, bk_grad, bv_grad, bo_grad, = self.exe.run(self.ref_main_prog,
                    feed={"q_input_3dim": self.Q.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "k_input_3dim": self.K.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "v_input_3dim": self.V.reshape((self.batch_size, self.seq_len, self.vec_size)),
                        "attn_mask_4dim": self.attn_mask_for_ori},
                    fetch_list=["q_input_3dim@GRAD", "k_input_3dim@GRAD", "v_input_3dim@GRAD",
                                "{}.w_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.w_0@GRAD".format(self.ref_mha.out_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.q_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.k_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.v_proj.full_name()),
                                "{}.b_0@GRAD".format(self.ref_mha.out_proj.full_name()),])

        q_input_grad, k_input_grad, v_input_grad, \
        cdunn_w_grad = self.exe.run(self.cudnn_main_prog,
                        feed={"q_input": self.Q,
                            "k_input": self.K,
                            "v_input": self.V,
                            "attn_mask": self.attn_mask_for_cudnn},
                        fetch_list=["q_input@GRAD", "k_input@GRAD", "v_input@GRAD",
                            "{}.w_0@GRAD".format(self.cudnn_mha.full_name())])

        ref_weight_grad = np.concatenate(
                            (np.array(wq_grad).flatten(), np.array(wk_grad).flatten(), 
                             np.array(wv_grad).flatten(), np.array(wo_grad).flatten(),
                             np.array(bq_grad).flatten(), np.array(bk_grad).flatten(), 
                             np.array(bv_grad).flatten(), np.array(bo_grad).flatten(),),
                            axis=0)
        self.assertTrue(compare(ref_weight_grad, cdunn_w_grad, self.atol, self.rtol),
            "[Test*CUDNNMHALayerWithSeqDataCache-Static] weight_grads are miss-matched.")
        self.assertTrue(
            compare(q_input_3dim_grad, q_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayerWithSeqDataCache-Static] Q_grads are miss-matched.")
        self.assertTrue(
            compare(k_input_3dim_grad, k_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayerWithSeqDataCache-Static] K_grads are miss-matched.")
        self.assertTrue(
            compare(v_input_3dim_grad, v_input_grad, self.atol, self.rtol),
                    "[Test*CUDNNMHALayerWithSeqDataCache-Static] V_grads are miss-matched.")

        print(f'UDNNMHALayerStaticWithSeqDataCache [Static] Layer {self.dtype} bwd passed.')

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16CUDNNMHALayerStaticWithSeqDataCache(TestFP32CUDNNMHALayerStaticWithSeqDataCache):
    def setUp(self):
        super().setUp()
        ori_fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.ref_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(self.place, self.ref_main_prog, to_fp16_var_names=ori_fp16_var_list)

        cudnn_fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.cudnn_main_prog)
        paddle.static.amp.cast_parameters_to_fp16(self.place, self.cudnn_main_prog, to_fp16_var_names=cudnn_fp16_var_list)

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3
        self.rtol = 1e-2
