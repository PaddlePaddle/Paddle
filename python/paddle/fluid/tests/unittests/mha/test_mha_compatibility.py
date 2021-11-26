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
import paddle.inference as paddle_infer


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

    def forward(self, query, key, value, attn_mask):
        seq_data_info = self.seq_info_infer(attn_mask)
        return self.cudnn_mha(query, key, value, seq_data_info)

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNMHALayerConvertToPaddleMHA(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        nheads = 4
        seq_len = 4
        vec_size = 8

        paddle.disable_static()

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = _generate_data(batch_size, seq_len, vec_size, self.dtype)

        self.cudnn_mha = CUDNNMHAWithSeqInfer(vec_size, nheads)
        self.cudnn_mha.cudnn_mha.weight.set_value(self.W)

        self.q_tensor = paddle.to_tensor(
            self.Q, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor = paddle.to_tensor(
            self.K, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor = paddle.to_tensor(
            self.V, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.attn_mask = np.ones((batch_size, nheads, seq_len, seq_len), dtype=np.int32)
        self.attn_mask_tensor = paddle.to_tensor(
                self.attn_mask,
                dtype=np.int32,
                place=self.place)

        self.output_ref = self.cudnn_mha(self.q_tensor, self.k_tensor, self.v_tensor, self.attn_mask_tensor)


        paddle.enable_static()

        self.path_with_coverting = '/tmp/paddle_mha_convert_with_to_static'
        self.exe = paddle.static.Executor(self.place)

        main_prog, startup_prog, inputs, outputs = \
            CUDNNMultiHeadAttention.convert_inference_program_with_paddleMHA_replacement(
                self.cudnn_mha, self.exe, [self.q_tensor, self.k_tensor, self.v_tensor, self.attn_mask_tensor])
        paddle.static.save_inference_model(self.path_with_coverting,
                                       inputs,
                                       outputs, self.exe, program=main_prog)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_compatibility_with_save_inference_model(self):
        [inference_program, feed_target_names, fetch_targets] = \
            paddle.static.load_inference_model(self.path_with_coverting, self.exe)

        loaded_output = self.exe.run(inference_program,
                feed={feed_target_names[0]: self.Q,
                        feed_target_names[1]: self.K,
                        feed_target_names[2]: self.V,
                        feed_target_names[3]: self.attn_mask},
                fetch_list=fetch_targets)

        self.assertTrue(
            compare(self.output_ref.numpy(), loaded_output[0], self.atol, self.rtol),
            "[TestCUDNNMHALayerConvertToPaddleMHA] outputs are miss-matched.")

    def test_with_converting_and_paddle_inference(self):
        model_file = self.path_with_coverting + ".pdmodel"
        params_file = self.path_with_coverting + ".pdiparams"

        config = paddle_infer.Config(model_file, params_file)
        config.enable_use_gpu(4096, 0)
        if core.is_compiled_with_tensorrt():
            config.enable_tensorrt_engine(max_batch_size=self.Q.shape[0])

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        q_handle = predictor.get_input_handle(input_names[0])
        k_handle = predictor.get_input_handle(input_names[1])
        v_handle = predictor.get_input_handle(input_names[2])
        mask_handle = predictor.get_input_handle(input_names[3])

        q_handle.reshape(self.Q.shape)
        q_handle.copy_from_cpu(self.Q)
        k_handle.reshape(self.K.shape)
        k_handle.copy_from_cpu(self.K)
        v_handle.reshape(self.V.shape)
        v_handle.copy_from_cpu(self.V)
        mask_handle.reshape(self.attn_mask.shape)
        mask_handle.copy_from_cpu(self.attn_mask)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        self.assertTrue(
            compare(self.output_ref.numpy(), output_data, self.atol, self.rtol),
            "[TestCUDNNMHALayerConvertToPaddleMHA] outputs are miss-matched.")


class CUDNNMHAWithSeqInferWithToStatic(paddle.nn.Layer):
    def __init__(self, hidden, heads):
        super(CUDNNMHAWithSeqInferWithToStatic, self).__init__()
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
class TestCUDNNMHALayerConvertToPaddleMHAWithJitToStatic(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        nheads = 4
        seq_len = 4
        vec_size = 8

        paddle.disable_static()

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.Q, self.K, self.V, self.W, \
        self.WQ, self.WK, self.WV, self.WO = _generate_data(batch_size, seq_len, vec_size, self.dtype)

        self.cudnn_mha = CUDNNMHAWithSeqInferWithToStatic(vec_size, nheads)
        self.cudnn_mha.cudnn_mha.weight.set_value(self.W)

        self.q_tensor = paddle.to_tensor(
            self.Q, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor = paddle.to_tensor(
            self.K, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor = paddle.to_tensor(
            self.V, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.attn_mask = np.ones((batch_size, nheads, seq_len, seq_len), dtype=np.int32)
        self.attn_mask_tensor = paddle.to_tensor(
                self.attn_mask,
                dtype=np.int32,
                place=self.place)

        self.output_ref = self.cudnn_mha(self.q_tensor, self.k_tensor, self.v_tensor, self.attn_mask_tensor)

        paddle.enable_static()

        self.path_with_coverting = '/tmp/paddle_mha_convert_with_to_static'
        self.exe = paddle.static.Executor(self.place)

        main_prog, startup_prog, inputs, outputs = \
            CUDNNMultiHeadAttention.convert_inference_program_with_paddleMHA_replacement(
                self.cudnn_mha, self.exe, [self.q_tensor, self.k_tensor, self.v_tensor, self.attn_mask_tensor])
        paddle.static.save_inference_model(self.path_with_coverting,
                                       inputs,
                                       outputs, self.exe, program=main_prog)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-3

    def test_compatibility_with_save_inference_model(self):
        [inference_program, feed_target_names, fetch_targets] = \
            paddle.static.load_inference_model(self.path_with_coverting, self.exe)

        loaded_output = self.exe.run(inference_program,
                feed={feed_target_names[0]: self.Q,
                        feed_target_names[1]: self.K,
                        feed_target_names[2]: self.V,
                        feed_target_names[3]: self.attn_mask},
                fetch_list=fetch_targets)

        self.assertTrue(
            compare(self.output_ref.numpy(), loaded_output[0], self.atol, self.rtol),
            "[TestCUDNNMHALayerConvertToPaddleMHAWithJitToStatic] outputs are miss-matched.")

    def test_with_converting_and_paddle_inference(self):
        model_file = self.path_with_coverting + ".pdmodel"
        params_file = self.path_with_coverting + ".pdiparams"

        config = paddle_infer.Config(model_file, params_file)
        config.enable_use_gpu(4096, 0)
        if core.is_compiled_with_tensorrt():
            config.enable_tensorrt_engine(max_batch_size=self.Q.shape[0])

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        q_handle = predictor.get_input_handle(input_names[0])
        k_handle = predictor.get_input_handle(input_names[1])
        v_handle = predictor.get_input_handle(input_names[2])
        mask_handle = predictor.get_input_handle(input_names[3])

        q_handle.reshape(self.Q.shape)
        q_handle.copy_from_cpu(self.Q)
        k_handle.reshape(self.K.shape)
        k_handle.copy_from_cpu(self.K)
        v_handle.reshape(self.V.shape)
        v_handle.copy_from_cpu(self.V)
        mask_handle.reshape(self.attn_mask.shape)
        mask_handle.copy_from_cpu(self.attn_mask)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        self.assertTrue(
            compare(self.output_ref.numpy(), output_data, self.atol, self.rtol),
            "[TestCUDNNMHALayerConvertToPaddleMHAWithJitToStatic] outputs are miss-matched.")


if __name__ == "__main__":
    unittest.main()
