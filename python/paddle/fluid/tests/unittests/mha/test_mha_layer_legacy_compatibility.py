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
from paddle.static import InputSpec
import paddle.inference as paddle_infer
from utils import compare, generate_weight, generate_data, get_dtype_str, skip_unit_test


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestcuDNNMultiHeadAttentionToLegacy(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        self.nheads = 4
        self.seqlen = 4
        self.embed_dim = 8

        paddle.disable_static()

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.query, self.key, self.value = generate_data(
            batch_size, self.seqlen, self.embed_dim, self.dtype)
        self.weight, _, _, _, _, _, _, _, _, = generate_weight(self.embed_dim,
                                                               np.single)

        self.cudnn_mha = MultiHeadAttention(self.embed_dim, self.nheads)
        self.cudnn_mha.weight.set_value(self.weight)

        self.q_tensor = paddle.to_tensor(
            self.query, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.k_tensor = paddle.to_tensor(
            self.key, dtype=self.dtype, place=self.place, stop_gradient=False)
        self.v_tensor = paddle.to_tensor(
            self.value, dtype=self.dtype, place=self.place, stop_gradient=False)

        self.attn_mask = np.ones(
            (batch_size, self.nheads, self.seqlen, self.seqlen), dtype=np.int32)
        self.attn_mask_tensor = paddle.to_tensor(
            self.attn_mask, dtype=np.int32, place=self.place)

        self.output_ref = self.cudnn_mha(self.q_tensor, self.k_tensor,
                                         self.v_tensor, self.attn_mask_tensor)

        self.pre_legacy_convert_hook()
        self.path = '/tmp/paddle_mha_to_legacy'
        paddle.jit.save(
            self.cudnn_mha,
            self.path,
            input_spec=[
                InputSpec(
                    shape=[None, self.seqlen, self.embed_dim],
                    dtype='float32'), InputSpec(
                        shape=[None, self.seqlen, self.embed_dim],
                        dtype='float32'), InputSpec(
                            shape=[None, self.seqlen, self.embed_dim],
                            dtype='float32'),
                InputSpec(
                    shape=[None, self.nheads, self.seqlen, self.seqlen],
                    dtype='int32')
            ])

    def pre_legacy_convert_hook(self):
        pass

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-6
        self.rtol = 1e-4

    def test_compatibility_with_load_inference_model(self):
        paddle.enable_static()
        exe = paddle.static.Executor(self.place)

        [inference_program, feed_target_names, fetch_targets] = \
            paddle.static.load_inference_model(self.path, exe)

        loaded_output = exe.run(inference_program,
                                feed={
                                    feed_target_names[0]: self.query,
                                    feed_target_names[1]: self.key,
                                    feed_target_names[2]: self.value,
                                    feed_target_names[3]: self.attn_mask
                                },
                                fetch_list=fetch_targets)

        self.assertTrue(
            compare(self.output_ref.numpy(), loaded_output[0], self.atol,
                    self.rtol),
            "[{}] outputs are miss-matched.".format(type(self).__name__))

    def test_to_legacy_with_paddle_inference(self):
        model_file = self.path + ".pdmodel"
        params_file = self.path + ".pdiparams"

        config = paddle_infer.Config(model_file, params_file)
        config.enable_use_gpu(4096, 0)

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        q_handle = predictor.get_input_handle(input_names[0])
        k_handle = predictor.get_input_handle(input_names[1])
        v_handle = predictor.get_input_handle(input_names[2])
        mask_handle = predictor.get_input_handle(input_names[3])

        q_handle.reshape(self.query.shape)
        q_handle.copy_from_cpu(self.query)
        k_handle.reshape(self.key.shape)
        k_handle.copy_from_cpu(self.key)
        v_handle.reshape(self.value.shape)
        v_handle.copy_from_cpu(self.value)
        mask_handle.reshape(self.attn_mask.shape)
        mask_handle.copy_from_cpu(self.attn_mask)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        self.assertTrue(
            compare(self.output_ref.numpy(), output_data, self.atol, self.rtol),
            "[{}] outputs are miss-matched.".format(type(self).__name__))


@unittest.skipIf(skip_unit_test(),
                 "core is not compiled with CUDA or cuDNN version < 8300")
class TestcuDNNMultiHeadAttentionToLegacyWithToStatic(
        TestcuDNNMultiHeadAttentionToLegacy):
    def pre_legacy_convert_hook(self):
        self.path = '/tmp/paddle_mha_to_legacy_with_to_static'
        self.cudnn_mha.forward = paddle.jit.to_static(
            self.cudnn_mha.forward,
            input_spec=[
                InputSpec(
                    shape=[None, self.seqlen, self.embed_dim],
                    dtype='float32'), InputSpec(
                        shape=[None, self.seqlen, self.embed_dim],
                        dtype='float32'), InputSpec(
                            shape=[None, self.seqlen, self.embed_dim],
                            dtype='float32'),
                InputSpec(
                    shape=[None, self.nheads, self.seqlen, self.seqlen],
                    dtype='int32')
            ])


if __name__ == "__main__":
    unittest.main()
