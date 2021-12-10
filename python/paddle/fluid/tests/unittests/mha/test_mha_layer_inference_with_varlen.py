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
from paddle.nn import MultiHeadAttention, cuDNNMultiHeadAttention
from paddle.nn.layer import cuDNNSeqInfoInfer
from paddle.static import InputSpec
import paddle.inference as paddle_infer
from utils import compare, generate_weight


def _generate_data(vocab_size, batch_size, seqlen, nheads):
    # Input for TensorRT must be int32, even Paddle could take int64
    sent_ids = np.random.randint(
        0, vocab_size, (batch_size, seqlen), dtype='int32')
    src_ids = np.zeros((batch_size, seqlen), dtype='int32')
    pos_ids = np.array(
        [np.arange(seqlen) for _ in range(batch_size)], dtype='int32')
    mask = np.ones((batch_size, nheads, seqlen, seqlen), dtype='float32')
    for i in range(batch_size):
        mask[i, :, :, i + 1:] = 0.0

    return sent_ids, src_ids, pos_ids, mask


_NHEADS = 12
_SEQLEN = 4


class cuDNNBERT(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden, heads, max_seqlen):
        super(cuDNNBERT, self).__init__()
        self.word_embeddings = paddle.nn.Embedding(vocab_size, hidden)
        self.position_embeddings = paddle.nn.Embedding(max_seqlen, hidden)
        self.token_type_embeddings = paddle.nn.Embedding(2, hidden)
        self.layer_norm = paddle.nn.LayerNorm(hidden)

        self.seq_info_infer = cuDNNSeqInfoInfer()
        self.cudnn_mha = cuDNNMultiHeadAttention(hidden, heads)

    # Paddle-Inference's TRT converter use unordered_map to map inputs, but 
    # take Input Tensor by index, so name of InputSpec should follow lexicographic order
    @paddle.jit.to_static(input_spec=[
        InputSpec(
            shape=[None, _SEQLEN], dtype='int64', name="input_0"), InputSpec(
                shape=[None, _SEQLEN], dtype='int64', name="input_1"),
        InputSpec(
            shape=[None, _SEQLEN], dtype='int64', name="input_2"), InputSpec(
                shape=[None, _NHEADS, _SEQLEN, _SEQLEN],
                dtype='float32',
                name="input_3")
    ])
    def forward(self, sent_ids, src_ids, pos_ids, mask):
        input_embedings = self.word_embeddings(sent_ids)
        position_embeddings = self.position_embeddings(pos_ids)
        token_type_embeddings = self.token_type_embeddings(src_ids)
        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        seq_data_info = self.seq_info_infer(mask)
        return self.cudnn_mha(embeddings, embeddings, embeddings, seq_data_info)


@unittest.skipIf((not core.is_compiled_with_cuda()) or
                 (not core.is_compiled_with_tensorrt()),
                 "core is not compiled with CUDA or TensorRT")
class TestcuDNNMultiHeadAttentionWithPaddleTRTVarSeqLen(unittest.TestCase):
    def setUp(self):
        self.var_lens = np.array([0, 1, 3, 6, 10], dtype='int32')
        self.batch_size = len(self.var_lens) - 1
        self.nheads = _NHEADS
        self.seqlen = _SEQLEN
        self.embed_dim = 768
        self.vocab_size = 16000

        paddle.disable_static()

        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        sent_ids, src_ids, pos_ids, mask = _generate_data(
            self.vocab_size, self.batch_size, self.seqlen, self.nheads)

        self.cudnn_bert = cuDNNBERT(self.vocab_size, self.embed_dim,
                                    self.nheads, self.seqlen)
        weight, _, _, _, _, _, _, _, _, = generate_weight(self.embed_dim,
                                                          np.single)
        self.cudnn_bert.cudnn_mha.weight.set_value(weight)
        self.cudnn_bert.word_embeddings.weight.set_value(
            np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embed_dim))
            .astype(dtype='float32'))
        self.cudnn_bert.position_embeddings.weight.set_value(
            np.random.uniform(-0.5, 0.5, (self.seqlen, self.embed_dim)).astype(
                dtype='float32'))
        self.cudnn_bert.token_type_embeddings.weight.set_value(
            np.random.uniform(-0.5, 0.5, (2, self.embed_dim)).astype(
                dtype='float32'))

        sent_ids_tensor = paddle.to_tensor(
            sent_ids, dtype='int64', place=self.place)
        src_ids_tensor = paddle.to_tensor(
            src_ids, dtype='int64', place=self.place)
        pos_ids_tensor = paddle.to_tensor(
            pos_ids, dtype='int64', place=self.place)
        mask_tensor = paddle.to_tensor(mask, dtype='float32', place=self.place)

        output = self.cudnn_bert(sent_ids_tensor, src_ids_tensor,
                                 pos_ids_tensor, mask_tensor)
        self.output_ref = np.concatenate(
            [output[i, :i + 1, :].numpy() for i in range(self.batch_size)])
        self.sent_ids = np.concatenate(
            [sent_ids[i, :i + 1] for i in range(self.batch_size)],
            dtype='int32')
        self.src_ids = np.concatenate(
            [src_ids[i, :i + 1] for i in range(self.batch_size)], dtype='int32')

        self.path = '/tmp/paddle_mha_inference_with_varlen'

        layer = cuDNNMultiHeadAttention.to_legacy(
            self.cudnn_bert,
            [sent_ids_tensor, src_ids_tensor, pos_ids_tensor, mask_tensor])
        paddle.jit.save(layer, self.path)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.atol = 1e-1
        self.rtol = 5e-2

    def test_paddle_inference_with_variable_length(self):
        config = paddle_infer.Config(self.path + ".pdmodel",
                                     self.path + ".pdiparams")
        config.enable_use_gpu(4096, 0)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=1,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False)

        min_input_shape = {
            "input_0": [1],
            "input_1": [1],
            "input_2": [1],
            "input_3": [1, 1, 1]
        }
        max_input_shape = {
            "input_0": [self.batch_size * self.seqlen],
            "input_1": [self.batch_size * self.seqlen],
            "input_2": [self.batch_size + 1],
            "input_3": [1, self.seqlen, 1]
        }
        opt_input_shape = {
            "input_0": [self.batch_size * self.seqlen],
            "input_1": [self.batch_size * self.seqlen],
            "input_2": [self.batch_size + 1],
            "input_3": [1, self.seqlen, 1]
        }

        config.set_trt_dynamic_shape_info(
            min_input_shape=min_input_shape,
            max_input_shape=max_input_shape,
            optim_input_shape=opt_input_shape)
        config.enable_tensorrt_oss()

        predictor = paddle_infer.create_predictor(config)

        input_names = predictor.get_input_names()

        sent_ids_handle = predictor.get_input_handle(input_names[0])
        src_ids_handle = predictor.get_input_handle(input_names[1])
        pos_ids_handle = predictor.get_input_handle(input_names[2])
        mask_handle = predictor.get_input_handle(input_names[3])

        sent_ids_handle.reshape([self.var_lens[-1], ])
        sent_ids_handle.copy_from_cpu(self.sent_ids)
        src_ids_handle.reshape([self.var_lens[-1], ])
        src_ids_handle.copy_from_cpu(self.src_ids)
        pos_ids_handle.reshape([len(self.var_lens), ])
        pos_ids_handle.copy_from_cpu(self.var_lens)
        mask_handle.reshape([1, self.seqlen, 1])
        last_input = np.zeros((1, self.seqlen, 1), dtype='float32')
        mask_handle.copy_from_cpu(last_input)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu().reshape(10, 768)

        self.assertTrue(
            compare(self.output_ref, output_data, self.atol, self.rtol),
            "[TestcuDNNMultiHeadAttentionWithPaddleTRTVarSeqLen] output is miss-match."
        )


if __name__ == "__main__":
    unittest.main()
