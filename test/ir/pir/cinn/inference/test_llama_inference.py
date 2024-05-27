# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from os.path import dirname

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils
from llama_test_model import LlamaConfig, LlamaModel


class LlamaInference(nn.Layer):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig()
        self._forward_ = LlamaModel(self.config)

    def update_scores_for_generation(
        self, scores, next_scores, length, unfinished_flag
    ):
        # update scores
        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def _post_process_(
        self, logits, input_ids, cur_len, origin_len, scores, unfinished_flag
    ):
        # [batch_size, vocab_size]
        logits = logits[:, -1, :]
        probs = F.softmax(logits)

        temperature = paddle.full([1], 1)
        top_p = paddle.full([1], 0)

        # sample
        origin_probs = F.log_softmax(logits)
        # compute next_tokens
        logits = logits / temperature
        top_ps_tensor = paddle.full(
            shape=[paddle.shape(probs)[0], 1],
            fill_value=top_p,
            dtype=probs.dtype,
        )
        _, next_tokens = paddle.tensor.top_p_sampling(probs, top_ps_tensor)

        next_scores = paddle.index_sample(origin_probs, next_tokens)
        scores = self.update_scores_for_generation(
            scores, next_scores, cur_len - origin_len, unfinished_flag
        )

        input_ids = paddle.concat([input_ids, next_tokens], axis=1)

        return input_ids, scores, unfinished_flag

    def forward(self, input_ids, position_ids, attention_mask, use_cache=None):
        batch_size, cur_len = paddle.shape(input_ids)

        batch_size, cur_len = paddle.shape(input_ids)
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = paddle.shape(input_ids)[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")

        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype()
        )

        max_new_tokens = paddle.full([1], 16, dtype="int64")

        outputs = self._forward_(
            input_ids, position_ids, attention_mask, use_cache
        )
        input_ids, scores, unfinished_flag = self._post_process_(
            outputs,
            input_ids,
            cur_len_gpu,
            origin_len_gpu,
            scores,
            unfinished_flag,
        )
        paddle.increment(cur_len)
        paddle.increment(cur_len_gpu)

        while cur_len < max_new_tokens and paddle.any(unfinished_flag):
            (
                input_ids,
                scores,
                unfinished_flag,
                model_kwargs,
            ) = self._post_process_(
                self._forward_(
                    input_ids, position_ids, attention_mask, use_cache
                ),
                input_ids,
                cur_len_gpu,
                origin_len_gpu,
                scores,
                unfinished_flag,
            )
            paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

        return input_ids[:, origin_len:]


class TestLlamaInference(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.config = LlamaConfig()
        self.input_ids = paddle.to_tensor(
            [
                [
                    1,
                    29871,
                    31201,
                    236,
                    138,
                    141,
                    30287,
                    30557,
                    30015,
                    233,
                    187,
                    172,
                    31969,
                    31325,
                    31043,
                    30374,
                    30024,
                ]
            ],
            dtype="int64",
        )
        self.position_ids = paddle.to_tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype="int64",
        )
        self.attention_mask = paddle.to_tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="int64"
        )

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2024)
        net = LlamaModel(self.config)
        input_spec = [
            InputSpec(shape=[None, None], dtype='int64'),  # input_ids
            InputSpec(shape=[None, None], dtype='int64'),  # position_ids
            InputSpec(shape=[None, None], dtype='int64'),  # attention_mask
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.input_ids, self.position_ids, self.attention_mask)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        # TODO(Aurelius84): deny embedding and softmax in prim
        paddle.set_flags(
            {
                "FLAGS_prim_forward_blacklist": "pd_op.embedding;pd_op.softmax",
            }
        )
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
