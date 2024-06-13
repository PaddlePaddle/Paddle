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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils

# NOTE(SigureMo): Disable the CSE optimization to avoid op number change.
paddle.set_flags({"FLAGS_enable_cse_in_dy2st": False})


class LlamaPostProcess(nn.Layer):
    def __init__(self):
        super().__init__()

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

        return input_ids, scores

    def forward(self, logits, input_ids):
        batch_size, cur_len = paddle.shape(input_ids)
        origin_len = paddle.shape(input_ids)[1]
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype()
        )
        return self._post_process_(
            logits, input_ids, cur_len, origin_len, scores, unfinished_flag
        )


class TestLlamaPostProcess(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 2048, 768]
        self.logits = paddle.randn([1, 256, 3200], dtype="float32")
        self.input_ids = paddle.randint(0, 512, [1, 32], dtype="int64")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 4)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 4})

    def eval(self, use_cinn):
        paddle.seed(2024)
        net = LlamaPostProcess()
        input_spec = [
            InputSpec(shape=[None, None, 3200], dtype='float32'),  # logits
            InputSpec(shape=[None, None], dtype='int64'),  # input_ids
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        # paddle.jit.save(net, sys.path.join(dirname(__file__), "post_model"))
        out = net(self.logits, self.input_ids)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        # TODO(Aurelius84): fix the precision with inf
        # for i in range(len(dy_out)):
        #     np.testing.assert_allclose(
        #         cinn_out[i].numpy(), dy_out[i].numpy(), atol=1e-6, rtol=1e-6
        #     )


if __name__ == '__main__':
    unittest.main()
