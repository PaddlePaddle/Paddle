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


class LlamaWhile(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, logits, input_ids):
        batch_size, cur_len = paddle.shape(input_ids)
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="float32")
        max_new_tokens = paddle.full([1], 16, dtype="int64")
        while cur_len < max_new_tokens and paddle.any(unfinished_flag):
            # [batch_size, vocab_size]
            probs = F.softmax(logits[:, -1, :])

            # compute next_tokens
            top_ps_tensor = paddle.full(
                shape=[paddle.shape(probs)[0], 1],
                fill_value=0,
                dtype=probs.dtype,
            )
            _, next_tokens = paddle.tensor.top_p_sampling(probs, top_ps_tensor)
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            cur_len += 1

        return input_ids


class TestLlamaPostProcess(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.logits = paddle.randn([1, 256, 3200], dtype="float32")
        self.input_ids = paddle.randint(0, 512, [1, 8], dtype="int64")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2024)
        net = LlamaWhile()
        input_spec = [
            InputSpec(shape=[None, None, 3200], dtype='float32'),  # logits
            InputSpec(shape=[None, None], dtype='int64'),  # input_ids
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.logits, self.input_ids)
        return out

    @unittest.skip("TODO: xiongkun")
    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
