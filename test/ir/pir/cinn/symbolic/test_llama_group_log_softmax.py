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
from paddle.base import core
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
sys.path.append("../")
import utils


def update_scores_for_generation(
    scores, next_scores, length, unfinished_flag=None
):
    # update scores

    unfinished_scores = (scores * length.astype(scores.dtype) + next_scores) / (
        length + 1
    ).astype(scores.dtype)
    return unfinished_scores


def tmp(logits, scores, next_tokens, length):
    origin_probs = F.log_softmax(logits)  # [-1,32000], f16

    # compute next_tokens
    # logits = logits / temperature
    # top_ps_tensor = paddle.full(shape=[paddle.shape(probs)[0], 1], fill_value=top_p, dtype=probs.dtype)
    # _, next_tokens = paddle.tensor.top_p_sampling(probs, top_ps_tensor)

    next_scores = paddle.index_sample(
        origin_probs, next_tokens
    )  # (builtin.tensor<-1x32000xf16>, builtin.tensor<-1x1xi64>) -> builtin.tensor<-1x1xf16>
    scores = update_scores_for_generation(scores, next_scores, length)
    return scores


class TestGroupOpNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, scores, next_tokens, length):
        # "O" represents COPY semantics.
        out = tmp(x, scores, next_tokens, length)
        return out


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape1 = [1, 32000]
        self.x = paddle.randn(self.shape1, dtype="float16")
        self.x.stop_gradient = False
        self.score_s = [1, 1]
        self.score = paddle.randn(self.score_s, dtype="float16")
        self.score.stop_gradient = False

        self.shape2 = [1, 1]
        self.y = paddle.full(self.shape2, 1, dtype="int64")
        self.y.stop_gradient = False
        self.shape3 = [1]
        self.z = paddle.full(self.shape3, 1, dtype="int64")
        self.z.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn=False, mode="jit"):
        net = TestGroupOpNet()
        if mode == "eager":
            out = net(self.x, self.score, self.y, self.z)
        else:
            input_spec = [
                InputSpec(shape=[None, 32000], dtype="float16"),
                InputSpec(shape=[None, 1], dtype="float16"),
                InputSpec(shape=[None, 1], dtype="int64"),
                InputSpec(shape=[1], dtype="int64"),
            ]
            net = utils.apply_to_static(net, use_cinn, input_spec)
            net.eval()
            out = net(self.x, self.score, self.y, self.z)
            if use_cinn:
                self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(mode="eager")
        core._set_prim_all_enabled(True)
        # cinn_out = self.eval(use_cinn=utils.unittest_use_cinn())
        cinn_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )
        core._set_prim_all_enabled(True)


if __name__ == '__main__':
    unittest.main()
