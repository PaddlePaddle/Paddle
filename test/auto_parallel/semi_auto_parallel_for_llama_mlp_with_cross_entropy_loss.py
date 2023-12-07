# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard

BATCH_NUM = 4
BATCH_SIZE = 16
HIDDEN_SIZE = 1024
SEQ_LEN = 128
VOCAB_SIZE = 32000


class LlamaLMHead(nn.Layer):
    def __init__(self, is_tensor_parallel=False, mesh=None):
        super().__init__()

        self.weight = paddle.create_parameter(
            shape=[HIDDEN_SIZE, VOCAB_SIZE],
            dtype=paddle.get_default_dtype(),
        )
        if is_tensor_parallel:
            self.weight = dist.shard_tensor(self.weight, mesh, [dist.Shard(0)])

    def forward(self, hidden_states):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


class LlamaCriterion(nn.Layer):
    def __init__(self):
        super().__init__()
        self.ignore_index = -100
        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(self, prediction_scores, masked_lm_labels):
        # TODO(liyurui): Unsqueeze inferspmd is not implemented yet. Uncomment this after we support.
        # masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
        masked_lm_loss = self.loss_func(
            prediction_scores.astype("float32"), masked_lm_labels
        )
        masked_lm_loss = paddle.masked_select(
            masked_lm_loss, masked_lm_loss > 0
        ).astype("float32")
        loss = paddle.mean(masked_lm_loss)
        return loss


class TestLlamaCriterionForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.loss = paddle.nn.loss.CrossEntropyLoss()
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def init_input_data(self):
        input = np.random.random([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]).astype(
            self._dtype
        )
        label = np.random.randint(0, SEQ_LEN, [BATCH_SIZE, SEQ_LEN, 1])
        input = paddle.to_tensor(input)
        label = paddle.to_tensor(label, dtype='int64')
        return input, label

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        self.base_out, self.base_parameters = self.train_loop(LlamaLMHead())

    def train_loop(self, layer, shard_input=False):
        # run forward and backward
        input_dist_attr = [Shard(0)]

        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        for _ in range(5):
            input, label = self.init_input_data()
            if shard_input:
                input = dist.shard_tensor(input, self._mesh, input_dist_attr)
                label = dist.shard_tensor(label, self._mesh, input_dist_attr)
            out = layer(input)
            loss = self.loss(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return loss, layer.parameters()

    def check_tensor_eq(self, a, b, rtol=1e-04, atol=1e-05, verbose=True):
        if a is None:
            assert b is None
            return
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=rtol, atol=atol, verbose=verbose
        )

    def test_dp(self):
        self.set_random_seed(self._seed)

        dp_layer = LlamaLMHead()

        dp_out, dp_parameters = self.train_loop(
            dp_layer,
            shard_input=True,
        )
        self.check_tensor_eq(dp_out, self.base_out)
        for param, param_base in zip(dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp(self):
        self.set_random_seed(self._seed)

        mp_layer = LlamaLMHead(True, self._mesh)

        mp_out, mp_parameters = self.train_loop(mp_layer)
        self.check_tensor_eq(mp_out, self.base_out)
        for param, param_base in zip(mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp()
        self.test_mp()


if __name__ == '__main__':
    TestLlamaCriterionForSemiAutoParallel().run_test_case()
