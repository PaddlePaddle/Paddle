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

import random
import unittest

import numpy as np

import paddle
from paddle.autograd import PyLayer
from paddle.distributed import fleet

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 10
batch_size = 4


class Concat(PyLayer):
    @staticmethod
    def forward(ctx, inp, axis, group):
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=axis)
        ctx.args_axis = axis
        ctx.args_group = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        axis = ctx.args_axis
        group = ctx.args_group
        with paddle.no_grad():
            grads = paddle.split(
                grad, paddle.distributed.get_world_size(group), axis=axis
            )
        grad = grads[paddle.distributed.get_rank(group)]
        return grad


class Split(PyLayer):
    @staticmethod
    def forward(ctx, inp, axis, group):
        with paddle.no_grad():
            inps = paddle.split(
                inp, paddle.distributed.get_world_size(group), axis=axis
            )
        inp = inps[paddle.distributed.get_rank(group)]

        ctx.args_axis = axis
        ctx.args_group = group
        return inp

    @staticmethod
    def backward(ctx, grad):
        axis = ctx.args_axis
        group = ctx.args_group
        grads = []
        paddle.distributed.all_gather(grads, grad, group=group)
        with paddle.no_grad():
            grad = paddle.concat(grads, axis=axis)
        return grad


class SimpleNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class SEPModel(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self._net = SimpleNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        self._hcg = fleet.get_hybrid_communicate_group()

    def forward(self, x):
        x = Split.apply(x, axis=1, group=self._hcg.get_sep_parallel_group())
        x = self._net.forward(x)
        x = Concat.apply(x, axis=1, group=self._hcg.get_sep_parallel_group())
        loss = x.mean()
        return loss


class DPModel(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self._net = SimpleNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )

    def forward(self, x):
        x = self._net.forward(x)
        loss = x.mean()
        return loss


class TestDistSEPTraining(unittest.TestCase):
    def setUp(self):
        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 1,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sep_degree": 2,
        }
        fleet.init(is_collective=True, strategy=self.strategy)

    def test_basic_hcg(self):
        hcg = fleet.get_hybrid_communicate_group()
        assert hcg.get_sep_parallel_rank() >= 0
        assert hcg.get_sep_parallel_world_size() == 2
        assert hcg.get_sep_parallel_group_src_rank() == 0
        assert hcg.get_sep_parallel_group() is not None
        assert hcg.get_dp_sep_parallel_group() is not None
        assert hcg.get_pp_mp_parallel_group() is not None

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )
        return optimizer

    def build_model(self, model_cls):
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = model_cls(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )

        return model

    def test_sep_train(self):
        sep_model = self.build_model(SEPModel)
        sep_model = fleet.distributed_model(sep_model)
        sep_optimizer = self.build_optimizer(sep_model)
        sep_optimizer = fleet.distributed_optimizer(sep_optimizer)
        dp_model = self.build_model(DPModel)
        dp_optimizer = self.build_optimizer(dp_model)

        for _ in range(5):
            np_data = np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            batch = paddle.to_tensor(np_data)
            loss_sep = self.train_batch(batch, sep_model, sep_optimizer)
            loss_dp = self.train_batch(batch, dp_model, dp_optimizer)

            np.testing.assert_allclose(
                loss_sep.numpy(), loss_dp.numpy(), rtol=1e-3
            )


if __name__ == "__main__":
    unittest.main()
