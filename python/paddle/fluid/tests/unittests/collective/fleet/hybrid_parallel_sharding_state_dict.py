# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10


class SimpleDPNet(paddle.nn.Layer):
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


class TestDistMPTraning(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

    def build_optimizer(self, model, strategy=None, Optimizer="adam"):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        if Optimizer == "adam":
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=0.001,
                weight_decay=0.00001,
                grad_clip=clip,
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                parameters=model.parameters(),
                grad_clip=clip,
            )
        return optimizer

    def test_set_state_dict(self):
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )

        local_optimizer = self.build_optimizer(
            model_a,
            strategy=self.strategy,
            Optimizer="adam",
        )
        dist_optimizer = fleet.distributed_optimizer(local_optimizer)
        # prepare state_dict
        state_dict = {}
        state_dict["master_weights"] = {}
        all_param_names = []
        for p in model_a.parameters():
            var_name = dist_optimizer._gen_master_weight_var_name(p)
            var = paddle.static.create_global_var(
                name=var_name,
                shape=p.shape,
                value=0,
                dtype='float32',
                persistable=True,
            )
            var = paddle.randn(shape=var.shape, dtype=var.dtype, name=var.name)
            state_dict["master_weights"][p.name] = var
            all_param_names.append(p.name)
        # test api
        tmp_state_dict = copy.deepcopy(state_dict)
        dist_optimizer.set_state_dict(state_dict)
        # check result
        local_params = dist_optimizer._rank2params[
            dist_optimizer._sharding_rank
        ]
        local_param_names = [p.name for p in local_params]
        other_param_names = [
            p_name
            for p_name in all_param_names
            if p_name not in local_param_names
        ]
        inner_opt = dist_optimizer._inner_opt
        assert hasattr(inner_opt, "_master_weights")
        for p_name, weight in inner_opt._master_weights.items():
            assert p_name in local_param_names
            assert p_name not in other_param_names
            assert p_name in tmp_state_dict["master_weights"]
            np.testing.assert_array_almost_equal(
                weight.numpy(), tmp_state_dict["master_weights"][p_name].numpy()
            )


if __name__ == "__main__":
    unittest.main()
