# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.nn import Layer


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 1
micro_batch_size = 1
vocab_size = 128
hidden_size = 16


class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear_local = nn.Linear(vocab_size, vocab_size)
        self.linear_shared = nn.Linear(vocab_size, hidden_size)

        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size]
        )
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size], is_bias=False
        )

    def forward(self, x1, x2, y1):
        x_linear_local = self.linear_local(x1)
        x_linear_shared = self.linear_shared(x_linear_local)
        fc = paddle.matmul(x_linear_shared, self.softmax_weight)
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])

        projection = (
            paddle.matmul(projection, self.linear_shared.weight)
            + self.linear_shared.bias
        )

        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=y1, soft_label=True
        )
        return loss.mean()


class SharedLinear(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(vocab_size, hidden_size)

    @property
    def linear_weight(self):
        return self.linear.weight

    @property
    def linear_bias(self):
        return self.linear.bias

    def forward(self, args):
        x1, x2 = args
        x_linear = self.linear(x1)
        return x_linear, x2


class LocalLinear(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(vocab_size, vocab_size)

    def forward(self, args):
        x1, x2 = args
        x_linear = self.linear(x1)
        return x_linear, x2


class MatmulNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size]
        )

    def forward(self, args):
        x1, x2 = args
        fc = paddle.matmul(x1, self.softmax_weight)

        return fc, x2


class BiasNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_bias = self.create_parameter(shape=[vocab_size])

    def forward(self, args):
        fc, x2 = args
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])
        return projection, x2


class LossNet(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, args, y1):
        projection = args
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=y1[0], soft_label=True
        )
        return loss.mean()


class SimpleNetPipe(PipelineLayer):
    def __init__(self, **kwargs):
        self.descs = []
        self.descs.append(LayerDesc(LocalLinear))
        self.descs.append(
            SharedLayerDesc(
                'linear',
                SharedLinear,
                shared_weight_attr=['linear_weight', 'linear_bias'],
            )
        )
        self.descs.append(LayerDesc(MatmulNet))

        self.descs.append(LayerDesc(BiasNet))

        def _logits_helper_0(linear, output):
            return paddle.matmul(output[0], linear.linear_weight)

        def _logits_helper_1(linear, output):
            return output + linear.linear_bias

        self.descs.append(
            SharedLayerDesc(
                'linear',
                SharedLinear,
                forward_func=_logits_helper_0,
                shared_weight_attr=['linear_weight'],
            )
        )
        self.descs.append(
            SharedLayerDesc(
                'linear',
                SharedLinear,
                forward_func=_logits_helper_1,
                shared_weight_attr=['linear_bias'],
            )
        )

        super().__init__(layers=self.descs, loss_fn=LossNet(), **kwargs)


class TestSharedWeightWithMultiAttrsAndSubsetPivotNotAtFirst(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs["pp_configs"].clear_every_step_cache = True

        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        # construct model a
        model_a = SimpleNet()
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a, parameters=model_a.parameters()
        )

        model_b = SimpleNetPipe(topology=hcg.topology())

        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        model_b_params = model_b.parameters()

        if pp_id == 0:
            model_b_params[0].set_value(parameters[4])
            model_b_params[1].set_value(parameters[5])
            model_b_params[2].set_value(parameters[2])
            model_b_params[3].set_value(parameters[3])
            model_b_params[4].set_value(parameters[0])
        else:
            model_b_params[0].set_value(parameters[4])
            model_b_params[1].set_value(parameters[5])
            model_b_params[2].set_value(parameters[1])

        for step in range(5):
            x1_data = np.random.randn(batch_size, vocab_size)
            x2_data = np.random.randn(batch_size, vocab_size)
            y1_data = np.random.randn(batch_size, hidden_size)

            x1 = paddle.to_tensor(x1_data, dtype='float32')
            x2 = paddle.to_tensor(x2_data, dtype='float32')
            y1 = paddle.to_tensor(y1_data, dtype='float32')

            x1.stop_gradient = True
            x2.stop_gradient = True
            y1.stop_gradient = True

            loss_a = model_a(x1, x2, y1)
            loss_a.backward()

            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch(
                [(x1, x2), (y1,)], optimizer_b, scheduler_b
            )

            np.testing.assert_array_equal(
                loss_a.numpy(), loss_b.reshape(shape=[]).numpy()
            )


if __name__ == "__main__":
    unittest.main()
