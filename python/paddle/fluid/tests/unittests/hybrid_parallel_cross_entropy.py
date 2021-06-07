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

from __future__ import division
from __future__ import print_function

import unittest

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle import framework
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)


batch_size = 1
sequence_length = 1
class_num = 2


class CrossEntropyNet(fluid.dygraph.Layer):
    def __init__(self):
        super(CrossEntropyNet, self).__init__()
        # self.parallel_linear = fleet.meta_parallel.VocabParallelCrossEntropy(
        #     in_features=input_size,
        #     out_features=output_size,
        #     weight_attr=None,
        #     has_bias=True,
        #     gather_output=True,
        #     name="test_column_linear")

        #self.loss_layer = paddle.nn.loss.CrossEntropyLoss()
        self.loss_layer = fleet.meta_parallel.ParallelCrossEntropy()

    def forward(self, x, label):
        output = self.loss_layer(x, label)
        return output


strategy = fleet.DistributedStrategy()
model_parallel_size = 2
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": model_parallel_size,
    "pp_degree": 1
}
fleet.init(is_collective=True, strategy=strategy)
# set_random_seed(99)

hcg = fleet.get_hybrid_communicate_group()
word_size = hcg.get_model_parallel_world_size()
mp_id = hcg.get_model_parallel_rank()
dp_id = hcg.get_data_parallel_rank()
rank_id = dist.get_rank()

# tracker = get_rng_state_tracker()
# tracker.add('global_seed', 1024)
# tracker.add('local_seed', rank_id)
# print("rank_id ", rank_id)

paddle.seed(rank_id * 10)
random.seed(1024)
np.random.seed(1024)

model = CrossEntropyNet()
for _ in range(3):
    label_data = np.random.randint(
        0, 2 * class_num, size=[batch_size, sequence_length])
    label = paddle.to_tensor(label_data)
    data = paddle.randn(
        shape=[batch_size, sequence_length, class_num], dtype='float32')

    data.stop_gradient = False
    print("data ", data)
    print("label ", label)
    loss = model(data, label)
    loss.backward()

    print("grad: ", data.grad)
    print("loss: ", loss)
