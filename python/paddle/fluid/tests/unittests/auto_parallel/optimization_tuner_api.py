# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.distributed.fleet import auto
=======
import unittest
import time
import tempfile
import copy
import os
import numpy as np
import subprocess
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
from paddle.fluid import layers
from paddle.io import Dataset, IterableDataset, DataLoader
from paddle.static import InputSpec
from paddle.distributed import fleet
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.engine import Engine
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
from engine_api_dp import MyDataset

paddle.enable_static()
batch_size = 16
batch_num = 5
hidden_size = 1024
sequence_len = 512
image_size = hidden_size
class_num = 10

paddle.seed(44)

<<<<<<< HEAD

class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
=======
# class MyDataset(Dataset):

#     def __init__(self, num_samples):
#         super(MyDataset, self).__init__()
#         self.num_samples = num_samples

#     def __getitem__(self, index):
#         input = np.random.uniform(size=image_size).astype("float32")
#         label = np.random.randint(0, class_num - 1, dtype="int64")
#         return input, label

#     def __len__(self):
#         return self.num_samples


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=bias_attr)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        self.out = out
        return out


def train(fetch):
<<<<<<< HEAD
    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02,
    )
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.fluid.optimizer.AdamOptimizer(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None,
    )

    dist_strategy = auto.Strategy()
    dist_strategy.auto_mode = "semi"
    # sharding config
    sharding = dist_strategy.sharding
    sharding.enable = True
    sharding.degree = 2
    sharding.stage = 3
    sharding.enable_tuning = True
    sharding.tuning_range = [0, 1, 2, 3]
    # Tuning configuration
    tuning = dist_strategy.tuning
    tuning.enable = True
    tuning.profile_start_step = 1
    tuning.profile_end_step = 5
    tuning.run_after_tuning = True
    tuning.verbose = True

    dataset = MyDataset(batch_num * batch_size)
    engine = auto.Engine(
        mlp, loss, optimizer, paddle.metric.Accuracy(), strategy=dist_strategy
    )
    engine._tune(dataset, batch_size=batch_size)

    # check tuned
    assert engine._dist_contexts['train'].strategy.sharding.stage != 3
=======
    mlp = MLPLayer(hidden_size=hidden_size,
                   intermediate_size=4 * hidden_size,
                   dropout_ratio=0.1,
                   initializer_range=0.02)
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.00001,
                                                     beta1=0.9,
                                                     beta2=0.999,
                                                     epsilon=1e-08,
                                                     grad_clip=None)

    inputs_spec = InputSpec([batch_size, hidden_size], 'float32', 'x')
    labels_spec = InputSpec([batch_size], 'int64', 'label')

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.amp = False
    dist_strategy.pipeline = False
    dist_strategy.recompute = False
    # init parallel optimizer
    dist_strategy.semi_auto = True
    dist_strategy.sharding = True
    dist_strategy.sharding_configs = {
        "sharding_degree": 2,
        "stage": 3,
        "enable_tuning": True,
    }
    fleet.init(is_collective=True, strategy=dist_strategy)

    # init engine
    import tempfile
    tmp_dir = tempfile.TemporaryDirectory()
    dataset = MyDataset(batch_num * batch_size)

    # Tuning configuration
    tuning_config = {
        "batch_size": batch_size,
        "dataset": dataset,
        "profile_start_step": 1,
        "profile_end_step": 5,
        "run_after_tuning": True,
        "sharding": {
            "stage_range": [0, 1, 2, 3]
        },
        "verbose": True,
    }
    engine = Engine(mlp,
                    inputs_spec=inputs_spec,
                    labels_spec=labels_spec,
                    strategy=dist_strategy,
                    user_tuning_config=tuning_config)
    engine.prepare(optimizer, loss, metrics=paddle.metric.Accuracy())

    # check tuned
    assert (engine._dist_contexts['train'].strategy.sharding_configs['stage'] !=
            3)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


if __name__ == "__main__":
    train(True)
