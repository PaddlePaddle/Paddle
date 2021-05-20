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

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
import unittest
from hybrid_parallel_pp_layer import AlexNetPipeDesc, AlexNetPipe, ReshapeHelp
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn as nn
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.dygraph.container import Sequential
import paddle.nn.functional as F


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super(ReshapeHelp, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


class AlexNetPipeDesc(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(
                nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                ReshapeHelp, shape=[-1, 256]),
            LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super(AlexNetPipeDesc, self).__init__(
            layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


class AlexNet(Layer):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features_1 = Sequential(
            nn.Conv2D(
                1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            nn.Conv2D(
                64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            nn.Conv2D(
                192, 384, kernel_size=3, padding=1))

        self.features_2 = Sequential(
            nn.ReLU(),
            nn.Conv2D(
                384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2), )

        self.reshape_layer = ReshapeHelp(shape=[-1, 256])
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.loss.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features_1(x)
        # print(x.shape)
        # print("x", x.numpy())
        x = self.features_2(x)
        x = self.reshape_layer(x)
        x = self.classifier(x)
        print(x.numpy())
        return self.loss_fn(x, y)


batch_size = 4
micro_batch_size = 2


class TestDistPPTraning(unittest.TestCase):
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
            "micro_batch_size": micro_batch_size
        }
        paddle.distributed.init_parallel_env()
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        #construct model a
        model_a = AlexNet(10)
        optimizer_a = paddle.optimizer.SGD(learning_rate=0.001,
                                           parameters=model_a.parameters())

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        # construct model b
        model_b = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        optimizer_b = paddle.optimizer.SGD(learning_rate=0.001,
                                           parameters=model_b.parameters())
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        for idx, param in enumerate(model_b.parameters()):
            param.set_value(parameters[idx + pp_id * (param_len // 2)])

        # construct reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size, drop_last=True)

        for step_id, data in enumerate(train_reader()):
            x_data = np.array([x[0] for x in data]).astype('float32').reshape(
                batch_size, 1, 28, 28)
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                batch_size, 1)
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            img.stop_gradient = True
            label.stop_gradient = True

            if step_id >= 1:
                return

            # print("shape", img.shape, label.shape)
            # tmp = img[:2, :]
            # print("tmp shape", tmp.shape)
            # loss_a = model_a(img[:2, :], label[:2, :])

            loss_a = model_a(img, label)
            # loss_a_2 = model_a(img[2:4, :], label[2:4, :])
            # loss_a_2 = model_a(img[2:4, :], label[2:4, :])
            # loss_a_2 = model_a(img[2:4, :], label[2:4, :])

            # loss_a.backward()
            # optimizer_a.step()
            # optimizer_a.clear_grad()
            # print("loss_a", loss_a)

            # loss_b = model_b.train_batch([img, label], optimizer_b)
            # print("loss_a loss_b", loss_a.numpy(), loss_b.numpy())
        return True


if __name__ == "__main__":
    unittest.main()
