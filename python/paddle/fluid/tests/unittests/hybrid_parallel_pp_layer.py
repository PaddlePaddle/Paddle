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

import unittest
import numpy as np
from paddle.distributed import fleet
from paddle.fluid.dygraph.container import Sequential
import paddle.nn as nn
from paddle.fluid.dygraph.layers import Layer
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn.functional as F


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super(ReshapeHelp, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


class AlexNet(Layer):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            nn.Conv2D(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
        )

        self.reshape_layer = ReshapeHelp(shape=[-1, 256])
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.loss.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = self.reshape_layer(x)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(AlexNet):
    def to_layers(self):
        feat = [self.features[i] for i in range(len(self.features))]
        loss_fn = [self.reshape_layer, self.classifier]
        feat.extend(loss_fn)
        return feat


class AlexNetPipeDesc(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(ReshapeHelp, shape=[-1, 256]),
            LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super(AlexNetPipeDesc, self).__init__(
            layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs
        )


class TestPipeLayerAPI(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        pipe_model = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        np.testing.assert_array_equal(len(pipe_model.parameters()), 6)

    def test_pipelayer_sequential(self):
        init_net = AlexNetPipe()
        pipe_model = PipelineLayer(
            layers=init_net.to_layers(),
            num_stages=self.pipeline_parallel_size,
            loss_fn=nn.CrossEntropyLoss(),
        )
        stage_id = self.hcg.get_stage_id()
        init_parameters = init_net.parameters()
        pipe_parameters = pipe_model.parameters()
        part_number = len(init_parameters) // 2

        if stage_id == 0:
            for idx in range(part_number):
                param_a = init_parameters[idx]
                param_b = pipe_parameters[idx]
                np.testing.assert_array_equal(param_a.name, param_b.name)
                np.testing.assert_allclose(param_a.numpy(), param_b.numpy())

        elif stage_id == 1:
            for idx in range(part_number):
                param_a = init_parameters[idx + part_number]
                param_b = pipe_parameters[idx]

                np.testing.assert_array_equal(param_a.name, param_b.name)
                np.testing.assert_allclose(param_a.numpy(), param_b.numpy())


if __name__ == '__main__':
    unittest.main()
