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

import unittest
import numpy as np
import os
import paddle
from paddle.distributed import fleet
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


class FakeAlexNetPipeDesc(PipelineLayer):

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.Conv2D, 64, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            LayerDesc(nn.Conv2D, 192, 192, kernel_size=5, padding=2),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            LayerDesc(nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(ReshapeHelp, shape=[-1, 256]),
            LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super(FakeAlexNetPipeDesc, self).__init__(layers=decs,
                                                  loss_fn=nn.CrossEntropyLoss(),
                                                  **kwargs)


class TestPipeLayerAPI(unittest.TestCase):

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        pipe_model = FakeAlexNetPipeDesc(seg_method="layer:Conv2D",
                                         num_stages=self.pipeline_parallel_size,
                                         num_virtual_pipeline_stages=2)
        assert len(pipe_model.parameters()) > 0
        dist_model = fleet.distributed_model(pipe_model)


if __name__ == '__main__':
    unittest.main()
