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


class MLPForVirtualStageLayerTest(PipelineLayer):

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(nn.Linear, 2, self.num_classes),
            LayerDesc(nn.Linear, self.num_classes, 2),
            LayerDesc(nn.Linear, 2, self.num_classes),
            LayerDesc(nn.Linear, self.num_classes, 2),
            LayerDesc(nn.Linear, 2, self.num_classes),
            LayerDesc(nn.Linear, self.num_classes, 2),
            LayerDesc(nn.Linear, 2, self.num_classes),
            LayerDesc(nn.Linear, self.num_classes, 2),
        ]
        super(MLPForVirtualStageLayerTest,
              self).__init__(layers=decs,
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
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        pipe_model = MLPForVirtualStageLayerTest(
            seg_method="layer:Linear",
            num_stages=self.pipeline_parallel_size,
            num_virtual_pipeline_stages=2,
            recompute_interval=1,
            recompute_ctx={
                "mp_group": self.hcg.get_model_parallel_group(),
                "offload": False,
                "partition": False
            })
        assert len(pipe_model.parameters()) > 0
        model_chunks = pipe_model.get_model_chunks()
        assert model_chunks is not None
        assert len(model_chunks) == 2

        optimizer = paddle.optimizer.SGD(parameters=pipe_model.parameters())

        try:
            model_chunks[0](paddle.to_tensor([1., 2.]))
        except NotImplementedError:
            pass

        # fake call for the forward function of virtual pipeline layer
        for i in range(len(model_chunks)):
            out = pipe_model(paddle.to_tensor([1., 2.]), chunk_id=i)
            assert list(out.shape) == [2]
            out = F.relu(out)
            loss = paddle.mean(out)
            loss.backward()

        optimizer.step()

        # just make sure the model can be wrapped with distributed model
        dist_model = fleet.distributed_model(pipe_model)


if __name__ == '__main__':
    unittest.main()
