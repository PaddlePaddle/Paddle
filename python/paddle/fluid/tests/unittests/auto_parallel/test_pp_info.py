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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto

from paddle.io import Dataset
from paddle.static import InputSpec
from paddle.distributed.auto_parallel.engine import Engine

paddle.enable_static()

global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3, 4, 5, 6, 7])
PP_MESH_0 = auto.ProcessMesh([6, 7])
PP_MESH_1 = auto.ProcessMesh([4, 5])
PP_MESH_2 = auto.ProcessMesh([2, 3])
PP_MESH_3 = auto.ProcessMesh([0, 1])


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
        self.linear2 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear3 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear4 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = auto.shard_op(self.linear0, dist_attr={"process_mesh":
                                                     PP_MESH_0})(input)[0]
        # auto.shard_tensor(input, dist_attr={"process_mesh": PP_MESH_0, "dims_mapping": [0, -1]})
        out = F.gelu(out, approximate=True)

        out = auto.shard_op(self.linear1, dist_attr={"process_mesh":
                                                     PP_MESH_1})(out)[0]
        # auto.shard_tensor(out, dist_attr={"process_mesh": PP_MESH_1, "dims_mapping": [0, -1]})
        out = F.gelu(out, approximate=True)

        out = auto.shard_op(self.linear2, dist_attr={"process_mesh":
                                                     PP_MESH_2})(out)[0]
        # auto.shard_tensor(out, dist_attr={"process_mesh": PP_MESH_2, "dims_mapping": [0, -1]})
        out = F.gelu(out, approximate=True)

        out = auto.shard_op(self.linear3, dist_attr={"process_mesh":
                                                     PP_MESH_3})(out)[0]
        # auto.shard_tensor(out, dist_attr={"process_mesh": PP_MESH_3, "dims_mapping": [0, -1]})
        out = F.gelu(out, approximate=True)

        out = auto.shard_op(self.dropout,
                            dist_attr={"process_mesh":
                                       global_process_mesh})(out)[0]
        # out = self.dropout(out)
        out = self.linear4(out)
        return out


class TestPPInfo(unittest.TestCase):

    def setUp(self):
        paddle.seed(2022)
        np.random.seed(2022)

    def test_pp_info(self):

        mlp = MLPLayer()
        inputs = InputSpec([4, 1024], 'float32', 'x')
        labels = InputSpec([4], 'int64', 'label')
        engine = Engine(model=mlp,
                        inputs_spec=inputs,
                        labels_spec=labels,
                        strategy=None)

        all_ranks = list(range(8))
        engine._optimizer = None
        engine._gradient_scale = True
        engine._build('predict')
        engine._plan('predict')
        engine._parallel('predict', all_ranks=all_ranks)

        engine.mode = 'predict'
        assert engine.dist_context._pp_info.pp_stages() == 4
        for rank in all_ranks:
            print(engine.dist_context._pp_info.pp_index(rank))

        for rank in all_ranks:
            print(engine.dist_context._pp_info.ups(rank))
            print(engine.dist_context._pp_info.downs(rank))


if __name__ == "__main__":
    unittest.main()
