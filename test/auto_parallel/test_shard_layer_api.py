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

import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn


class MyLayer(nn.Layer):
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(num_features, num_features) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq:
            m.reset_parameters()


class TestShardLayer(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_features = 10
        self.num_layers = 10

    def test_shard_layer_base(self):
        layer = MyLayer(self.num_features, self.num_layers)
        dist_attr = dist.DistAttr(mesh=self.mesh, sharding_specs=[None, None])

        def shard_fn(layer_name, layer):
            if isinstance(layer, nn.Linear):
                for name, param in layer.named_parameters():
                    dist_param = dist.shard_tensor(param, dist_attr=dist_attr)
                    layer.add_parameter(name, dist_param)

        sharded_layer = dist.shard_layer(layer, self.mesh, shard_fn)

        for param in sharded_layer.parameters():
            self.assertTrue(param.is_dist())
            self.assertEqual(param.dist_attr.dims_mapping, [-1, -1])

    def test_shard_layer_input_fn_and_output_fn(self):
        layer = MyLayer(self.num_features, self.num_layers)

        def input_fn(inputs, process_mesh):
            return dist.shard_tensor(
                inputs[0], dist.DistAttr(process_mesh, [None, None])
            )

        def output_fn(outputs, process_mesh):
            assert outputs.is_dist()
            # replace by dist.unshard_dtensor later
            return outputs.numpy()

        replicate_layer = dist.shard_layer(
            layer, self.mesh, input_fn=input_fn, ouput_fn=output_fn
        )

        x = paddle.randn(5, self.num_features)
        local_out_numpy = replicate_layer(x)
        self.assertTrue(isinstance(local_out_numpy, np.ndarray))


if __name__ == '__main__':
    unittest.main()
