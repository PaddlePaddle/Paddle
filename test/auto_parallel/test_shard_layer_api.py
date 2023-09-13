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

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.base import framework


class Layer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)

    def forward(self, input):
        temp = self._linear(input)
        return temp


class TestShardLayer(unittest.TestCase):
    def test_shard_layer(self):
        # Create a simple layer
        x = paddle.randn([10, 1], 'float32')
        layer = Layer()
        # layer = mylayer(x)
        # mesh = dist.ProcessMesh([0,1], dim_names=["x"])
        # dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        print(type(x), type(layer), type(dist_attr))
        print(layer)
        print(dist_attr)

        # Define shard function
        def shard_fn(name, layer, process_mesh):
            if isinstance(layer, nn.Linear):
                for name, param in layer._parameters.items():
                    # layer里遍历出来两个tensor，tensor的shape还不一样？？？？？？？？？？？？？？
                    print("1.param.data: ", type(param.data))
                    print("2.param: ", param)
                    print("3.dist_attr: ", dist_attr)
                    print(
                        "4.len(dist_attr.dims_mapping)= ",
                        len(dist_attr.dims_mapping),
                    )
                    print("5.param.data.shape= ", param.data.shape)
                    dist_param = dist.shard_tensor(
                        param.data, dist_attr=dist_attr
                    )
                    result = framework.EagerParamBase(
                        dist_param.shape,
                        dist_param.dtype,
                        stop_gradient=dist_param.stop_gradient,
                    )
                    result.set_value(dist_param)

        sharded_layer = dist.shard_layer(layer, mesh, shard_fn)

    # for param in sharded_layer.parameters():
    #    self.assertIsInstance(param, dist_attr)
    #    self.assertEqual(param.mesh, mesh)

    def test_shard_layer_input_fn_output_fn(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        layer = Layer()

        def input_fn(inputs, process_mesh):
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=[None, None]
            )
            return dist.shard_tensor(inputs, dist_attr=dist_attr)

        def output_fn(outputs, process_mesh):
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=[None, None]
            )
            assert isinstance(outputs, paddle.dtensor)
            return dist.shard_tensor(outputs, dist_attr=dist_attr)

        sharded_layer = dist.shard_layer(
            layer,
            mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )


if __name__ == '__main__':
    unittest.main()
