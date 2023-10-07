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


class Layer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()


class TestShardLayer(unittest.TestCase):
    def test_shard_layer(self):
        # Create a simple layer
        x = paddle.randn([10, 1], 'float32')
        layer = Layer()
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])

        # Define shard function
        def shard_fn(name, layer, process_mesh):
            if isinstance(layer, nn.Linear):
                for name, param in layer._parameters.items():
                    dist_param = dist.shard_tensor(param, dist_attr=dist_attr)
                    layer.add_parameter(name, dist_param)
            else:
                raise ValueError("layer is not a nn.Linear")

        sharded_layer = dist.shard_layer(layer, mesh, shard_fn)

        for param in sharded_layer.parameters():
            self.assertIsInstance(param, dist_attr)
            self.assertEqual(param.mesh, mesh)

    def test_shard_layer_input_fn_output_fn(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        layer = Layer()

        def input_fn(inputs, process_mesh):
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=[None, None]
            )
            for key, param in inputs._parameters.items():
                if param is not None:
                    inputs.add_parameter(
                        key,
                        dist.shard_tensor(param.data, dist_attr=dist_attr),
                    )
            return inputs

        def output_fn(outputs, process_mesh):
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=[None, None]
            )
            for key, param in outputs._parameters.items():
                if param is not None:
                    outputs.add_parameter(
                        key,
                        dist.shard_tensor(param.data, dist_attr=dist_attr),
                    )
            return outputs

        sharded_layer = dist.shard_layer(
            layer,
            mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        # manually complete the process once
        sharded_layer_inpput = input_fn(layer, mesh)
        shard_layer_fn_output = dist.shard_layer(sharded_layer_inpput, mesh)
        sharded_layer_output = output_fn(shard_layer_fn_output, mesh)

        self.assertIsInstance(sharded_layer, nn.Layer)
        self.assertEqual(sharded_layer, sharded_layer_output)


if __name__ == "__main__":
    unittest.main()
