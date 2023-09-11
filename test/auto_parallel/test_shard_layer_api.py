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
        layer = Layer()
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])

        # Define shard function
        def shard_fn(name, layer, process_mesh):
            if isinstance(layer, nn.Linear):
                for name, param in layer.named_parameters():
                    dist_param = paddle.nn.ParameterList(
                        dist.shard_tensor(param, dist_attr)
                    )
                    layer.add_parameter(name, dist_param)

        sharded_layer = dist.shard_layer(layer, mesh, shard_fn)
        for param in sharded_layer.parameters():
            self.assertIsInstance(param, dist_attr)
            self.assertEqual(param.mesh, mesh)

    def test_shard_layer_input_fn_output_fn(self):
        mesh_input = dist.ProcessMesh([0, 1], dim_names=["x"])
        mesh_output = dist.ProcessMesh([0, 1], dim_names=["y"])
        mesh = dist.ProcessMesh([2, 4], dim_names=["x"])
        layer = Layer()

        def input_fn(inputs, process_mesh):
            dist_attr = dist.DistAttr(mesh=process_mesh, sharding_specs=[None])
            return dist.shard_tensor(inputs, dist_attr)

        def output_fn(outputs, process_mesh):
            dist_attr = dist.DistAttr(mesh=process_mesh, sharding_specs=[None])
            assert isinstance(outputs, paddle.dtensor)
            return dist.shard_tensor(outputs, dist_attr)

        sharded_layer = dist.shard_layer(
            layer,
            mesh,
            input_fn=input_fn(layer, mesh_input),
            output_fn=output_fn(),
        )

        layer_input_fn_output = input_fn(layer, mesh_input)

        layer_output_fn_input = dist.shard_layer(layer_input_fn_output, mesh)

        layer_output_fn_output = output_fn(layer_output_fn_input, mesh_output)

        self.assertEqual(sharded_layer.mesh, layer_output_fn_output)


"""
        # Define input hook
        def input_fn(layer, input):
            input_return = input[0] * 2
            return input_return

        # Define output hook
        def output_fn(layer, input, output):
            return output * 2

        # Verify input_fn
        input_fn_handle = model.register_forward_pre_hook(input_fn)

        value0 = np.arange(26).reshape(2, 13).astype("float32")
        in0 = paddle.to_tensor(value0)
        out0 = model(in0)

        input_fn_handle.remove()

        value1 = value0 * 2
        in1 = paddle.to_tensor(value1)
        out1 = model(in1)

        # hook change the linear's input to input * 2, so out0 is equal to out1.
        assert (out0.numpy() == out1.numpy()).any()

        # Verify output_fn
        output_fn_handle = model.register_forward_post_hook(output_fn)

        value1 = np.arange(26).reshape(2, 13).astype("float32")
        in1 = paddle.to_tensor(value1)

        out0 = model(in1)

        output_fn_handle.remove()

        out1 = model(in1)

        # hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
        assert (out0.numpy() == (out1.numpy()) * 2).any()

        # Shard the model
        model = dist.shard_layer(
            model,
            process_mesh=mesh,
            shard_fn=shard_fn,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        # Verify the parameters
        for name, param in model.named_parameters():
            if param is not None:
                self.assertIsInstance(param, paddle.Tensor)
                self.assertTrue(param.shape == [13, 5] or param.shape == [5])
"""

if __name__ == '__main__':
    unittest.main()
