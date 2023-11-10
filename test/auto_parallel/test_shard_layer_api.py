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


# TODO(chenweihang): test for paddle nn Layer API
class DemoLayer(nn.Layer):
    def __init__(self, num_features):
        super().__init__()
        self.w0 = self.create_parameter(shape=[num_features, num_features])
        self.w1 = self.create_parameter(shape=[num_features, num_features])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class MyLayer(nn.Layer):
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.seq = nn.Sequential(
            *[DemoLayer(num_features) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.seq(x)


class TestShardLayer(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_features = 10
        self.num_layers = 10

    def test_shard_layer_base(self):
        layer = MyLayer(self.num_features, self.num_layers)

        def shard_fn(layer_name, layer, process_mesh):
            if isinstance(layer, nn.Linear):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        dist_param = dist.shard_tensor(
                            param,
                            dist_attr=dist.DistAttr(
                                mesh=process_mesh, sharding_specs=[None, None]
                            ),
                        )
                    else:
                        dist_param = dist.shard_tensor(
                            param,
                            dist_attr=dist.DistAttr(
                                mesh=process_mesh, sharding_specs=[None]
                            ),
                        )
                    layer.add_parameter(name, dist_param)

        # test shard parameters
        sharded_params_layer = dist.shard_layer(layer, self.mesh, shard_fn)

        for param in sharded_params_layer.parameters():
            self.assertTrue(param.is_dist())
            for x in param.dist_attr.dims_mapping:
                self.assertEqual(x, -1)

        # test shard buffers
        test_buffer = paddle.randn([10])
        layer.register_buffer("test_buffer", test_buffer, persistable=True)
        sharded_buffers_layer = dist.shard_layer(layer, self.mesh, shard_fn)
        self.assertTrue(sharded_buffers_layer.test_buffer.is_dist())
        self.assertEqual(
            sharded_buffers_layer.test_buffer.dist_attr.dims_mapping, [-1]
        )

    def test_shard_layer_input_fn_and_output_fn(self):
        layer = MyLayer(self.num_features, self.num_layers)

        def input_fn(inputs, process_mesh):
            return dist.shard_tensor(
                inputs[0], dist_attr=dist.DistAttr(process_mesh, [None, None])
            )

        def output_fn(outputs, process_mesh):
            assert outputs.is_dist()
            # TODO(chenweihang): replace by dist.unshard_dtensor later
            return paddle.to_tensor(outputs.numpy())

        # test shard parameters
        replicate_params_layer = dist.shard_layer(
            layer, self.mesh, input_fn=input_fn, output_fn=output_fn
        )

        x = paddle.randn([5, self.num_features])
        dense_out = replicate_params_layer(x)
        self.assertTrue(dense_out.is_dense())

        for param in replicate_params_layer.parameters():
            self.assertTrue(param.is_dist())
            for x in param.dist_attr.dims_mapping:
                self.assertEqual(x, -1)

        # test shard buffers
        test_buffer = paddle.randn([10])
        layer.register_buffer("test_buffer", test_buffer, persistable=True)
        sharded_buffers_layer = dist.shard_layer(
            layer, self.mesh, input_fn=input_fn, output_fn=output_fn
        )
        self.assertTrue(sharded_buffers_layer.test_buffer.is_dist())
        self.assertEqual(
            sharded_buffers_layer.test_buffer.dist_attr.dims_mapping, [-1]
        )

    def test_process_mesh_argument_error(self):
        layer = MyLayer(self.num_features, self.num_layers)

        exception = None
        try:
            dist.shard_layer(layer, None)
        except ValueError as ex:
            self.assertIn(
                "The argument `process_mesh` cannot be empty",
                str(ex),
            )
            exception = ex
        self.assertIsNotNone(exception)

        exception = None
        try:
            dist_attr = dist.DistAttr(
                mesh=self.mesh, sharding_specs=[None, None]
            )
            dist.shard_layer(layer, dist_attr)
        except ValueError as ex:
            self.assertIn(
                "The argument `process_mesh` is not `dist.ProcessMesh` type",
                str(ex),
            )
            exception = ex
        self.assertIsNotNone(exception)

    def test_shard_layer_static_mode(self):
        paddle.enable_static()
        layer = MyLayer(self.num_features, self.num_layers)

        exception = None
        try:
            dist.shard_layer(layer, self.mesh)
        except NotImplementedError as ex:
            self.assertIn(
                "`paddle.distributed.shard_layer` only supports dynamic graph mode now",
                str(ex),
            )
            exception = ex
        self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()
