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


def shard_fn(layer_name, layer, process_mesh):
    if isinstance(layer, nn.Linear):
        for name, param in layer.named_parameters():
            dist_param = dist.shard_tensor(
                param, process_mesh, [dist.Replicate()]
            )
            layer.add_parameter(name, dist_param)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestShardLayer(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_features = 10
        self.num_layers = 10

    def test_shard_layer_base(self):
        layer = MyLayer(self.num_features, self.num_layers)

        # test shard parameters
        sharded_params_layer = dist.shard_layer(layer, self.mesh, shard_fn)

        for param in sharded_params_layer.parameters():
            self.assertTrue(param.is_dist())
            for x in param.placements:
                self.assertEqual(x, dist.Replicate())

        # test shard buffers
        test_buffer = paddle.randn([10])
        layer.register_buffer("test_buffer", test_buffer, persistable=True)
        sharded_buffers_layer = dist.shard_layer(layer, self.mesh, shard_fn)
        self.assertTrue(sharded_buffers_layer.test_buffer.is_dist())
        self.assertEqual(
            sharded_buffers_layer.test_buffer.placements, [dist.Replicate()]
        )

    def test_shard_layer_input_fn_and_output_fn(self):
        layer = MyLayer(self.num_features, self.num_layers)

        def input_fn(inputs, process_mesh):
            return dist.shard_tensor(
                inputs[0], process_mesh, [dist.Replicate()]
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
            for x in param.placements:
                self.assertEqual(x, dist.Replicate())

        # test shard buffers
        test_buffer = paddle.randn([10])
        layer.register_buffer("test_buffer", test_buffer, persistable=True)
        sharded_buffers_layer = dist.shard_layer(
            layer, self.mesh, input_fn=input_fn, output_fn=output_fn
        )
        self.assertTrue(sharded_buffers_layer.test_buffer.is_dist())
        self.assertEqual(
            sharded_buffers_layer.test_buffer.placements, [dist.Replicate()]
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
            placements = [dist.Replicate()]
            dist.shard_layer(layer, placements)
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
                "`paddle.distributed.shard_layer` only supports dynamic graph mode.",
                str(ex),
            )
            exception = ex
        self.assertIsNotNone(exception)
        paddle.disable_static()

    def create_data_loader(self):
        batch_size = 4
        hidden_size = self.num_features
        images = np.random.rand(batch_size, hidden_size).astype('float32')
        labels = np.random.rand(batch_size, hidden_size).astype('float32')
        dataset = RandomDataset(images, labels, batch_size)
        loader = paddle.io.DataLoader(dataset, batch_size=batch_size)
        return loader

    def test_shard_layer_to_static(self):
        def input_fn(inputs, process_mesh):
            return dist.shard_tensor(
                inputs[0], process_mesh, [dist.Replicate()]
            )

        def output_fn(outputs, process_mesh):
            return dist.shard_tensor(outputs, process_mesh, [dist.Shard(0)])

        layer = MyLayer(self.num_features, self.num_layers)

        sharded_layer = dist.shard_layer(
            layer, self.mesh, shard_fn, input_fn=input_fn, output_fn=output_fn
        )

        loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(loader, [self.mesh])
        dist_model = dist.to_static(sharded_layer, dist_loader)

        serial_main_program = dist_model.serial_main_program()
        for param in serial_main_program.all_parameters():
            self.assertTrue(param.dist_attr.is_annotated("dims_mapping"))
            self.assertEqual(param.dist_attr.dims_mapping, [-1, -1])

        input_var = serial_main_program.global_block().var("input0")
        output_var = serial_main_program.global_block().var(
            "matmul_v2_19.tmp_0"
        )
        self.assertListEqual(input_var.dist_attr.dims_mapping, [-1, -1])
        self.assertListEqual(output_var.dist_attr.dims_mapping, [0, -1])

        paddle.disable_static()

    def test_shard_layer_to_static_with_buffer(self):
        layer = MyLayer(self.num_features, self.num_layers)
        test_buffer0 = paddle.randn([3])
        layer.register_buffer("test_buffer0", test_buffer0, persistable=True)
        test_buffer1 = paddle.randn([10])
        layer.register_buffer("test_buffer1", test_buffer1, persistable=True)
        layer.test_buffer1 = dist.shard_tensor(
            layer.test_buffer1, self.mesh, [dist.Shard(0)]
        )
        sharded_buffers_layer = dist.shard_layer(layer, self.mesh, shard_fn)

        loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(loader, [self.mesh])
        dist_model = dist.to_static(sharded_buffers_layer, dist_loader)

        serial_main_program = dist_model.serial_main_program()
        for param in serial_main_program.all_parameters():
            self.assertTrue(param.dist_attr.is_annotated("dims_mapping"))
            self.assertEqual(param.dist_attr.dims_mapping, [-1, -1])

        buffer_vars = [
            var
            for var in serial_main_program.list_vars()
            if var.name.startswith("generated")
        ]
        buffer0_var = buffer_vars[1]
        buffer1_var = buffer_vars[0]
        self.assertTrue(buffer0_var.dist_attr.is_annotated("dims_mapping"))
        self.assertEqual(buffer0_var.dist_attr.dims_mapping, [-1])
        self.assertTrue(buffer1_var.dist_attr.is_annotated("dims_mapping"))
        self.assertEqual(buffer1_var.dist_attr.dims_mapping, [0])


if __name__ == '__main__':
    unittest.main()
