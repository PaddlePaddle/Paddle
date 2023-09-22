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
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)


class TestDistAttrBasic(unittest.TestCase):
    def test_mesh_argument_error(self):
        exception = None
        try:
            mesh = [[0, 1], [2, 3]]
            dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        except ValueError as ex:
            self.assertIn(
                "The mesh must be an instance of paddle.distributed.ProcessMesh",
                str(ex),
            )
            exception = ex

        self.assertIsNotNone(exception)

    def test_sharding_specs_argument_error(self):
        exception = None
        try:
            mesh = dist.ProcessMesh(
                [[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"]
            )
            dist_attr = dist.DistAttr(
                mesh=mesh, sharding_specs={"x": None, "y": None}
            )
        except ValueError as ex:
            self.assertIn(
                "The sharding_specs must be an instance of list", str(ex)
            )
            exception = ex

        self.assertIsNotNone(exception)


class TestShardTensorDynamic(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
        )

    def test_dynamic_mode_basic(self):
        dist_attr = dist.DistAttr(
            mesh=self.mesh, sharding_specs=[None, None, None]
        )

        input = paddle.rand([4, 1024, 512])
        d_tensor = dist.shard_tensor(input, dist_attr=dist_attr)
        print(dist_attr.dims_mapping)

        self.assertEqual(d_tensor.dist_attr.process_mesh, self.mesh)
        self.assertEqual(d_tensor.dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(d_tensor.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(d_tensor.dist_attr.is_annotated("dims_mapping"))

    def test_dynamic_mode_property_change(self):
        dist_attr = dist.DistAttr(
            mesh=self.mesh, sharding_specs=[None, None, None]
        )

        x = np.random.random([4, 1024, 512]).astype("float32")
        input = paddle.to_tensor(
            x, dtype="float32", place='cpu', stop_gradient=False
        )
        d_tensor = dist.shard_tensor(
            input,
            dtype="float64",
            place='gpu:0',
            stop_gradient=True,
            dist_attr=dist_attr,
        )

        self.assertEqual(d_tensor.dtype, paddle.float64)
        self.assertTrue(d_tensor.place.is_gpu_place())
        self.assertEqual(d_tensor.stop_gradient, True)

        self.assertEqual(d_tensor.dist_attr.process_mesh, self.mesh)
        self.assertEqual(d_tensor.dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(d_tensor.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(d_tensor.dist_attr.is_annotated("dims_mapping"))


class TestShardTensorStatic(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
        )

    @switch_to_static_graph
    def test_static_mode(self):
        dist_attr = dist.DistAttr(
            mesh=self.mesh, sharding_specs=['x', None, None]
        )

        input = paddle.static.data(
            name="input",
            shape=[4, 1024, 512],
            dtype='float32',
        )
        d_tensor = dist.shard_tensor(input, dist_attr=dist_attr)

        default_dist_context = get_default_distributed_context()
        dist_input = default_dist_context.get_dist_tensor_for_program(input)
        self.assertEqual(dist_input.dist_attr.process_mesh, self.mesh)
        self.assertEqual(dist_input.dist_attr.dims_mapping, [0, -1, -1])
        self.assertTrue(dist_input.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(dist_input.dist_attr.is_annotated("dims_mapping"))


class TestShardTensorStaticDy2Static(unittest.TestCase):
    def test_dy2static(self):
        @paddle.jit.to_static
        def func():
            mesh = dist.ProcessMesh(
                [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
            )
            dist_attr = dist.DistAttr(
                mesh=mesh, sharding_specs=[None, None, None]
            )

            input = paddle.rand([4, 1024, 512])
            d_tensor = dist.shard_tensor(input, dist_attr=dist_attr)
            return input, mesh

        dy_tensor, mesh = func()
        static_tensor = func.outputs[0]  # get the inputs of static program

        default_dist_context = get_default_distributed_context()
        dist_input = default_dist_context.get_dist_tensor_for_program(
            static_tensor
        )
        self.assertEqual(dist_input.dist_attr.process_mesh, mesh)
        self.assertEqual(dist_input.dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(dist_input.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(dist_input.dist_attr.is_annotated("dims_mapping"))


class DemoNet(paddle.nn.Layer):
    def __init__(self, dist_attr):
        super().__init__()
        self.w0 = dist.shard_tensor(
            self.create_parameter(shape=[784, 784]), dist_attr=dist_attr
        )

    def forward(self, x):
        return paddle.matmul(x, self.w0)


class TestShardTensorParameter(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.dist_attr = dist.DistAttr(
            mesh=self.mesh, sharding_specs=[None, None]
        )

    def test_shard_parameter(self):
        x = np.random.random(size=[16, 784]).astype("float32")
        dist_x = dist.shard_tensor(x, dist_attr=self.dist_attr)
        net = DemoNet(self.dist_attr)
        out = net(dist_x)
        self.assertEqual(out.shape, [16, 784])
        self.assertEqual(out.is_dist(), True)
        self.assertEqual(out.dist_attr, self.dist_attr)


if __name__ == "__main__":
    unittest.main()
