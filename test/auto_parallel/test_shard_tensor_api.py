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

    def test_dynamic(self):
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


if __name__ == "__main__":
    unittest.main()
