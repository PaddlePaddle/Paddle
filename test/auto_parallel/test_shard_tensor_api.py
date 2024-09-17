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
from paddle.distributed import Replicate, Shard

in_pir_mode = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
    "FLAGS_enable_pir_api"
]


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
        input = paddle.rand([4, 1024, 512])
        d_tensor = dist.shard_tensor(
            input, self.mesh, [Replicate(), Replicate()]
        )

        self.assertEqual(d_tensor.process_mesh, self.mesh)

    def test_dynamic_mode_property_change(self):
        x = np.random.random([4, 1024, 512]).astype("float32")
        input = paddle.to_tensor(
            x, dtype="float32", place='cpu', stop_gradient=False
        )
        d_tensor = dist.shard_tensor(
            input,
            dtype="float64",
            place='gpu:0',
            stop_gradient=True,
            mesh=self.mesh,
            placements=[Replicate(), Replicate()],
        )

        self.assertEqual(d_tensor.dtype, paddle.float64)
        self.assertTrue(d_tensor.place.is_gpu_place())
        self.assertEqual(d_tensor.stop_gradient, True)

        self.assertEqual(d_tensor.process_mesh, self.mesh)

    def test_stop_gradient(self):
        x = paddle.ones([4, 1024, 512])
        x.stop_gradient = False
        x = dist.shard_tensor(x, self.mesh, [Shard(0), Replicate()])
        assert not x.stop_gradient


class TestShardTensorStatic(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
        )

    @switch_to_static_graph
    def test_static_mode(self):
        input = paddle.static.data(
            name="input",
            shape=[4, 1024, 512],
            dtype='float32',
        )
        d_tensor = dist.shard_tensor(input, self.mesh, [Shard(0), Replicate()])
        self.assertEqual(d_tensor.dist_attr().process_mesh, self.mesh)


class TestShardTensorStaticDy2Static(unittest.TestCase):
    def test_dy2static(self):
        @paddle.jit.to_static(full_graph=True, input_spec=[])
        def func():
            mesh = dist.ProcessMesh(
                [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
            )
            input = paddle.rand([4, 1024, 512])
            d_tensor = dist.shard_tensor(
                input, mesh, [Replicate(), Replicate()]
            )
            return d_tensor, mesh

        # dy_tensor, mesh = func()
        static_tensor = func.outputs[0]  # get the inputs of static program
        mesh = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
        )
        self.assertEqual(static_tensor.dist_attr().process_mesh, mesh)


class DemoNet(paddle.nn.Layer):
    def __init__(self, dist_attr):
        super().__init__()
        self.w0 = dist.shard_tensor(
            self.create_parameter(shape=[784, 784]), *dist_attr
        )

    def forward(self, x):
        return paddle.matmul(x, self.w0)


class TestShardTensorParameter(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.placements_and_mesh = (self.mesh, [Replicate()])

    def test_shard_parameter(self):
        x = np.random.random(size=[16, 784]).astype("float32")
        dist_x = dist.shard_tensor(x, *self.placements_and_mesh)
        net = DemoNet(self.placements_and_mesh)
        out = net(dist_x)
        self.assertEqual(out.shape, [16, 784])
        self.assertEqual(out.is_dist(), True)


if __name__ == "__main__":
    unittest.main()
