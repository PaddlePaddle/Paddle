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


class TestDistAttrBasic(unittest.TestCase):
    def test_mesh_argument_error(self):
        exception = None
        try:
            mesh = [[0, 1], [2, 3]]
            dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])
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
                mesh=mesh, sharding_specs={"x": 0, "y": 1}
            )
        except ValueError as ex:
            self.assertIn(
                "The sharding_specs must be an instance of list", str(ex)
            )
            exception = ex

        self.assertIsNotNone(exception)


class TestShardTensorBasic(unittest.TestCase):
    # remove this test after static mode is supported
    def test_static_mode_unimplemented(self):
        exception = None
        try:
            paddle.enable_static()
            mesh = dist.ProcessMesh(
                [[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"]
            )
            dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])
            a = paddle.to_tensor([[1, 2, 3], [5, 6, 7]])
            d_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        except NotImplementedError as ex:
            self.assertIn(
                "The `paddle.distributed.shard_tensor` for static mode will be implemented later",
                str(ex),
            )
            exception = ex
            paddle.disable_static()

        self.assertIsNotNone(exception)


if __name__ == "__main__":
    unittest.main()
