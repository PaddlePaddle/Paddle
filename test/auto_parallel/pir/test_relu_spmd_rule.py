# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestReluSpmdRule(unittest.TestCase):
    def test_build_replicated_program(self):
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program):
            mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
            x = paddle.static.data(name='x', shape=[64, 36])
            dist_x = dist.shard_tensor(x, mesh, [dist.Replicate()])
            dist_out = paddle.nn.functional.relu(dist_x)
        # relu out
        self.assertEqual(dist_out.shape, [64, 36])
        self.assertEqual(dist_out._local_shape, [64, 36])
        self.assertEqual(dist_out.dist_attr().dims_mapping, [-1, -1])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertEqual(dist_out.dist_attr().process_mesh.shape, [2])
        self.assertEqual(dist_out.dist_attr().process_mesh.process_ids, [0, 1])
        self.assertEqual(len(dist_out.dist_attr().partial_dims), 0)

    def test_build_col_parallel_program(self):
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program):
            mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
            x = paddle.static.data(name='x', shape=[64, 36])
            dist_x = dist.shard_tensor(x, mesh, [dist.Shard(0)])
            dist_out = paddle.nn.functional.relu(dist_x)

        # relu out
        self.assertEqual(dist_out.shape, [64, 36])
        self.assertEqual(dist_out._local_shape, [32, 36])
        self.assertEqual(dist_out.dist_attr().dims_mapping, [0, -1])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertEqual(dist_out.dist_attr().process_mesh.shape, [2])
        self.assertEqual(dist_out.dist_attr().process_mesh.process_ids, [0, 1])
        self.assertEqual(len(dist_out.dist_attr().partial_dims), 0)

    def test_build_row_parallel_program(self):
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program):
            mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
            x = paddle.static.data(name='x', shape=[64, 36])
            dist_x = dist.shard_tensor(x, mesh, [dist.Shard(1)])
            dist_out = paddle.nn.functional.relu(dist_x)

        # relu out
        self.assertEqual(dist_out.shape, [64, 36])
        self.assertEqual(dist_out._local_shape, [64, 18])
        self.assertEqual(dist_out.dist_attr().dims_mapping, [-1, 0])
        self.assertTrue(
            isinstance(
                dist_out.dist_attr().process_mesh,
                paddle.base.libpaddle.ProcessMesh,
            )
        )
        self.assertEqual(dist_out.dist_attr().process_mesh.shape, [2])
        self.assertEqual(dist_out.dist_attr().process_mesh.process_ids, [0, 1])
        self.assertEqual(dist_out.dist_attr().partial_dims, set())


if __name__ == "__main__":
    unittest.main()
