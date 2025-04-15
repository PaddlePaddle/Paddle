# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.distributed as dist


class TestProcessMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def test_get_submesh_with_dim(self):
        # Test 2D mesh
        mesh_2d = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["dp", "tp"]
        )

        # Test case 1: Get submesh for dp dimension normally
        dp_mesh = mesh_2d.get_submesh_with_dim("dp")
        curr_rank = dist.get_rank()
        if curr_rank in [0, 4]:
            self.assertEqual(dp_mesh.process_ids, [0, 4])
        elif curr_rank in [1, 5]:
            self.assertEqual(dp_mesh.process_ids, [1, 5])

        # Test case 2: Get submesh for tp dimension normally
        tp_mesh = mesh_2d.get_submesh_with_dim("tp")
        if curr_rank in [0, 1, 2, 3]:
            self.assertEqual(tp_mesh.process_ids, [0, 1, 2, 3])
        elif curr_rank in [4, 5, 6, 7]:
            self.assertEqual(tp_mesh.process_ids, [4, 5, 6, 7])

        # Test case 3: 3D mesh
        mesh_3d = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "tp"]
        )

        # Test each dimension
        pp_mesh = mesh_3d.get_submesh_with_dim("pp")
        dp_mesh = mesh_3d.get_submesh_with_dim("dp")
        tp_mesh = mesh_3d.get_submesh_with_dim("tp")

        # Verify results based on current rank
        if curr_rank in [0, 4]:
            self.assertEqual(pp_mesh.process_ids, [0, 4])
        elif curr_rank in [1, 5]:
            self.assertEqual(pp_mesh.process_ids, [1, 5])

        # Test case 4: When rank is not in the mesh
        mesh_small = dist.ProcessMesh([0, 1], dim_names=["x"])
        if curr_rank not in [0, 1]:
            self.assertIsNone(mesh_small.get_submesh_with_dim("x"))

    def test_get_group(self):
        dist.init_parallel_env()
        # Test case 1: Single dimension mesh without specifying dim_name
        mesh_1d = dist.ProcessMesh([0, 1], dim_names=["x"])
        group_1d = mesh_1d.get_group()
        self.assertIsInstance(group_1d, dist.communication.group.Group)

        # Test case 2: Single dimension mesh with correct dim_name
        group_1d_with_name = mesh_1d.get_group(dim_name="x")
        self.assertIsInstance(
            group_1d_with_name, dist.communication.group.Group
        )

        # Test case 3: Single dimension mesh with wrong dim_name
        with self.assertRaises(ValueError):
            mesh_1d.get_group(dim_name="wrong_name")

        # Test case 4: Multi-dimension mesh without specifying dim_name
        mesh_2d = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["dp", "tp"])
        with self.assertRaises(ValueError):
            mesh_2d.get_group()

        # Test case 5: Multi-dimension mesh with correct dim_name
        group_2d = mesh_2d.get_group(dim_name="dp")
        self.assertIsInstance(group_2d, dist.communication.group.Group)

        # Test case 6: Multi-dimension mesh with wrong dim_name
        with self.assertRaises(ValueError):
            mesh_2d.get_group(dim_name="wrong_name")


if __name__ == '__main__':
    unittest.main()
