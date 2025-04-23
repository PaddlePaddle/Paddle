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


import paddle
import paddle.distributed as dist


class TestProcessMesh:
    def init_dist_env(self):
        dist.init_parallel_env()
        paddle.seed(2025)

    def test_get_submesh_with_dim(self):
        curr_rank = dist.get_rank()

        # Test 2D mesh
        mesh_2d = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["dp", "tp"])

        # Test case 1: Get submesh for dp dimension
        dp_mesh = mesh_2d.get_submesh_with_dim("dp")
        dp_mesh_ = mesh_2d["dp"]
        assert dp_mesh == dp_mesh_
        if curr_rank == 0:
            assert dp_mesh.process_ids == [0, 2]
        elif curr_rank == 1:
            assert dp_mesh.process_ids == [1, 3]

        # Test case 2: Get submesh for tp dimension
        tp_mesh = mesh_2d.get_submesh_with_dim("tp")
        tp_mesh_ = mesh_2d["tp"]
        assert tp_mesh == tp_mesh_
        if curr_rank == 0:
            assert tp_mesh.process_ids == [0, 1]
        elif curr_rank == 1:
            assert tp_mesh.process_ids == [0, 1]

        # Test case 3: 3D mesh with 8 cards (2x2x2)
        mesh_3d = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "tp"]
        )

        # Test each dimension
        pp_mesh = mesh_3d.get_submesh_with_dim("pp")
        pp_mesh_ = mesh_3d["pp"]
        assert pp_mesh == pp_mesh_
        dp_mesh = mesh_3d.get_submesh_with_dim("dp")
        dp_mesh_ = mesh_3d["dp"]
        assert dp_mesh == dp_mesh_
        tp_mesh = mesh_3d.get_submesh_with_dim("tp")
        tp_mesh_ = mesh_3d["tp"]
        assert tp_mesh == tp_mesh_

        # Verify pp dimension results
        if curr_rank == 0:
            assert pp_mesh.process_ids == [0, 4]
        elif curr_rank == 1:
            assert pp_mesh.process_ids == [1, 5]

        # Verify dp dimension results
        if curr_rank == 0:
            assert dp_mesh.process_ids == [0, 2]
        elif curr_rank == 1:
            assert dp_mesh.process_ids == [1, 3]

        # Verify tp dimension results
        if curr_rank == 0:
            assert tp_mesh.process_ids == [0, 1]
        elif curr_rank == 1:
            assert tp_mesh.process_ids == [0, 1]

        # Test case 4: When rank is not in the mesh
        mesh_small = dist.ProcessMesh([0, 1], dim_names=["x"])
        if curr_rank not in [0, 1]:
            assert mesh_small.get_submesh_with_dim("x") is None

    def test_get_group(self):
        curr_rank = dist.get_rank()

        # Test case 1: Single dimension mesh without dim_name
        mesh_1d = dist.ProcessMesh([0, 1], dim_names=["x"])
        if curr_rank in [0, 1]:
            group_1d = mesh_1d.get_group()
            assert isinstance(group_1d, dist.communication.group.Group)

            # Test case 2: Single dimension mesh with correct dim_name
            group_1d_with_name = mesh_1d.get_group(dim_name="x")
            assert isinstance(
                group_1d_with_name, dist.communication.group.Group
            )

            # Test case 3: Single dimension mesh with wrong dim_name
            try:
                mesh_1d.get_group(dim_name="wrong_name")
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

        # Test case 4: Multi-dimension mesh
        mesh_2d = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["dp", "tp"])
        if curr_rank in [0, 1, 2, 3]:
            # Test without dim_name
            try:
                mesh_2d.get_group()
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

            # Test with correct dim_name
            group_2d = mesh_2d.get_group(dim_name="dp")
            assert isinstance(group_2d, dist.communication.group.Group)

            # Test with wrong dim_name
            try:
                mesh_2d.get_group(dim_name="wrong_name")
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass

    def test_process_mesh(self):
        self.init_dist_env()
        self.test_get_submesh_with_dim()
        self.test_get_group()


if __name__ == '__main__':
    TestProcessMesh().test_process_mesh()
