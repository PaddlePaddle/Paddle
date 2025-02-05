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
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

paddle.enable_static()


def get_program(mesh, placements, local_mesh_dim):
    main_program = paddle.base.Program()
    with paddle.base.program_guard(main_program):
        x = paddle.static.data(name='x', shape=[64, 36, 24])
        y = paddle.static.data(name='y', shape=[64, 36, 24])
        x.stop_gradient = False
        y.stop_gradient = False
        dist_x = dist.shard_tensor(x, mesh, placements)
        dist_y = dist.shard_tensor(y, mesh, placements)
        local_tensors = dist.auto_parallel.api.moe_sub_mesh_tensors(
            dist_x, mesh, local_mesh_dim, placements
        )
        out = dist.auto_parallel.api.moe_global_mesh_tensor(
            local_tensors, mesh, placements, local_mesh_dim
        )
        loss = dist_y - out

    dist_program = main_program.clone()
    apply_mix2dist_pass(dist_program)
    dist_loss_value = dist_program.global_block().ops[-1].result(0)

    with paddle.static.program_guard(dist_program):
        params_grads = paddle.autograd.ir_backward.append_backward(
            dist_loss_value
        )

    return dist_program


class TestMoEApi(unittest.TestCase):
    def test_1Dmesh_2experts(self):
        mesh = dist.ProcessMesh([0, 1])
        global_placements = [dist.Shard(0)]
        local_mesh_dim = 0
        dist_program = get_program(mesh, global_placements, local_mesh_dim)
        ops = dist_program.global_block().ops

        global_mesh = [0, 1]
        global_dims_mapping = [0, -1, -1]
        local_meshes = [[0], [1]]
        local_dims_mapping = [-1, -1, -1]
        self.check_results(
            ops,
            global_mesh,
            global_dims_mapping,
            local_meshes,
            local_dims_mapping,
        )

    def test_2Dmesh_4experts(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3], [4, 5], [6, 7]])
        global_placements = [dist.Shard(0), dist.Shard(2)]
        local_mesh_dim = -2
        dist_program = get_program(mesh, global_placements, local_mesh_dim)
        ops = dist_program.global_block().ops

        global_mesh = [0, 1, 2, 3, 4, 5, 6, 7]
        local_meshes = [[0, 1], [2, 3], [4, 5], [6, 7]]
        global_dims_mapping = [0, -1, 1]
        local_dims_mapping = [-1, -1, 1]
        self.check_results(
            ops,
            global_mesh,
            global_dims_mapping,
            local_meshes,
            local_dims_mapping,
        )

    def test_error(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3], [4, 5], [6, 7]])
        global_placements = [dist.Shard(0), dist.Shard(2)]
        local_mesh_dim = -3
        with self.assertRaises(ValueError):
            dist_program = get_program(mesh, global_placements, local_mesh_dim)

        with self.assertRaises(ValueError):
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                x = paddle.static.data(name='x', shape=[64, 36, 24])
                y = paddle.static.data(name='y', shape=[64, 36, 24])
                x.stop_gradient = False
                y.stop_gradient = False
                dist_x = dist.shard_tensor(x, mesh, global_placements)
                dist_y = dist.shard_tensor(y, mesh, global_placements)
                local_tensors = dist.auto_parallel.api.moe_sub_mesh_tensors(
                    dist_x, None, local_mesh_dim, global_placements
                )

    def check_dist_attr(self, op, meshes, dims_mapping):
        results = op.results()
        self.assertEqual(len(results), len(meshes))
        for i, result in enumerate(results):
            dist_attr = result.dist_attr()
            self.assertEqual(dist_attr.process_mesh.process_ids, meshes[i])
            self.assertEqual(dist_attr.dims_mapping, dims_mapping)

    def check_results(
        self,
        ops,
        global_mesh,
        global_dims_mapping,
        local_meshes,
        local_dims_mapping,
    ):
        # local_tensors_from_dtensor op
        self.check_dist_attr(ops[4], local_meshes, local_dims_mapping)
        # dtensor_from_local_list op
        self.check_dist_attr(ops[5], [global_mesh], global_dims_mapping)
        # grad op for dtensor_from_local_list
        self.check_dist_attr(ops[10], local_meshes, local_dims_mapping)
        # grad op for local_tensors_from_dtensor op
        self.check_dist_attr(ops[11], [global_mesh], global_dims_mapping)


if __name__ == "__main__":
    unittest.main()
