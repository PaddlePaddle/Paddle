# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.distributed.auto_parallel.process_mesh import (
    ProcessMesh,
    compute_compatible_process_mesh,
    merge_process_meshes,
)
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        param_initializer = nn.initializer.Normal(
            mean=0.0, std=initializer_range
        )

        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        return out


class TestProcessMesh(unittest.TestCase):
    def test_construction(self):
        mesh = [[0, 1, 2], [3, 4, 5]]
        process_mesh = ProcessMesh(mesh, dim_names=["x", "y"])
        self.assertEqual(process_mesh.shape, [2, 3])
        self.assertEqual(process_mesh.process_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(process_mesh.dim_names, ["x", "y"])
        self.assertEqual(process_mesh.ndim, 2)
        self.assertEqual(process_mesh, process_mesh)
        self.assertEqual(str(process_mesh), str(process_mesh))

        sub_process_mesh1 = process_mesh[0]
        self.assertEqual(sub_process_mesh1.shape, [3])
        self.assertEqual(sub_process_mesh1.process_ids, [0, 1, 2])
        self.assertEqual(sub_process_mesh1.dim_names, ["y"])
        self.assertEqual(sub_process_mesh1.ndim, 1)

        sub_process_mesh2 = process_mesh[:, 1]
        self.assertEqual(sub_process_mesh2.shape, [2])
        self.assertEqual(sub_process_mesh2.process_ids, [1, 4])
        self.assertEqual(sub_process_mesh2.dim_names, ["x"])
        self.assertEqual(sub_process_mesh2.ndim, 1)

        sub_process_mesh3 = sub_process_mesh2[:]
        self.assertEqual(sub_process_mesh3.shape, [2])
        self.assertEqual(sub_process_mesh3.process_ids, [1, 4])
        self.assertEqual(sub_process_mesh3.dim_names, ["x"])
        self.assertEqual(sub_process_mesh3.ndim, 1)

        sub_process_mesh4 = process_mesh[1, 1]
        self.assertEqual(sub_process_mesh4.shape, [1])
        self.assertEqual(sub_process_mesh4.process_ids, [4])
        self.assertEqual(sub_process_mesh4.dim_names, ["d0"])
        self.assertEqual(sub_process_mesh4.ndim, 1)

        sub_process_mesh5 = sub_process_mesh3[0]
        self.assertEqual(sub_process_mesh5.shape, [1])
        self.assertEqual(sub_process_mesh5.process_ids, [1])
        self.assertEqual(sub_process_mesh5.dim_names, ["d0"])
        self.assertEqual(sub_process_mesh5.ndim, 1)

    def test_context_manager(self):
        mesh = np.array([1, 2, 3, 4])
        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32',
        )
        label = static.data(
            name="label", shape=[batch_size, sequence_len, 1], dtype='float32'
        )

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )

        with ProcessMesh(mesh, ["d"]):
            out = mlp(input)

        default_program = paddle.base.default_main_program()
        default_dist_context = get_default_distributed_context()

        for block in default_program.blocks:
            for tensor in block.vars.values():
                dist_tensor = default_dist_context.get_dist_tensor_for_program(
                    tensor
                )
                if dist_tensor is not None:
                    self.assertEqual(
                        dist_tensor.dist_attr.process_mesh, ProcessMesh(mesh)
                    )
            for op in block.ops:
                dist_op = default_dist_context.get_dist_op_for_program(op)
                if dist_op is not None:
                    self.assertEqual(
                        dist_op.dist_attr.process_mesh, ProcessMesh(mesh)
                    )

    def test_compute_compatible_process_mesh(self):
        process_mesh1 = ProcessMesh(
            [[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"]
        )
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, None]
        )
        self.assertEqual(compatible_process_mesh, process_mesh1)
        compatible_process_mesh = compute_compatible_process_mesh(
            [None, process_mesh1]
        )
        self.assertEqual(compatible_process_mesh, process_mesh1)

        process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(compatible_process_mesh, process_mesh1)
        self.assertEqual(compatible_process_mesh, process_mesh2)

        process_mesh2 = ProcessMesh([[0, 1, 2, 3, 4, 5]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(compatible_process_mesh, process_mesh1)

        process_mesh2 = ProcessMesh([[0, 1, 2]])
        compatible_process_mesh = compute_compatible_process_mesh(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(compatible_process_mesh, process_mesh1)

    def test_merge_process_meshes(self):
        process_mesh1 = ProcessMesh(
            [[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"]
        )
        merged_process_mesh = merge_process_meshes([process_mesh1, None])
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))
        merged_process_mesh = merge_process_meshes([None, process_mesh1])
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        merged_process_mesh = merge_process_meshes(
            [process_mesh1, paddle.base.core.ProcessMesh()]
        )
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))
        merged_process_mesh = merge_process_meshes(
            [paddle.base.core.ProcessMesh(), process_mesh1]
        )
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        merged_process_mesh = merge_process_meshes(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[0, 1, 2]])
        merged_process_mesh = merge_process_meshes(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5]))

        process_mesh2 = ProcessMesh([[6, 7]])
        merged_process_mesh = merge_process_meshes(
            [process_mesh1, process_mesh2]
        )
        self.assertEqual(
            merged_process_mesh, ProcessMesh([0, 1, 2, 3, 4, 5, 6, 7])
        )

    def test_get_rank_and_dim_size(self):
        mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"])
        self.assertEqual(mesh.get_dim_size("x"), 2)
        self.assertEqual(mesh.get_dim_size(0), 2)
        self.assertEqual(mesh.get_dim_size("y"), 3)
        self.assertEqual(mesh.get_dim_size(1), 3)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id(None, 0), 0)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id(None, 8), -1)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id('x', 2), 0)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id(0, 4), 1)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id('y', 3), 0)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id('y', 4), 1)
        self.assertEqual(mesh.get_rank_by_dim_and_process_id(1, 5), 2)


if __name__ == "__main__":
    unittest.main()
