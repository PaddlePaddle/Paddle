#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn as nn

paddle.enable_static()


def _append_attr_suffix(name):
    return name + core.kAutoParallelSuffix()


def _remove_attr_suffix(name):
    return name.strip(core.kAutoParallelSuffix())


LAST_PP_STAGE = 3
MASK = [1]


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size=128, hidden_size=4):
        super(SimpleNet, self).__init__()
        mesh = paddle.distributed.ProcessMesh([2, 3])
        self.mesh = mesh
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x, y):
        x = paddle.distributed.shard_tensor(x, self.mesh, dims_mapping=[0, -1])
        x = paddle.distributed.set_shard_mask(x, MASK)
        emb_out = self.word_embeddings(x)

        paddle.distributed.set_pipeline_stage(LAST_PP_STAGE)

        y = paddle.distributed.shard_tensor(y, self.mesh, dims_mapping=[0, -1])
        paddle.distributed.set_offload_device(y, "gpu:3")
        linear1 = self.dense1(y)
        out = self.dense2(linear1)

        return x, y, self.mesh


class TestAutoParallelAPI(unittest.TestCase):
    def test_api(self):
        net = SimpleNet()
        data1 = fluid.layers.fill_constant(shape=[2, 4], value=1, dtype="int64")
        data2 = fluid.layers.fill_constant(
            shape=[2, 4], value=2, dtype="float32")
        data3 = fluid.layers.fill_constant(
            shape=[4, 3], value=2, dtype="float32")
        x, y, mesh = net.forward(data1, data2)
        mesh_attr = _append_attr_suffix('mesh_id')
        x_mesh_id = x.attr(mesh_attr)
        self.assertEqual(x_mesh_id, mesh.id)
        x_mesh = x.process_mesh
        self.assertEqual(x_mesh, mesh)
        shard_mask_attr = _append_attr_suffix('mask_out')
        self.assertEqual(x.attr(shard_mask_attr), MASK)
        self.assertEqual(x.shard_mask, MASK)
        offload_attr = _append_attr_suffix('offload_device')
        self.assertEqual(y.attr(offload_attr), "gpu:3")
        self.assertEqual(y.offload_device, "gpu:3")
        ops = paddle.static.default_main_program().block(0).ops
        first_op = ops[0]
        last_op = ops[-1]

        self.assertEqual(first_op.pipeline_stage, 0)
        self.assertEqual(last_op.pipeline_stage, LAST_PP_STAGE)

        DIMS_MAPPING1 = [0, 1, -1]
        DIMS_MAPPING2 = [-1, 2, 0]
        paddle.distributed.shard_op(
            paddle.matmul(data2, data3),
            mesh=mesh,
            input_dims_mapping={
                data2.name: DIMS_MAPPING1,
                data3.name: DIMS_MAPPING2
            })
        ops = paddle.static.default_main_program().block(0).ops
        last_op = ops[-1]
        self.assertEqual(last_op.process_mesh, mesh)
        self.assertEqual(last_op.dims_mapping(data2.name), DIMS_MAPPING1)
        self.assertEqual(last_op.dims_mapping(data3.name), DIMS_MAPPING2)

    def test_process_mesh(self):
        mesh1 = paddle.distributed.ProcessMesh([2, 3])
        mesh2 = paddle.distributed.ProcessMesh([2, 3])
        mesh3 = paddle.distributed.ProcessMesh(
            [2, 3], process_group=[2, 3, 4, 5, 6, 7])
        mesh4 = paddle.distributed.ProcessMesh([2, 2], parent_id=mesh1.id)
        mesh5 = paddle.distributed.ProcessMesh(
            [2, 2], process_group=[4, 5, 6, 7], parent_id=mesh1.id)

        self.assertEqual(mesh1.parent, None)
        self.assertEqual(mesh4.parent, mesh1)
        self.assertEqual(mesh5.parent, mesh1)
        self.assertEqual(mesh1, mesh2)
        self.assertNotEqual(mesh1, mesh3)
        self.assertNotEqual(mesh4, mesh5)
        self.assertEqual(mesh2.id, mesh2.desc.id)
        self.assertEqual(mesh3.topology, mesh3.desc.topology)
        self.assertEqual(mesh3.topology, [2, 3])
        self.assertEqual(mesh3.process_group, [2, 3, 4, 5, 6, 7])
        self.assertEqual(mesh5.process_group, mesh5.desc.process_group)
        self.assertEqual(mesh1.rank, 2)


if __name__ == '__main__':
    unittest.main()
