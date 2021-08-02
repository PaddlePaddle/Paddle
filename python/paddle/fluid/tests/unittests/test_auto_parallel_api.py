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
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn as nn
import paddle.distributed as dist

paddle.enable_static()


def _append_attr_suffix(name):
    return name + core.kAutoParallelSuffix()


def _remove_attr_suffix(name):
    return name.strip(core.kAutoParallelSuffix())


LAST_PP_STAGE = 3
MASK = np.array([[0, 1], [1, 0], [1, 1]])
MESH = dist.ProcessMesh(np.array([[0, 1, 2], [3, 4, 5]]))


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size=128, hidden_size=4):
        super(SimpleNet, self).__init__()
        self.mesh = MESH
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x, y):
        x = dist.shard_tensor(x, self.mesh, dims_mapping=[0, -1])
        x = dist.set_shard_mask(x, MASK)
        emb_out = self.word_embeddings(x)

        dist.set_pipeline_stage(LAST_PP_STAGE)

        y = dist.shard_tensor(y, self.mesh, dims_mapping=[0, -1])
        dist.set_offload_device(y, "gpu:3")
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
            shape=[2, 4], value=4, dtype="float32")
        x, y, mesh = net.forward(data1, data2)
        mesh_attr = _append_attr_suffix('mesh_id')
        x_mesh_id = x.attr(mesh_attr)
        self.assertEqual(x_mesh_id, mesh._id)
        x_mesh = x.process_mesh
        self.assertEqual(x_mesh, mesh)
        shard_mask_attr = _append_attr_suffix('mask')
        self.assertEqual(x.attr(shard_mask_attr), MASK.flatten().tolist())
        self.assertEqual(x.shard_mask, MASK.flatten().tolist())
        offload_attr = _append_attr_suffix('offload_device')
        self.assertEqual(y.attr(offload_attr), "gpu:3")
        self.assertEqual(y.offload_device, "gpu:3")
        ops = paddle.static.default_main_program().block(0).ops
        first_op = ops[0]
        last_op = ops[-1]

        self.assertEqual(last_op.pipeline_stage, LAST_PP_STAGE)

        DIMS_MAPPING1 = [0, 1, -1]
        DIMS_MAPPING2 = [-1, 2, 0]
        kwargs = {'x': data2, 'y': data3}
        dist.shard_op(
            paddle.add,
            mesh=mesh,
            dims_mapping_dict={
                data2.name: DIMS_MAPPING1,
                data3.name: DIMS_MAPPING2
            },
            **kwargs)
        ops = paddle.static.default_main_program().block(0).ops
        last_op = ops[-1]
        print("main:", paddle.static.default_main_program())
        print("in type:", last_op.type)
        self.assertEqual(last_op.process_mesh, mesh)
        self.assertEqual(last_op.dims_mapping(data2.name), DIMS_MAPPING1)
        self.assertEqual(last_op.dims_mapping(data3.name), DIMS_MAPPING2)

    def test_process_mesh(self):
        mesh1 = dist.ProcessMesh(np.array([[0, 1, 2], [3, 4, 5]]), parent=MESH)
        mesh2 = dist.ProcessMesh(np.array([[0, 1, 2], [3, 4, 5]]), parent=mesh1)
        mesh3 = dist.ProcessMesh(np.array([[2, 3, 4], [5, 6, 7]]), parent=mesh1)
        mesh4 = dist.ProcessMesh(np.array([[0, 1], [2, 3]]), parent=mesh1)
        mesh5 = dist.ProcessMesh(np.array([[4, 5], [6, 7]]), parent=mesh1)

        self.assertEqual(MESH.parent, None)
        self.assertEqual(mesh1.parent, MESH)
        self.assertEqual(mesh4.parent, mesh1)
        self.assertEqual(mesh5.parent, mesh1)
        self.assertEqual(mesh1, mesh2)
        self.assertNotEqual(mesh1, mesh3)
        self.assertNotEqual(mesh4, mesh5)
        self.assertEqual(mesh2._id, mesh2._desc.id)
        self.assertEqual(mesh3.topology, mesh3._desc.topology)
        self.assertEqual(mesh3.topology, [2, 3])
        self.assertEqual(mesh3.process_group, [2, 3, 4, 5, 6, 7])
        self.assertEqual(mesh5.process_group, mesh5._desc.process_group)


if __name__ == '__main__':
    unittest.main()
