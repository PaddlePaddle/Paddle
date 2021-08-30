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
import functools
import operator
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn as nn
import paddle.distributed as dist

paddle.enable_static()


def _flatten_nested_list(nested_list):
    result = functools.reduce(operator.iconcat, nested_list, [])
    return result


def _append_attr_suffix(name):
    return name + core.kAutoParallelSuffix()


LAST_PP_STAGE = 3
MASK = [[0, 1], [1, 0], [1, 1]]
MESH = dist.ProcessMesh([[0, 1, 2], [3, 4, 5]])


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size=128, hidden_size=4):
        super(SimpleNet, self).__init__()
        self.mesh = MESH
        self.mesh.set_placement([5, 4, 3, 2, 1, 0])
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x, y):
        x = dist.shard_tensor(x, self.mesh, dim_mapping=[0, -1])
        x = dist.set_shard_mask(x, MASK)
        emb_out = self.word_embeddings(x)

        dist.set_pipeline_stage(LAST_PP_STAGE)

        y = dist.shard_tensor(y, self.mesh, dim_mapping=[0, -1])
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
        x_mesh_id = x._get_attr(mesh_attr)
        self.assertEqual(x_mesh_id, mesh._id)
        x_mesh = x.process_mesh

        allatts = x.attr_names
        self.assertEqual(x_mesh, mesh)
        shard_mask_attr = _append_attr_suffix('mask')
        self.assertEqual(
            x._get_attr(shard_mask_attr), _flatten_nested_list(MASK))
        self.assertEqual(x.shard_mask, _flatten_nested_list(MASK))
        offload_attr = _append_attr_suffix('offload_device')
        self.assertEqual(y._get_attr(offload_attr), "gpu:3")
        self.assertEqual(y.desc.has_attr(offload_attr), True)
        self.assertEqual(y.offload_device, "gpu:3")
        y._remove_attr(offload_attr)
        self.assertEqual(y._has_attr(offload_attr), False)
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
            dim_mapping_dict={
                data2.name: DIMS_MAPPING1,
                data3.name: DIMS_MAPPING2
            },
            **kwargs)
        ops = paddle.static.default_main_program().block(0).ops
        last_op = ops[-1]

        self.assertEqual(last_op.process_mesh, mesh)
        attr_name = "IN_" + data2.name
        attr_name = _append_attr_suffix(attr_name)
        self.assertEqual(last_op.attr(attr_name), DIMS_MAPPING1)
        attr_name = "IN_" + data3.name
        attr_name = _append_attr_suffix(attr_name)
        self.assertEqual(last_op.attr(attr_name), DIMS_MAPPING2)

    def test_process_mesh(self):
        mesh1 = dist.ProcessMesh([[0, 1, 2], [3, 4, 5]], parent=MESH)
        mesh2 = dist.ProcessMesh([[0, 1, 2], [3, 4, 5]], parent=mesh1)
        mesh3 = dist.ProcessMesh([[0, 1], [2, 3]], parent=mesh1)
        mesh4 = dist.ProcessMesh([[2, 3], [4, 5]], parent=mesh1)

        self.assertEqual(MESH.parent, None)
        self.assertEqual(mesh1.parent, MESH)
        self.assertEqual(mesh1._desc.parent, MESH._id)
        self.assertEqual(mesh3.parent, mesh1)
        self.assertEqual(mesh4.parent, mesh1)
        self.assertEqual(mesh1, mesh2)
        self.assertNotEqual(mesh3, mesh4)
        self.assertEqual(mesh2._id, mesh2._desc.id)
        self.assertEqual(mesh3.topology, mesh3._desc.topology)
        self.assertEqual(mesh3.topology, [2, 2])
        self.assertEqual(mesh3.process_group, [0, 1, 2, 3])
        self.assertEqual(mesh4.process_group, mesh4._desc.process_group)


if __name__ == '__main__':
    unittest.main()
