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
import paddle.nn as nn

paddle.enable_static()


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size=128, hidden_size=4):
        super(SimpleNet, self).__init__()
        mesh = paddle.distributed.ProcessMesh([2, 3])
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x, y):
        x = paddle.distributed.shard_tensor(x, mesh, dims_mapping=[0, -1])
        emb_out = self.word_embeddings(x)

        y = paddle.distributed.shard_tensor(y, mesh, dims_mapping=[0, -1])
        linear1 = self.dense1(y)
        out = self.dense2(linear1)
        return emb_out, linear1, out


class TestAutoParallelAPI(unittest.TestCase):
    #def test_api(self):
    #    net = SimpleNet()
    #    x = fluid.layers.fill_constant(shape=[2, 4], value=1, dtype="int64")
    #    y = fluid.layers.fill_constant(shape=[2, 4], value=2, dtype="float32")
    #    emb_out, linear1, out = net.forward(x, y)
    #    self.assertEqual(x.desc.distributed_attr('mesh_topology'), [2, 3])
    #    self.assertEqual(
    #        x.desc.distributed_attr('mesh_group'), [0, 1, 2, 3, 4, 5])
    #    self.assertEqual(y.desc.distributed_attr('mesh_topology'), [2, 3])
    #    self.assertEqual(
    #        y.desc.distributed_attr('mesh_group'), [0, 1, 2, 3, 4, 5])

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
