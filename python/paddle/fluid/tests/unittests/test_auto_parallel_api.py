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
import paddle.distributed as dist
from paddle.distributed.auto_parallel.dist_context import get_default_distributed_context
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh

paddle.enable_static()

process_mesh1 = [0, 1, 2, 3]
process_mesh2 = [[0, 1, 2], [3, 4, 5]]


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size=128, hidden_size=4):
        super(SimpleNet, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x, y):
        # Test shard_tensor interface with dist_attr arg
        x = dist.shard_tensor(
            x,
            dist_attr={"process_mesh": process_mesh1,
                       "dims_mapping": [0, -1]})
        emb_out = self.word_embeddings(x)
        # Test shard_tensor interface with no dist_attr arg
        y = dist.shard_tensor(y)
        linear1 = self.dense1(y)
        out = self.dense2(linear1)

        return x, y


class TestAutoParallelAPI(unittest.TestCase):
    def test_api(self):
        dist_context = get_default_distributed_context()

        net = SimpleNet()
        data1 = fluid.layers.fill_constant(shape=[2, 4], value=1, dtype="int64")
        data2 = fluid.layers.fill_constant(
            shape=[2, 4], value=2, dtype="float32")
        data3 = fluid.layers.fill_constant(
            shape=[2, 4], value=4, dtype="float32")

        x, y = net.forward(data1, data2)

        dist_x = dist_context.get_dist_tensor_for_program(x)
        self.assertEqual(dist_x.dist_attr.process_mesh.processes, process_mesh1)
        self.assertEqual(dist_x.dist_attr.dims_mapping, [0, -1])
        self.assertEqual(dist_x.dist_attr.shard_sizes, None)
        self.assertEqual(dist_x.dist_attr.device_placement, None)
        self.assertTrue(dist_x.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(dist_x.dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(dist_x.dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(dist_x.dist_attr.is_annotated("device_placement"))

        dist_y = dist_context.get_dist_tensor_for_program(y)
        self.assertEqual(dist_y.dist_attr.process_mesh, None)
        self.assertEqual(dist_y.dist_attr.dims_mapping, [-1, -1])
        self.assertEqual(dist_y.dist_attr.shard_sizes, None)
        self.assertEqual(dist_y.dist_attr.device_placement, None)
        self.assertFalse(dist_y.dist_attr.is_annotated("process_mesh"))
        self.assertFalse(dist_y.dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(dist_y.dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(dist_y.dist_attr.is_annotated("device_placement"))

        # Test shard_op interface with dist_attr
        dims_mapping1 = [0, 1]
        dims_mapping2 = [-1, 0]
        dist_add = dist.shard_op(
            paddle.add,
            dist_attr={
                data2: {
                    "process_mesh": process_mesh2,
                    "dims_mapping": dims_mapping1
                },
                data3: {
                    "dims_mapping": dims_mapping2
                }
            })
        results = dist_add(data2, data3)
        ops = paddle.static.default_main_program().block(0).ops
        last_op = ops[-1]

        dist_op = dist_context.get_dist_op_for_program(last_op)
        self.assertEqual(dist_op.dist_attr.process_mesh,
                         ProcessMesh(process_mesh2))
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))

        data2_dist_attr = dist_op.dist_attr.get_input_dist_attr(data2.name)
        self.assertEqual(data2_dist_attr.process_mesh,
                         dist_op.dist_attr.process_mesh)
        self.assertEqual(data2_dist_attr.dims_mapping, dims_mapping1)
        self.assertEqual(data2_dist_attr.shard_sizes, None)
        self.assertEqual(data2_dist_attr.device_placement, None)
        self.assertTrue(data2_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(data2_dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(data2_dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(data2_dist_attr.is_annotated("device_placement"))

        data3_dist_attr = dist_op.dist_attr.get_input_dist_attr(data3.name)
        self.assertEqual(data3_dist_attr.process_mesh,
                         dist_op.dist_attr.process_mesh)
        self.assertEqual(data3_dist_attr.dims_mapping, dims_mapping2)
        self.assertEqual(data3_dist_attr.shard_sizes, None)
        self.assertEqual(data3_dist_attr.device_placement, None)
        self.assertTrue(data3_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(data3_dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(data3_dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(data3_dist_attr.is_annotated("device_placement"))

        # Test shard_op interface with dist_attr
        dist_add = dist.shard_op(paddle.add)
        results = dist_add(data2, data3)
        ops = paddle.static.default_main_program().block(0).ops
        last_op = ops[-1]
        dist_op = dist_context.get_dist_op_for_program(last_op)
        self.assertEqual(dist_op.dist_attr.process_mesh, None)
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertFalse(dist_op.dist_attr.is_annotated("process_mesh"))

        data2_dist_attr = dist_op.dist_attr.get_input_dist_attr(data2.name)
        self.assertEqual(data2_dist_attr.process_mesh,
                         dist_op.dist_attr.process_mesh)
        self.assertEqual(data2_dist_attr.dims_mapping, [-1, -1])
        self.assertEqual(data2_dist_attr.shard_sizes, None)
        self.assertEqual(data2_dist_attr.device_placement, None)
        self.assertFalse(data2_dist_attr.is_annotated("process_mesh"))
        self.assertFalse(data2_dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(data2_dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(data2_dist_attr.is_annotated("device_placement"))

        data3_dist_attr = dist_op.dist_attr.get_input_dist_attr(data3.name)
        self.assertEqual(data3_dist_attr.process_mesh,
                         dist_op.dist_attr.process_mesh)
        self.assertEqual(data3_dist_attr.dims_mapping, [-1, -1])
        self.assertEqual(data3_dist_attr.shard_sizes, None)
        self.assertEqual(data3_dist_attr.device_placement, None)
        self.assertFalse(data3_dist_attr.is_annotated("process_mesh"))
        self.assertFalse(data3_dist_attr.is_annotated("dims_mapping"))
        self.assertFalse(data3_dist_attr.is_annotated("shard_sizes"))
        self.assertFalse(data3_dist_attr.is_annotated("device_placement"))


if __name__ == '__main__':
    unittest.main()
