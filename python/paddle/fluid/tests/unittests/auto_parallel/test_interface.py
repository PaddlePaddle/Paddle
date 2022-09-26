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
import paddle.nn.functional as F
import paddle.static as static
import paddle.distributed as dist
from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.dist_context import get_default_distributed_context
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
process_mesh1 = ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]],
                            dim_names=["x", "y"])
process_mesh2 = ProcessMesh(mesh=[0, 1, 2, 3], dim_names=["x"])


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        param_initializer = nn.initializer.Normal(mean=0.0,
                                                  std=initializer_range)

        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)

    def forward(self, input):
        auto.shard_tensor(self.linear0.weight, process_mesh1[0], [None, "y"])
        linear0 = auto.shard_op(self.linear0, process_mesh1,
                                [["y", None, None]], [[None, "x", None]])
        linear0_out = linear0(input)

        gelu = auto.shard_op(F.gelu, process_mesh1, [["y", "x", None], None])
        gelu_out = gelu(linear0_out, approximate=True)

        auto.shard_tensor(self.linear1.weight, shard_spec=["y", None])
        linear1 = auto.shard_op(self.linear1,
                                process_mesh1[1],
                                out_shard_specs=[["y", None, None]])
        linear1_out = linear1(gelu_out)

        return self.linear0, self.linear1, linear0_out, gelu_out, linear1_out


class TestAutoParallelAPI(unittest.TestCase):

    def test_api(self):
        # input
        input = static.data(name="input",
                            shape=[batch_size, sequence_len, hidden_size],
                            dtype='float32')
        label = static.data(name="label",
                            shape=[batch_size, sequence_len, 1],
                            dtype='float32')

        auto.shard_tensor(input, process_mesh1, ["x", None, None])
        auto.shard_tensor(label, process_mesh1, ["y", None, None])

        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       dropout_ratio=0.1,
                       initializer_range=0.02)

        with ProcessMesh(process_mesh1.mesh, process_mesh1.dim_names):
            linear0, linear1, linear0_out, gelu_out, linear1_out = mlp(input)

        default_program = paddle.fluid.default_main_program()
        default_dist_context = get_default_distributed_context()

        self.assertEqual(len(default_program.blocks[0].ops), 5)
        matmul0 = default_program.blocks[0].ops[0]
        self.assertEqual(matmul0.type, "matmul_v2")
        ewise_add0 = default_program.blocks[0].ops[1]
        self.assertEqual(ewise_add0.type, "elementwise_add")
        gelu = default_program.blocks[0].ops[2]
        self.assertEqual(gelu.type, "gelu")
        matmul1 = default_program.blocks[0].ops[3]
        self.assertEqual(matmul1.type, "matmul_v2")
        ewise_add1 = default_program.blocks[0].ops[4]
        self.assertEqual(ewise_add1.type, "elementwise_add")

        dist_input = default_dist_context.get_dist_tensor_for_program(input)
        self.assertEqual(dist_input.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_input.dist_attr.dims_mapping, [0, -1, -1])
        self.assertTrue(dist_input.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(dist_input.dist_attr.is_annotated("dims_mapping"))

        dist_input = default_dist_context.get_dist_tensor_for_program(label)
        self.assertEqual(dist_input.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_input.dist_attr.dims_mapping, [1, -1, -1])
        self.assertTrue(dist_input.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(dist_input.dist_attr.is_annotated("dims_mapping"))

        dist_linear0_weight = default_dist_context.get_dist_tensor_for_program(
            linear0.weight)
        self.assertEqual(dist_linear0_weight.dist_attr.process_mesh,
                         process_mesh1[0])
        self.assertEqual(dist_linear0_weight.dist_attr.dims_mapping, [-1, 0])
        self.assertTrue(
            dist_linear0_weight.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(
            dist_linear0_weight.dist_attr.is_annotated("dims_mapping"))

        dist_linear1_weight = default_dist_context.get_dist_tensor_for_program(
            linear1.weight)
        self.assertEqual(dist_linear1_weight.dist_attr.process_mesh,
                         process_mesh1)
        self.assertEqual(dist_linear1_weight.dist_attr.dims_mapping, [1, -1])
        self.assertTrue(
            dist_linear1_weight.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(
            dist_linear1_weight.dist_attr.is_annotated("dims_mapping"))

        dist_linear1_out = default_dist_context.get_dist_tensor_for_program(
            linear1_out)
        self.assertEqual(dist_linear1_out.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_linear1_out.dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(dist_linear1_out.dist_attr.is_annotated("process_mesh"))
        self.assertFalse(
            dist_linear1_out.dist_attr.is_annotated("dims_mapping"))

        dist_op = default_dist_context.get_dist_op_for_program(matmul0)
        self.assertEqual(dist_op.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))
        tensor_dist_attr = dist_op.dist_attr.get_input_dist_attr(input.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1)
        self.assertEqual(tensor_dist_attr.dims_mapping, [1, -1, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(tensor_dist_attr.is_annotated("dims_mapping"))

        dist_op = default_dist_context.get_dist_op_for_program(ewise_add0)
        self.assertEqual(dist_op.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(
            linear0_out.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1, 0, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(tensor_dist_attr.is_annotated("dims_mapping"))
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))

        dist_op = default_dist_context.get_dist_op_for_program(gelu)
        self.assertEqual(dist_op.dist_attr.process_mesh, process_mesh1)
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))
        tensor_dist_attr = dist_op.dist_attr.get_input_dist_attr(
            linear0_out.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1)
        self.assertEqual(tensor_dist_attr.dims_mapping, [1, 0, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(tensor_dist_attr.is_annotated("dims_mapping"))
        tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(gelu_out.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1)
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertFalse(tensor_dist_attr.is_annotated("dims_mapping"))

        dist_op = default_dist_context.get_dist_op_for_program(matmul1)
        self.assertEqual(dist_op.dist_attr.process_mesh, process_mesh1[1])
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))
        tensor_dist_attr = dist_op.dist_attr.get_input_dist_attr(gelu_out.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1[1])
        self.assertEqual(tensor_dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertFalse(tensor_dist_attr.is_annotated("dims_mapping"))

        dist_op = default_dist_context.get_dist_op_for_program(ewise_add1)
        self.assertEqual(dist_op.dist_attr.process_mesh, process_mesh1[1])
        self.assertEqual(dist_op.dist_attr.impl_type, "default")
        self.assertEqual(dist_op.dist_attr.impl_idx, 0)
        self.assertTrue(dist_op.dist_attr.is_annotated("process_mesh"))
        tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(
            linear1_out.name)
        self.assertEqual(tensor_dist_attr.process_mesh, process_mesh1[1])
        self.assertEqual(tensor_dist_attr.dims_mapping, [0, -1, -1])
        self.assertTrue(tensor_dist_attr.is_annotated("process_mesh"))
        self.assertTrue(tensor_dist_attr.is_annotated("dims_mapping"))


if __name__ == '__main__':
    unittest.main()
