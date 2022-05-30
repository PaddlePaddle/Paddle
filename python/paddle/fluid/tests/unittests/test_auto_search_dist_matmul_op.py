# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy

import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.fluid.core as core
from paddle.fluid import layers
from paddle.distributed.auto_parallel.operators.common import DistributedOperatorImplContainer
from paddle.distributed.auto_parallel.operators.common import DistributedOperatorImpl
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container
from paddle.distributed.auto_parallel.dist_context import DistributedContext, DistributedOperatorContext
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.distributed.auto_parallel.dist_op import DistributedOperator
paddle.enable_static()
device = "gpu" if core.is_compiled_with_cuda() else "cpu"


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sqrt_hidden_size = 32
        double_hidden_size = 64

        input = static.data(name="input", shape=[8, 8, 16], dtype='int32')
        input = paddle.reshape(input, [hidden_size])
        input = paddle.reshape(input, [sqrt_hidden_size, sqrt_hidden_size])
        embedding = paddle.nn.Embedding(2, batch_size, sparse=True)
        input = embedding(input)
        input = paddle.reshape(input, [hidden_size, batch_size])
        input = paddle.transpose(input, perm=[1, 0])
        matmulinput = static.data(
            name="matmulinput",
            shape=[hidden_size, hidden_size],
            dtype='float32')
        input = layers.matmul(x=input, y=matmulinput)
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')
        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)
        m = paddle.nn.Softmax()
        loss = m(loss)
    return loss, train_program, start_program


class TestCompatible(unittest.TestCase):
    def test_matmulv2_matmul_2_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)

        with static.program_guard(program,
                                  start_program), utils.unique_name.guard():
            matmulx3 = static.data(
                name="matmulx3", shape=[6, 2, 6], dtype='float32')
            matmuly3 = static.data(
                name="matmuly3", shape=[6, 6], dtype='float32')
            output1 = paddle.matmul(x=matmulx3, y=matmuly3)
            output_1 = layers.matmul(x=matmulx3, y=matmuly3)
            matmulx4 = static.data(
                name="matmulx4", shape=[6, 6, 2, 6], dtype='float32')
            matmuly4 = static.data(
                name="matmuly4", shape=[6, 6, 6, 6], dtype='float32')
            output2 = paddle.matmul(x=matmulx4, y=matmuly4)
            output_2 = layers.matmul(x=matmulx4, y=matmuly4)
        ops = program.global_block().ops
        vars = program.global_block().vars
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2' or op.type == 'matmul':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                impls = dist_op_impl_container.impls
                op_dist_attr = OperatorDistributedAttribute()
                X = op.input_arg_names[0]
                Y = op.input_arg_names[1]
                out = op.output_arg_names[0]
                if len(vars[X].shape) == 2 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1])
                    self.assertTrue(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, 1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, 1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 3 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1])
                    self.assertTrue(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [1, -1, -1])
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, 1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, 1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 4 and len(vars[Y].shape) == 4:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, -1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1, -1])
                    self.assertTrue(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [0, -1, -1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [0, -1, -1, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, 0, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 0, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, -1, 1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 0, -1])
                    self.assertFalse(impls[2].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))

    def test_matmulv2_matmul_1_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        with static.program_guard(program,
                                  start_program), utils.unique_name.guard():
            matmulx3 = static.data(
                name="matmulx3", shape=[6, 2, 6], dtype='float32')
            matmuly3 = static.data(
                name="matmuly3", shape=[6, 6], dtype='float32')
            output1 = paddle.matmul(x=matmulx3, y=matmuly3)
            output_1 = layers.matmul(x=matmulx3, y=matmuly3)
            matmulx4 = static.data(
                name="matmulx4", shape=[6, 6, 6, 6], dtype='float32')
            matmuly4 = static.data(
                name="matmuly4", shape=[6, 6, 6, 6], dtype='float32')
            output2 = paddle.matmul(x=matmulx4, y=matmuly4)
            output_2 = layers.matmul(x=matmulx4, y=matmuly4)
        ops = program.global_block().ops
        vars = program.global_block().vars
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2' or op.type == 'matmul':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                impls = dist_op_impl_container.impls
                op_dist_attr = OperatorDistributedAttribute()
                X = op.input_arg_names[0]
                Y = op.input_arg_names[1]
                out = op.output_arg_names[0]
                if len(vars[X].shape) == 2 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1])
                    op_dist_attr.set_input_dims_mapping(Y, [1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1])
                    dist_op = DistributedOperator(op, op_dist_attr)
                    op_dist_attr.set_output_dims_mapping(out, [1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 3 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, 1])
                    op_dist_attr.set_input_dims_mapping(Y, [1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1])
                    self.assertTrue(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [1, -1, 1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(out, [-1, -1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, 0, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 4 and len(vars[Y].shape) == 4:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1, 1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, 1, -1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1, -1])
                    self.assertTrue(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [0, -1, -1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [0, -1, -1, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, 0, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 0, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, -1, 1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 0, -1])
                    self.assertFalse(impls[1].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))

    def test_matmulv2_matmul_0_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        with static.program_guard(program,
                                  start_program), utils.unique_name.guard():
            matmulx3 = static.data(
                name="matmulx3", shape=[6, 2, 6], dtype='float32')
            matmuly3 = static.data(
                name="matmuly3", shape=[6, 6], dtype='float32')
            output1 = paddle.matmul(x=matmulx3, y=matmuly3)
            output_1 = layers.matmul(x=matmulx3, y=matmuly3)
            matmulx4 = static.data(
                name="matmulx4", shape=[6, 6, 2, 6], dtype='float32')
            matmuly4 = static.data(
                name="matmuly4", shape=[6, 6, 6, 6], dtype='float32')
            output2 = paddle.matmul(x=matmulx4, y=matmuly4)
            output_2 = layers.matmul(x=matmulx4, y=matmuly4)
        ops = program.global_block().ops
        vars = program.global_block().vars
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2' or op.type == 'matmul':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                impls = dist_op_impl_container.impls
                op_dist_attr = OperatorDistributedAttribute()
                X = op.input_arg_names[0]
                Y = op.input_arg_names[1]
                out = op.output_arg_names[0]
                if len(vars[X].shape) == 2 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, 1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, 1])
                    self.assertTrue(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [0, 0])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [0, -1])
                    op_dist_attr.set_output_dims_mapping(out, [1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 3 and len(vars[Y].shape) == 2:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, 1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 1])
                    self.assertTrue(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 0, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [1, -1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                if len(vars[X].shape) == 4 and len(vars[Y].shape) == 4:
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, -1, -1])
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, -1, 1])
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, -1, 1])
                    self.assertTrue(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [0, -1, -1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, 1, -1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(X, [-1, -1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [0, -1, -1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, 1, 1, 1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, -1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_output_dims_mapping(out, [-1, -1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))
                    op_dist_attr.set_input_dims_mapping(Y, [-1, -1, 1, -1])
                    self.assertFalse(impls[0].is_auto_compatible(
                        DistributedOperator(op, op_dist_attr)))


if __name__ == "__main__":
    unittest.main()
