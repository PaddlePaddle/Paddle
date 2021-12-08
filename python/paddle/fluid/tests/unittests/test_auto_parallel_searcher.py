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

# import os
# import copy
# import json
import unittest

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
# from paddle.distributed import fleet
import paddle.distributed.auto_parallel as auto
# from paddle.distributed.auto_parallel.cluster import Cluster
# from paddle.distributed.auto_parallel.utils import SerialProgramInfo
# from paddle.distributed.auto_parallel.searcher import Checker, Enumerater
from paddle.distributed.auto_parallel.dist_context import DistributedContext
# from paddle.distributed.auto_parallel.utils import get_all_distributed_main_program
from paddle.distributed.auto_parallel.dist_attribute import TensorDistributedAttribute
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
from paddle.distributed.auto_parallel.utils import update_op_dims_mapping_by_default_dist_impl
from paddle.distributed.auto_parallel.utils import update_op_dims_mapping_by_elementwise_like_dist_impl

paddle.enable_static()


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
        out = paddle.unsqueeze(out, axis=0)
        out = paddle.reshape(out, [4, 1024])
        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')
        loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)

        predict = mlp(input)
        error_cost = loss_func(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def set_default_dist_attr(program, dist_context, process_mesh):
    ops = program.global_block().ops
    vars = program.global_block().vars
    for op in ops:
        op_dist_attr = OperatorDistributedAttribute()
        op_dist_attr.process_mesh = process_mesh
        for var_name in op.input_arg_names:
            tensor_dist_attr = TensorDistributedAttribute()
            tensor_dist_attr.process_mesh = process_mesh
            tensor_dist_attr.dims_mapping = [-1 for i in vars[var_name].shape]
            dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                          tensor_dist_attr)
            op_dist_attr.set_input_dims_mapping(var_name,
                                                tensor_dist_attr.dims_mapping)

        for var_name in op.output_arg_names:
            tensor_dist_attr = TensorDistributedAttribute()
            tensor_dist_attr.process_mesh = process_mesh
            tensor_dist_attr.dims_mapping = [-1 for i in vars[var_name].shape]
            dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                          tensor_dist_attr)
            op_dist_attr.set_output_dims_mapping(var_name,
                                                 tensor_dist_attr.dims_mapping)
        dist_context.set_op_dist_attr_for_program(op, op_dist_attr)

    dist_context.add_process_mesh(process_mesh)


class TestMLPSearcher(unittest.TestCase):
    def test_update(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        _, train_program, startup_program = mlp_forward(train_program,
                                                        startup_program)
        global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
        dist_context = DistributedContext()
        set_default_dist_attr(train_program, dist_context, global_process_mesh)
        ops = train_program.global_block().ops
        vars = train_program.global_block().vars
        from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container
        from paddle.distributed.auto_parallel.completion import is_elementwise_like_op
        from paddle.distributed.auto_parallel.dist_op import DistributedOperator

        for op in ops:
            dist_op_impl_container = get_distributed_operator_impl_container(
                op.type)
            if dist_op_impl_container is None:
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                dist_op = DistributedOperator(op, op_dist_attr)
                if is_elementwise_like_op(op.type):
                    changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                        dist_op)
                    self.assertFalse(changed)

                    dist_op.dist_attr.set_output_dims_mapping(
                        op.output_arg_names[0], [0] + [
                            -1
                            for i in range(
                                1, len(vars[op.output_arg_names[0]].shape))
                        ])
                    try:
                        changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                            dist_op)
                    except:
                        continue
                    self.assertTrue(changed)
                else:
                    changed = update_op_dims_mapping_by_default_dist_impl(
                        dist_op)
                    self.assertFalse(changed)

                    dist_op.dist_attr.set_output_dims_mapping(
                        op.output_arg_names[0], [0] + [
                            -1
                            for i in range(
                                1, len(vars[op.output_arg_names[0]].shape))
                        ])
                    try:
                        changed = update_op_dims_mapping_by_default_dist_impl(
                            dist_op)
                    except:
                        continue
                    self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
