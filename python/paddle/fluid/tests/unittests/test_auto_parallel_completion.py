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

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.tensor as tensor
import paddle.distributed.auto_parallel as auto
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list


paddle.enable_static()


def compare_program(src_prog, dst_prog):
    """Compare program to check whether they are same."""

    if src_prog.num_blocks != dst_prog.num_blocks:
        print(
            "Block number of src_program {} is not equal to that of dst_program {}."
            .format(src_prog.num_blocks, dst_prog.num_blocks))
        return False

    for src_block, dst_block in zip(src_prog.blocks, dst_prog.blocks):
        # compare vars from src_block and dst_block
        if len(src_block.vars) != len(dst_block.vars):
            print(
                "The number of variables in src_block {} is not equal to that in dst_block {}."
                .format(src_block.idx, dst_block.idx))
            return False
        for src_var_name, src_var_value in src_block.vars.items():
            dst_var_value = dst_block.vars.get(src_var_name)
            if dst_var_value is None:
                print(
                    "The variable {} from src_block doesn't exist in dst_block.".
                    format(src_var_name))
                return False
            if src_var_value.to_string(True, True) != dst_var_value.to_string(
                    True, True):
                print(
                    "The variable {} of src_block is not equal to variable {} of dst_block."
                    .format(src_var_name, src_var_name))
                return False

        # compare ops from src_block and dst_block
        if len(src_block.ops) != len(dst_block.ops):
            print(
                "The number of operators in src_block {} is not equal to that in dst_block {}."
                .format(src_block.idx, dst_block.idx))
        for src_op, dst_op in zip(src_block.ops, dst_block.ops):
            if src_op.type != dst_op.type:
                print(
                    "The operator's type {} of src_block is not equal to the operator'type {} of dst_block."
                    .format(src_op.type, dst_op.type))
            src_op_callstack = src_op.attr("op_callstack")
            dst_op_callstack = dst_op.attr("op_callstack")
            # print(src_op_callstack, dst_op_callstack)
            src_op._remove_attr("op_callstack")
            dst_op._remove_attr("op_callstack")
            if src_op.to_string(True) != dst_op.to_string(True):
                print(
                    "The operator {}'s content of src_block is not equal to the operator {}'s content of dst_block."
                    .format(src_op.type, dst_op.type))
                # print(src_op.to_string(True), dst_op.to_string(True))
                return False
            else:
                src_op._set_attr("op_callstack", src_op_callstack)
                dst_op._set_attr("op_callstack", dst_op_callstack)

        return True


class TestMLPAutoCompletion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.sequence_len = 128
        self.hidden_size = 512 
        self.dropout_ratio = 0.1
        self.initializer_range = 0.02

        self.prog = static.Program()
        with static.program_guard(self.prog), utils.unique_name.guard():
            intermediate_size = 4 * self.hidden_size
            d_model = self.hidden_size
            dim_feedforward = intermediate_size
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=self.initializer_range))
            bias_attr = None
            weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
            bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

            self.linear0 = nn.Linear(
                d_model,
                dim_feedforward,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.linear1 = nn.Linear(
                dim_feedforward,
                d_model,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout = nn.Dropout(self.dropout_ratio, mode="upscale_in_train")

    def test_mlp_dp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim() == 1, "The number dimension of process mesh must to be 1"
        with static.program_guard(self.prog), utils.unique_name.guard():
            input = static.data(
                name="input", shape=[self.batch_size, self.sequence_len, self.hidden_size], dtype='float32')
            out0 = self.norm(input)
            out1 = self.linear0(out0)
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)
            out4 = self.dropout(out3)
            auto.shard_tensor(out4, proc_mesh, dims_mapping=[0, -1, -1])
        print(self.prog)
        complete_prog = auto.complete_annotation(self.prog)
        print(complete_prog)

    def test_mlp_mp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim() == 1, "The number dimension of process mesh must to be 1"
        with static.program_guard(self.prog), utils.unique_name.guard():
            input = static.data(
                name="input", shape=[self.batch_size, self.sequence_len, self.hidden_size], dtype='float32')
            out0 = self.norm(input)
            out1 = self.linear0(out0)
            auto.shard_tensor(self.linear0.weight, proc_mesh, dims_mapping=[-1, 1])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)
            auto.shard_tensor(self.linear1.weight, proc_mesh, dims_mapping=[1, -1])
            out4 = self.dropout(out3)
        print(self.prog)
        complete_prog = auto.complete_annotation(self.prog)
        print(complete_prog)

    def test_mlp_dp_mp(self):
        proc_mesh = auto.ProcessMesh(
            shape=[2, 4], process_group=[0, 1, 2, 3, 4, 5, 6, 7])
        assert proc_mesh.get_ndim() == 2, "The number dimension of process mesh must to be 2"
        with static.program_guard(self.prog), utils.unique_name.guard():
            input = static.data(
                name="input", shape=[self.batch_size, self.sequence_len, self.hidden_size], dtype='float32')
            out0 = self.norm(input)
            auto.shard_tensor(out0, proc_mesh, dims_mapping=[0, -1, -1])
            out1 = self.linear0(out0)
            auto.shard_tensor(self.linear0.weight, proc_mesh, dims_mapping=[-1, 1])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)
            auto.shard_tensor(self.linear1.weight, proc_mesh, dims_mapping=[1, -1])
            out4 = self.dropout(out3)
        print(self.prog)
        complete_prog = auto.complete_annotation(self.prog)
        print(complete_prog)


if __name__ == "__main__":
    unittest.main()
