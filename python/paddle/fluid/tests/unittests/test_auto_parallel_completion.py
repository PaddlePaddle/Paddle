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
import paddle.distributed.auto_parallel as auto

paddle.enable_static()


def build_partial_annotated_mlp_dp(batch_size, hidden_size, proc_mesh):
    prog = static.Program()
    assert proc_mesh.get_ndim(
    ) == 1, "The number dimension of process mesh must to be 1"
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        auto.shard_tensor(input, proc_mesh, dims_mapping=[0, -1])

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        out1 = F.linear(input, weight1, bias=bias1)
        # auto.shard_tensor(out1, proc_mesh, dims_mapping=[0, -1])

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # auto.shard_tensor(bias2, proc_mesh, dims_mapping=[-1])
        out2 = F.linear(F.gelu(out1), weight2, bias=bias2)
    return prog


def build_complete_annotated_mlp_dp(batch_size, hidden_size, proc_mesh):
    prog = static.Program()
    assert proc_mesh.ndim == 1, "The number dimension of process mesh must to be 1"
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        # shard_tensor(input, proc_mesh, dims_mapping=[0, -1])

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight1, proc_mesh, dims_mapping=[-1, -1])
        # shard_tensor(bias1, proc_mesh, dims_mapping=[-1])
        # shard_linear = shard_op(F.linear,  proc_mesh, {0: [0, -1], 1: [-1, -1], 2: [-1]}, {0: [0]})
        out1 = shard_linear(x=input, weight=weight1, bias=bias1)

        # gelu = shard_op(F.gelu, proc_mesh, {0: [0, -1]}, {0: [0, -1]})
        intermediate = F.gelu(out1)
        # shard_tensor(intermediate, proc_mesh, dims_mapping=[0, -1])

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight2, proc_mesh, dims_mapping=[0, -1])
        # shard_tensor(bias2, proc_mesh, dims_mapping=[-1])
        # linear = shard_op(F.linear, proc_mesh, {0: [0, -1], 1: [-1, 1], 2: [-1]}, {0: [0, -1]})
        out3 = F.linear(intermediate, weight2, bias=bias2)
        # shard_tensor(out3, proc_mesh, dims_mapping=[0, -1])

    return prog


def build_partial_annotated_mlp_mp(batch_size, hidden_size, proc_mesh):
    assert proc_mesh.ndim == 1, "The number dimension of process mesh must to be 1"
    prog = static.Program()
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight1, proc_mesh, dims_mapping=[-1, 0])
        out1 = F.linear(input, weight1, bias=bias1)

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight2, proc_mesh, dims_mapping=[0, -1])
        out2 = F.linear(F.gelu(out1), weight2, bias=bias2)
    return prog


def build_complete_annotated_mlp_mp(batch_size, hidden_size, proc_mesh):
    assert proc_mesh.ndim == 1, "The number dimension of process mesh must to be 1"
    prog = static.Program()
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[1, hidden_size], dtype='float32')
        # shard_tensor(input, proc_mesh, dims_mapping=[-1, -1])

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight1, proc_mesh, dims_mapping=[-1, 0])
        # shard_tensor(bias1, proc_mesh, dims_mapping=[-1, 0])
        # linear = shard_op(F.linear, proc_mesh, {0: [-1, -1], 1: [-1, 0], 2: [0]}, {0: [-1, 0]})
        # out1 = linear(x=input, weight=weight1, bias=bias1)
        # out1 = shard_op(F.linear(input, weight1, bias=bias1), 
        #                 proc_mesh, {0: [-1, -1], 1: [-1, 0], 2: [0]}, {0: [-1, 0]})

        out1 = F.linear(x=input, weight=weight1, bias=bias1)

        # gelu = shard_op(F.gelu, proc_mesh, {0: [-1, 0]}, {0: [-1, 0]})
        intermediate = F.gelu(out1)
        # shard_tensor(intermediate, proc_mesh, dims_mapping=[-1, 0])

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight2, proc_mesh, dims_mapping=[0, -1])
        # shard_tensor(bias2, proc_mesh, dims_mapping=[-1])
        # linear = shard_op(F.linear, proc_mesh, {0: [0, -1], 1: [-1, 0], 2: [-1]}, {0: [-1, -1]})
        out3 = F.linear(intermediate, weight2, bias=bias2)
        # shard_tensor(out3, proc_mesh, dims_mapping=[-1, -1])

    return prog


def build_partial_annotated_mlp_dp_mp(batch_size, hidden_size, proc_mesh):
    assert proc_mesh.ndim == 2, "The number dimension of process mesh must to be 2"
    prog = static.Program()
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        # shard_tensor(input, proc_mesh, dims_mapping=[0, -1])

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight1, proc_mesh, dims_mapping=[-1, 1])
        out1 = F.linear(input, weight1, bias=bias1)

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight2, proc_mesh, dims_mapping=[1, -1])
        out2 = F.linear(F.gelu(out1), weight2, bias=bias2)
    return prog


def build_complete_annotated_mlp_dp_mp(batch_size, hidden_size, proc_mesh):
    assert proc_mesh.ndim == 1, "The number dimension of process mesh must to be 1"
    prog = static.Program()
    with static.program_guard(prog), utils.unique_name.guard():
        input = static.data(
            name="input", shape=[1, hidden_size], dtype='float32')
        # shard_tensor(input, proc_mesh, dims_mapping=[0, -1])

        weight1 = static.create_parameter(
            shape=[hidden_size, 4 * hidden_size],
            dtype='float32',
            is_bias=False)
        bias1 = static.create_parameter(
            shape=[4 * hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight1, proc_mesh, dims_mapping=[-1, 1])
        # shard_tensor(bias1, proc_mesh, dims_mapping=[1])
        # out1 = shard_op(F.linear(input, weight1, bias=bias1), 
        #                 proc_mesh, {0: [0, -1], 1: [-1, 1], 2: [1]}, {0: [0, 1]})

        out1 = F.linear(x=input, weight=weight1, bias=bias1)

        # gelu = shard_op(F.gelu, proc_mesh, {0: [0, 1]}, {0: [0, 1]})
        intermediate = F.gelu(out1)
        # shard_tensor(intermediate, proc_mesh, dims_mapping=[0, 1])

        weight2 = static.create_parameter(
            shape=[4 * hidden_size, hidden_size],
            dtype='float32',
            is_bias=False)
        bias2 = static.create_parameter(
            shape=[hidden_size], dtype='float32', is_bias=True)
        # shard_tensor(weight2, proc_mesh, dims_mapping=[1, -1])
        # shard_tensor(bias2, proc_mesh, dims_mapping=[-1])
        # linear = shard_op(F.linear, proc_mesh, {0: [0, 1], 1: [1, -1], 2: [-1]}, {0: [0, -1]})
        out3 = F.linear(intermediate, weight2, bias=bias2)
        # shard_tensor(out3, proc_mesh, dims_mapping=[0, -1])

    return prog


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


class TestAutoCompletion(unittest.TestCase):
    def test_auto_completion_mlp_dp(self):
        process_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        batch_size = 4
        hidden_size = 1024
        prog1 = build_partial_annotated_mlp_dp(batch_size, hidden_size,
                                               process_mesh)
        print(prog1)
        complete_prog1 = auto.complete_annotation(prog1)
        print(complete_prog1)
        # prog1 = build_partial_annotated_mlp_program(input, 1024, proc_mesh) 
        # prog2 = build_complete_annotated_mlp_dp(batch_size, hidden_size, process_mesh) 
        # print(prog2.to_string(True, False))
        # print(prog2._to_readable_code())
        # result = compare_program(prog1, prog2)
        # self.assertTrue(result, "Two programs are not same.")


"""     def test_auto_completion_mlp_mp(self):
        process_mesh = None 
        batch_size = 4 
        hidden_size = 1024
        prog1 = build_partial_annotated_mlp_mp(batch_size, hidden_size, process_mesh) 
        prog2 = build_complete_annotated_mlp_mp(batch_size, hidden_size, process_mesh) 
        result = compare_program(prog1, prog2)
        self.assertTrue(result, "Two programs are not same.")

    def test_auto_completion_mlp_dp_mp(self):
        process_mesh = None 
        batch_size = 4 
        hidden_size = 1024
        prog1 = build_partial_annotated_mlp_dp_mp(batch_size, hidden_size, process_mesh) 
        prog2 = build_complete_annotated_mlp_dp_mp(batch_size, hidden_size, process_mesh) 
        result = compare_program(prog1, prog2)
        self.assertTrue(result, "Two programs are not same.") """

if __name__ == "__main__":
    unittest.main()
