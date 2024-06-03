# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from .process_group import get_process_group
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs

register_reshard_funcs()

partition_skip_op_list = ["builtin.combine", "builtin.split"]


def reshard_single_value(op, operand, attr):
    prev_var = operand.source()
    if prev_var.is_dist() and prev_var.dist_attr() != attr:
        operand_attr = attr.as_tensor_dist_attr()
        paddle.pir.set_insertion_point(op)
        # fold reshard
        if prev_var.get_defining_op().name() == 'dist_op.reshard':
            prev_reshard = prev_var.get_defining_op()
            prev_var = prev_reshard.operand_source(0)
            if prev_var.dist_attr() == operand_attr:
                return prev_var
            reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
            return reshard_var
        # insert reshard
        reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
        return reshard_var
    return prev_var


def reshard_combine_value(op, operand, attr):
    prev_var = operand.source()
    assert (
        prev_var.get_defining_op().name() == 'builtin.combine'
    ), "TensorList must be defined by builtin.combine op."
    combine_op = prev_var.get_defining_op()
    array_attr = attr.as_array_attr()
    assert len(combine_op.operands()) == len(
        array_attr
    ), "The number of combine op operands and the number of dist array_attr are not equal in op"
    reshard_vars = []
    for inner_operand, inner_attr in zip(combine_op.operands(), array_attr):
        reshard_vars.append(reshard_single_value(op, inner_operand, inner_attr))
    paddle.pir.set_insertion_point(op)
    return paddle._C_ops.builtin_combine(reshard_vars)


def apply_partition_pass(program):
    for op in program.global_block().ops:
        if op.name() in partition_skip_op_list:
            continue
        assert len(op.operands()) == len(
            op.dist_attr.operands()
        ), f"The number of operands and the number of op_dist_attr's operands are not equal in op: {op}"

        for operand, attr in zip(op.operands(), op.dist_attr.operands()):
            prev_var = operand.source()
            if prev_var.is_combine():
                operand.set_source(reshard_combine_value(op, operand, attr))
            else:
                operand.set_source(reshard_single_value(op, operand, attr))
            prev_op = prev_var.get_defining_op()
            if prev_op and prev_op.num_results() == 1 and prev_var.use_empty():
                prev_op.erase()

        for var, attr in zip(op.results(), op.dist_attr.results()):
            if var.initialized() and var.is_dist() and var.dist_attr() != attr:
                paddle.pir.set_insertion_point_after(op)
                old_dist_attr = var.dist_attr()
                var.update_dist_attr(attr.as_tensor_dist_attr())
                # insert reshard
                reshard_var = paddle._C_ops.reshard_v2(var, old_dist_attr)
                var.replace_all_uses_with(reshard_var)
                reshard_var.get_defining_op().operand(0).set_source(var)


def apply_reshard_pass(program):
    for op in program.global_block().ops:
        if op.name() == 'dist_op.reshard':
            var = op.operand_source(0)
            op_dist_attr = op.dist_attr
            src_dist_attr = op_dist_attr.operand(0).as_tensor_dist_attr()
            dst_dist_attr = op_dist_attr.result(0).as_tensor_dist_attr()
            assert (
                not var.initialized() or var.dist_attr() == src_dist_attr
            ), f"The dist_attr of reshard op's input and operand should be equal, but got {var.dist_attr()} and {src_dist_attr}"

            if src_dist_attr == dst_dist_attr:
                op.result(0).replace_all_uses_with(var)
                op.erase()
                continue
            reshard_func = choose_reshard_func(src_dist_attr, dst_dist_attr)
            assert (
                reshard_func is not None
            ), f'There is no reshard function that matches src_dist_attr: {src_dist_attr} and dst_dist_attr: {dst_dist_attr}'
            paddle.pir.set_insertion_point_after(op)
            out_value = reshard_func.reshard(
                src_dist_attr,
                dst_dist_attr,
                op.operand_source(0),
                op.result(0).type(),
            )
            if out_value is not None:
                op.result(0).replace_all_uses_with(out_value)
            if op.result(0).use_empty():
                op.erase()


# pruning op and value not belong to cur rank
def remove_other_rank_op_pass(dist_program):
    cur_rank = paddle.distributed.get_rank()
    for op in dist_program.global_block().ops[::-1]:
        if op.name() in partition_skip_op_list:
            can_delete = True
            for val in op.results():
                if not val.use_empty():
                    can_delete = False
            if can_delete:
                op.erase()
            continue
        if cur_rank not in op.dist_attr.process_mesh.process_ids:
            op.erase()
        elif op.name() == "dist_op.reshard":
            assert op.result(
                0
            ).use_empty(), f'There should not have useful dist.reshard op in remove_other_rank_op_pass. but find : {op}'
            op.erase()

    # merge pd.data ops for
    lr_ops = []
    for op in dist_program.global_block().ops[::-1]:
        if op.name() == 'pd_op.data' and "learning_rate" in op.attrs()["name"]:
            lr_ops.append(op)

    if len(lr_ops) > 1:
        lr_value = lr_ops[0].result(0)
        for op in lr_ops[1:]:
            lr = op.result(0)
            lr.replace_all_uses_with(lr_value)
            op.erase()


# Note: this is the pass in the dense program
comm_ops = ["pd_op.c_allreduce_sum_", "pd_op.c_allgather"]


def remove_unuseful_comm_op_pass(program):
    for op in program.global_block().ops:
        if op.name() in comm_ops:
            ring_id = op.int_attr("ring_id")
            process_group = get_process_group(ring_id)
            if process_group.nranks == 1:
                op.result(0).replace_all_uses_with(op.operand_source(0))
                op.erase()


# In sequence_parallel, we need to transpose hidden_states
# from [bs, seq, hidden] to [seq, bs, hidden] to perform
# split and allgather at dim 0.
# The transpose may lead to about 3% performance
# in llama-70B model (tp4pp8).
# We found that, when bs=1, which is the common case in llm
# training, the transpose is equal to reshape.
# So, this pass is to haddle the specific case.
def eliminate_transpose_by_reshape(program):
    for op in program.global_block().ops:
        if (
            op.name() == 'pd_op.transpose'
            or op.name() == 'pd_op.transpose_grad'
        ):
            var = op.operand(0).source()
            rank = len(var.shape)
            perm = op.attrs()['perm']
            perm = [p + rank if p < 0 else p for p in perm]
            # only support transpose dim 0 and dim 1
            expected_perm = [1, 0] + [i + 2 for i in range(rank - 2)]
            if perm == expected_perm and (
                var.shape[0] == 1 or var.shape[1] == 1
            ):
                paddle.pir.set_insertion_point(op)
                transpose_var = op.result(0)
                reshape_var = paddle._C_ops.reshape(var, transpose_var.shape)
                transpose_var.replace_all_uses_with(reshape_var)
                op.erase()
    return program
