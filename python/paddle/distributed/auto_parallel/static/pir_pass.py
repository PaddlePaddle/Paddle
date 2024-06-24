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
from paddle.autograd.backward_utils import ValueDict

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
        assert len(op.results()) == len(
            op.dist_attr.results()
        ), f"The number of results and the number of op_dist_attr's results are not equal in op: {op}"
        # deal with inplace value
        for out_idx, in_idx in paddle.core.pir.get_op_inplace_info(op).items():
            operand = op.operand(in_idx)
            operand_attr = op.dist_attr.operand(in_idx)
            prev_var = operand.source()
            if not prev_var.is_dist() or operand_attr == prev_var.dist_attr():
                continue
            assert (
                not prev_var.is_combine()
            ), f"The current partition pass not support inplace value of {op} is tensor list."
            operand_attr = operand_attr.as_tensor_dist_attr()
            # reshard input
            paddle.pir.set_insertion_point(op)
            reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
            operand.set_source(reshard_var)

            result = op.result(out_idx)
            result_attr = op.dist_attr.result(out_idx).as_tensor_dist_attr()
            assert (
                operand_attr == result_attr
            ), f"For inplace value, The operend dist attr should be equal to result dist attr , please check your infer_spmd func of {op}"

            # reshard output
            paddle.pir.set_insertion_point_after(op)
            old_dist_attr = result.dist_attr()
            result.update_dist_attr(result_attr)

            # reshard output to assign out input
            reshard_var_1 = paddle._C_ops.reshard_v2(
                result, prev_var.dist_attr()
            )
            paddle.assign(reshard_var_1, prev_var)

            if old_dist_attr == result.dist_attr():
                continue
            reshard_var_2 = reshard_var_1
            if old_dist_attr != reshard_var_1.dist_attr():
                reshard_var_2 = paddle._C_ops.reshard_v2(result, old_dist_attr)
            result.replace_all_uses_with(reshard_var_1)
            reshard_var_1.get_defining_op().operand(0).set_source(result)
            reshard_var_2.get_defining_op().operand(0).set_source(result)

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


def fold_reshard_pass(dist_program):
    del_ops = []
    value_dict = ValueDict()
    for op in dist_program.global_block().ops:
        if op.name() != 'dist_op.reshard':
            continue
        input = op.operand_source(0)
        result = op.result(0)
        if input.type() == result.type():
            result.replace_all_uses_with(input)
            del_ops.append(op)
            continue
        if input not in value_dict:
            value_dict[input] = [(result.type(), result)]
            continue
        no_find = True
        for type, val in value_dict[input]:
            if type == result.type():
                result.replace_all_uses_with(val)
                del_ops.append(op)
                no_find = False
                break
        if no_find:
            value_dict[input].append((result.type(), result))
    for op in del_ops:
        op.erase()


def apply_reshard_pass(dist_program):
    fold_reshard_pass(dist_program)
    for op in dist_program.global_block().ops:
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
            out_value.persistable = op.result(0).persistable
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

    for key, arg in dist_program.global_block().kwargs().items():
        if cur_rank not in arg.dist_attr().process_mesh.process_ids:
            dist_program.global_block().erase_kwarg(key)

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
comm_ops = ["pd_op.c_allreduce_sum", "pd_op.c_allgather"]


def remove_unuseful_comm_op_pass(program):
    for op in program.global_block().ops:
        if op.name() in comm_ops:
            ring_id = op.int_attr("ring_id")
            process_group = get_process_group(ring_id)
            if process_group.nranks == 1:
                op.result(0).replace_all_uses_with(op.operand_source(0))
                op.erase()
        if op.name() == "pd_op.share_data_":
            if op.operand_source(0).has_one_use():
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


def split_program_pass(main_program, last_fwd_op, last_bwd_op):
    ops = main_program.global_block().ops
    num_ops = len(ops)

    fwd_program = main_program.clone()
    bwd_program = main_program.clone()
    opt_program = main_program.clone()
    fwd_ops = fwd_program.global_block().ops
    bwd_ops = bwd_program.global_block().ops
    opt_ops = opt_program.global_block().ops
    opt_block = opt_program.global_block()
    bwd_block = bwd_program.global_block()

    region = "opt"
    for i in range(num_ops - 1, -1, -1):
        if ops[i] == last_bwd_op:
            region = "bwd"
        if ops[i] == last_fwd_op:
            region = "fwd"

        if (
            ops[i].name() == "pd_op.data"
            and "learning_rate" in ops[i].attrs()["name"]
        ):
            fwd_ops[i].erase()
            bwd_ops[i].erase()
            continue

        if region == "opt":
            fwd_ops[i].erase()
            bwd_ops[i].erase()
        elif region == "bwd":
            fwd_ops[i].erase()
            # in optimize program, both forward and backward ops should be removed
            for idx in range(opt_ops[i].num_results()):
                # if this op's output is used, create the persistable
                # var to be used in other programs.
                result_in_opt = opt_ops[i].result(idx)
                if result_in_opt.use_empty() is False:
                    name = (
                        "var_" + str(i) + "_" + ops[i].name() + "_" + str(idx)
                    )
                    paddle.pir.set_insertion_point_after(bwd_ops[i])
                    paddle._C_ops.set_persistable_value(
                        bwd_ops[i].result(idx), name
                    )
                    bwd_ops[i].result(idx).persistable = True
                    new_result_var_in_opt = opt_block.add_kwarg(
                        name, result_in_opt.type()
                    )
                    new_result_var_in_opt.persistable = True
                    opt_ops[i].result(idx).replace_all_uses_with(
                        new_result_var_in_opt
                    )
            opt_ops[i].erase()
        else:
            # in backward program, only the forward ops should be removed
            for idx in range(opt_ops[i].num_results()):
                # if this op's output is used, create the persistable
                # var to be used in other programs.
                result_in_opt = opt_ops[i].result(idx)
                result_in_bwd = bwd_ops[i].result(idx)
                if (
                    result_in_opt.use_empty() is False
                    or result_in_bwd.use_empty() is False
                ):
                    if (
                        fwd_ops[i].name() == "pd_op.data"
                        or fwd_ops[i].name() == "builtin.parameter"
                    ):
                        name = fwd_ops[i].result(idx).name
                        fwd_ops[i].result(idx).persistable = True
                    else:
                        result_value = ops[i].result(idx)
                        used_ops = result_value.all_used_ops()
                        shadow_output_op_used = None
                        for used_op in used_ops:
                            if used_op.name() == "builtin.shadow_output":
                                shadow_output_op_used = used_op
                        if shadow_output_op_used is not None:
                            name = shadow_output_op_used.attrs()["output_name"]
                            fwd_ops[i].result(idx).persistable = True
                        else:
                            name = (
                                "var_"
                                + str(i)
                                + "_"
                                + ops[i].name()
                                + "_"
                                + str(idx)
                            )
                            paddle.pir.set_insertion_point_after(fwd_ops[i])
                            paddle._C_ops.set_persistable_value(
                                fwd_ops[i].result(idx), name
                            )
                            fwd_ops[i].result(idx).persistable = True
                if result_in_opt.use_empty() is False:
                    new_result_var_in_opt = opt_block.add_kwarg(
                        name, result_in_opt.type()
                    )
                    new_result_var_in_opt.persistable = True
                    opt_ops[i].result(idx).replace_all_uses_with(
                        new_result_var_in_opt
                    )
                if result_in_bwd.use_empty() is False:
                    new_result_var_in_bwd = bwd_block.add_kwarg(
                        name, result_in_bwd.type()
                    )
                    new_result_var_in_bwd.persistable = True
                    bwd_ops[i].result(idx).replace_all_uses_with(
                        new_result_var_in_bwd
                    )
            opt_ops[i].erase()
            bwd_ops[i].erase()

    return fwd_program, bwd_program, opt_program
