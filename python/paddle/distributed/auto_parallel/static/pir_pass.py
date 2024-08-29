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
from paddle.distributed.passes.pass_base import PassContext, new_pass
from paddle.pir import get_current_insertion_point

from ...passes.pass_utils import auto_complete_op_role
from .process_group import get_process_group
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs
from .utils import get_pp_stage_by_pp_degree

register_reshard_funcs()

partition_skip_op_list = [
    "builtin.combine",
    "builtin.split",
    "pd_op.pylayer",
    "cf.yield",
    "cf.tuple_push",
    "cf.tuple_pop",
    "cf.stack_create",
]
amp_ops = ["pd_op.check_finite_and_unscale_", "pd_op.update_loss_scaling_"]


def reshard_single_value(program, op, operand, attr):
    prev_var = operand.source()
    if prev_var.is_dist() and prev_var.dist_attr() != attr:
        operand_attr = attr.as_tensor_dist_attr()
        paddle.pir.set_insertion_point(op)
        with auto_complete_op_role(
            program, op.op_role, get_current_insertion_point()
        ):
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


def reshard_combine_value(program, op, operand, attr):
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
        reshard_vars.append(
            reshard_single_value(program, op, inner_operand, inner_attr)
        )

    paddle.pir.set_insertion_point(op)
    with auto_complete_op_role(
        program, op.op_role, get_current_insertion_point()
    ):
        combine_value = paddle._C_ops.builtin_combine(reshard_vars)
    return combine_value


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
            ref_op_role = op.op_role

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
            with auto_complete_op_role(
                program, ref_op_role, get_current_insertion_point()
            ):
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

            with auto_complete_op_role(
                program, ref_op_role, get_current_insertion_point()
            ):
                # reshard output to assign out input
                reshard_var_1 = paddle._C_ops.reshard_v2(
                    result, prev_var.dist_attr()
                )
                paddle.assign(reshard_var_1, prev_var)

            if old_dist_attr == result.dist_attr():
                continue

            if ref_op_role is not None:
                paddle.pir.set_insertion_point_after(op)

            reshard_var_2 = reshard_var_1
            if old_dist_attr != reshard_var_1.dist_attr():
                with auto_complete_op_role(
                    program, ref_op_role, get_current_insertion_point()
                ):
                    reshard_var_2 = paddle._C_ops.reshard_v2(
                        result, old_dist_attr
                    )

            result.replace_all_uses_with(reshard_var_1)
            reshard_var_1.get_defining_op().operand(0).set_source(result)
            reshard_var_2.get_defining_op().operand(0).set_source(result)

        for operand, attr in zip(op.operands(), op.dist_attr.operands()):
            prev_var = operand.source()
            if prev_var.is_combine():
                operand.set_source(
                    reshard_combine_value(program, op, operand, attr)
                )
            else:
                operand.set_source(
                    reshard_single_value(program, op, operand, attr)
                )
            prev_op = prev_var.get_defining_op()
            if prev_op and prev_op.num_results() == 1 and prev_var.use_empty():
                prev_op.erase()

        for var, attr in zip(op.results(), op.dist_attr.results()):
            if var.initialized() and var.is_dist() and var.dist_attr() != attr:
                paddle.pir.set_insertion_point_after(op)
                old_dist_attr = var.dist_attr()
                var.update_dist_attr(attr.as_tensor_dist_attr())

                # insert reshard
                with auto_complete_op_role(
                    program, op.op_role, get_current_insertion_point()
                ):
                    reshard_var = paddle._C_ops.reshard_v2(var, old_dist_attr)
                    var.replace_all_uses_with(reshard_var)
                    reshard_var.get_defining_op().operand(0).set_source(var)
                    var.get_defining_op().set_bool_attr(
                        "replace_all_uses_with_reshard_var", True
                    )


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


def apply_reshard_pass(dist_program, params_grads=[]):
    fold_reshard_pass(dist_program)

    # {grad.id: grad}
    sharded_grad = {}
    grad_ids = [grad.id for _, grad in params_grads if grad is not None]

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
            ), f'There is no reshard function that matches src_dist_attr: {src_dist_attr} and dst_dist_attr: {dst_dist_attr}, {var.get_defining_op()}'

            paddle.pir.set_insertion_point(op)
            ref_op_role = op.op_role

            with auto_complete_op_role(
                dist_program, ref_op_role, get_current_insertion_point()
            ):
                out_value = reshard_func.reshard(
                    src_dist_attr,
                    dst_dist_attr,
                    op.operand_source(0),
                    op.result(0).type(),
                )

            if out_value is not None:
                op.result(0).replace_all_uses_with(out_value)
                if var.id in grad_ids:
                    if var.get_defining_op().has_attr(
                        "replace_all_uses_with_reshard_var"
                    ):
                        sharded_grad[var.id] = out_value

            if op.result(0).use_empty():
                op.erase()

            if out_value is not None and var.use_empty():
                if var.id in grad_ids:
                    sharded_grad[var.id] = out_value

    # update params_grads with sharded grad
    for idx, (param, grad) in enumerate(params_grads):
        if grad is None:
            continue

        if grad.id in sharded_grad:
            params_grads[idx] = (param, sharded_grad[grad.id])


def _remove_other_rank_params_grads(dist_params_grads):
    cur_rank = paddle.distributed.get_rank()
    need_remove_idx = []
    for idx, (_, grad) in enumerate(dist_params_grads):
        if grad is None:
            continue
        if cur_rank not in grad.dist_attr().process_mesh.process_ids:
            need_remove_idx.append(idx)
    for idx in need_remove_idx[::-1]:
        dist_params_grads.pop(idx)


# pruning op and value not belong to cur rank
def remove_other_rank_op_pass(dist_program, dist_params_grads):
    cur_rank = paddle.distributed.get_rank()

    _remove_other_rank_params_grads(dist_params_grads)
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


# Pruning value not belong to cur rank
# especially used for check_finite_and_unscale
# and update_loss_scaling op in amp
# For example, w0 on mesh0, w1 on mesh1, before pass, the ops is:
#  [w0_g, w1_g], is_finite = check_finite_and_scale([w0_g, w1_g], loss_scaling)
# after pass, on mesh0, the op is:
#  [w0_g], is_finite = check_finite_and_scale([w0_g], loss_scaling)
# Note that here we do not set the op_dist_attr, since it is not used
# afterwards.
def remove_other_rank_input_output_pass(dist_program):
    cur_rank = paddle.distributed.get_rank()
    for op in dist_program.global_block().ops[::-1]:
        if op.name() not in amp_ops:
            continue
        new_vars = []
        combine_op = op.operand_source(0).get_defining_op()
        for inner_operand in op.operand_source(0).get_defining_op().operands():
            if (
                cur_rank
                in inner_operand.source().dist_attr().process_mesh.process_ids
            ):
                new_vars.append(inner_operand.source())
                continue
        result = op.operand_source(0).get_defining_op().result(0)
        paddle.pir.set_insertion_point_after(combine_op)
        res = paddle._C_ops.builtin_combine(new_vars)
        result.replace_all_uses_with(res)
        combine_op.erase()
        # since it is inplace op, set type of output as the same as input
        op.result(0).set_type(res.type())


# Note: this is the pass in the dense program
comm_ops = [
    "pd_op.c_allreduce_sum",
    "pd_op.all_gather",
    "pd_op.c_allreduce_max",
    "pd_op.reduce_scatter",
]


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


def complete_op_role(main_program, op_role_scope: list):
    assert (
        len(op_role_scope) == 3 and len(op_role_scope[0]) == 2
    ), "op_role_scope should has the shape[3, 2]"
    forward_op_start = op_role_scope[0][0]
    forward_op_end = op_role_scope[0][1]

    backward_op_start = op_role_scope[1][0]
    backward_op_end = op_role_scope[1][1]

    opt_op_start = op_role_scope[2][0]
    opt_op_end = op_role_scope[2][1]

    global_op_idx = 0
    for blk in main_program.blocks:
        for op in blk.ops:
            if (
                global_op_idx >= forward_op_start
                and global_op_idx < forward_op_end
            ):
                op.op_role = 0
            elif (
                global_op_idx >= backward_op_start
                and global_op_idx < backward_op_end
            ):
                op.op_role = 1
            elif global_op_idx >= opt_op_start and global_op_idx < opt_op_end:
                op.op_role = 2
            else:
                pass
            global_op_idx += 1


def pipeline_pass(dense_main_program, dense_starup_program, pipeline_strategy):
    """
    Pipeline schedule pass for auto parallel. Enables the pipeline parallel scheduling
    strategies like FThenB, 1F1B, VPP, etc.
    """
    import os

    pass_name = pipeline_strategy.schedule_mode
    assert pass_name in [
        "FThenB",
        "1F1B",
    ], f"pipeline scheduler only support FThenB now, but receive {pass_name}"

    pass_attr = {}
    pass_attr["num_micro_batches"] = pipeline_strategy.accumulate_steps
    pass_attr["pp_degree"] = pipeline_strategy.pp_degree
    pass_attr["pp_stage"] = get_pp_stage_by_pp_degree(
        pipeline_strategy.pp_degree
    )

    if pass_name == "1F1B":
        # TODO(Ruibiao): Move FLAGS_1f1b_backward_forward_overlap and
        # FLAGS_mp_async_allreduce_in_backward to auto parallel Strategy
        # after these two optimizations are available.
        pass_attr["enable_backward_forward_overlap"] = int(
            os.environ.get("FLAGS_1f1b_backward_forward_overlap", 0)
        )

    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply(
        dense_main_program,
        dense_starup_program,
        pass_context,
    )
    plan = pass_context.get_attr("plan")
    return plan


# NOTE(zhangbo): Add Fused_FFN pass for dense program.
# 1. forward ffn pattern:
# (o1) = "pd_op.matmul" (in, w1)
# (o2) = "pd_op.matmul" (in, w2)
# (o3) = "pd_op.swiglu" (o1, o2)
# -->
# (fused_ffn_w) = concat_and_relocate_(w1, w2)
# (fused_ffn_o) = "pd_op.matmul" (in, fused_ffn_w)
# (o3) = "pd_op.swiglu"(fused_ffn_o)
#
# 2. backward ffn pattern:
# (o2_g, o1_g) = "pd_op.swiglu_grad" (o1, o2, o3_g)
# (in_g1, w2_g) = "pd_op.matmul_grad" (in, w2, o2_g)
# (in_g2, w1_g) = "pd_op.matmul_grad" (in, w1, o1_g)
# (in_g_combine) = "builtin.combine" (in_g1, in_g2)
# (in_g) = "pd_op.add_n" (in_g_combine)
# -->
# (fused_ffn_o_g, null) = "pd_op.swiglu_grad" (fused_ffn_o, null, o3_g)
# (in_g, fused_ffn_w_g) = "pd_op.matmul_grad" (in, fused_ffn_w, fused_ffn_o_g)
#
# 3. opt pattern:
# (w1_tmp, w2_tmp) = "pd_op.split" (fused_ffn_w, 2)
# (-) = "pd_op.assign_out_" (w1_tmp, w1)
# (-) = "pd_op.assign_out_" (w2_tmp, w2)
# (w2_g, w1_g) = "pd_op.split" (fused_ffn_w_g, 2)


def prepare_for_vjp(fwd_op):
    fwd_inputs = [[value] for value in fwd_op.operands_source()]
    fwd_outputs = [[value] for value in fwd_op.results()]
    stop_gradients = []
    for v in fwd_inputs:
        stop_gradients.append([v[0].stop_gradient])
    return fwd_inputs, fwd_outputs, stop_gradients


def fused_ffn_pass(dense_main_program):
    # (1) Search for source pattern of ffn
    all_ops = dense_main_program.global_block().ops
    src_pattern_list = []
    for i in range(len(all_ops) - 4):
        if (
            all_ops[i].name() == "pd_op.matmul"
            and all_ops[i + 1].name() == "pd_op.matmul"
            and all_ops[i + 2].name() == "pd_op.swiglu"
        ):
            src_pattern = {}
            res_pattern = {}
            # find fwd src_pattern
            src_pattern['mm1'] = all_ops[i]
            src_pattern['mm2'] = all_ops[i + 1]
            src_pattern['swiglu'] = all_ops[i + 2]

            # insert fwd dst_pattern
            paddle.pir.set_insertion_point_after(src_pattern['swiglu'])
            w_list = [
                src_pattern['mm1'].operand_source(1),
                src_pattern['mm2'].operand_source(1),
            ]
            _, fused_w = paddle._C_ops.concat_and_relocate_(w_list)
            res_pattern['concat_and_relocate_'] = fused_w.get_defining_op()

            fused_o = paddle.matmul(
                src_pattern['mm1'].operand_source(0),
                fused_w,
                transpose_x=False,
                transpose_y=False,
            )
            res_pattern['mm'] = fused_o.get_defining_op()
            res_pattern['mm'].copy_attrs_from(src_pattern['mm1'])

            out = paddle.incubate.nn.functional.swiglu(fused_o)
            res_pattern['swiglu'] = out.get_defining_op()
            res_pattern['swiglu'].copy_attrs_from(src_pattern['swiglu'])

            src_pattern['swiglu'].result(0).replace_all_uses_with(out)

            for op in dense_main_program.global_block().ops:
                if not op.has_attr("op_role"):
                    op.op_role = 0

            # solve bwd pattern
            for op in src_pattern['swiglu'].operand_source(0).all_used_ops():
                if op.name() == "pd_op.swiglu_grad":
                    src_pattern['swiglu_g'] = op
                    src_pattern['mm2_g'] = all_ops[all_ops.index(op) + 1]
                    src_pattern['mm1_g'] = all_ops[all_ops.index(op) + 2]
                    src_pattern['combine'] = all_ops[all_ops.index(op) + 3]
                    src_pattern['add_n'] = all_ops[all_ops.index(op) + 4]

                    # insert bwd dst_pattern
                    paddle.pir.set_insertion_point_after(src_pattern['add_n'])
                    fwd_inputs, fwd_outputs, stop_gradients = prepare_for_vjp(
                        res_pattern['swiglu']
                    )
                    grad_outs = paddle.framework.core.call_vjp(
                        res_pattern['swiglu'],
                        fwd_inputs,
                        fwd_outputs,
                        [[src_pattern['swiglu_g'].operand_source(2)]],
                        stop_gradients,
                    )
                    res_pattern['swiglu_g'] = grad_outs[0][0].get_defining_op()
                    res_pattern['swiglu_g'].copy_attrs_from(
                        src_pattern['swiglu_g']
                    )

                    fwd_inputs, fwd_outputs, stop_gradients = prepare_for_vjp(
                        res_pattern['mm']
                    )
                    grad_outs = paddle.framework.core.call_vjp(
                        res_pattern['mm'],
                        fwd_inputs,
                        fwd_outputs,
                        [[res_pattern['swiglu_g'].result(0)]],
                        stop_gradients,
                    )
                    res_pattern['mm_g'] = grad_outs[0][0].get_defining_op()
                    res_pattern['mm_g'].copy_attrs_from(src_pattern['mm1_g'])

                    src_pattern['add_n'].result(0).replace_all_uses_with(
                        res_pattern['mm_g'].result(0)
                    )

                    # solve bwd amp pattern
                    if (
                        src_pattern['mm2_g']
                        .result(1)
                        .first_use()
                        .owner()
                        .name()
                        == "pd_op.cast"
                        and src_pattern['mm2_g']
                        .result(1)
                        .first_use()
                        .owner()
                        .name()
                        == "pd_op.cast"
                    ):
                        src_pattern['cast_w_up_g'] = (
                            src_pattern['mm2_g'].result(1).first_use().owner()
                        )
                        src_pattern['cast_w_gate_g'] = (
                            src_pattern['mm1_g'].result(1).first_use().owner()
                        )

                        # insert bwd amp dst_pattern
                        paddle.pir.set_insertion_point_after(
                            src_pattern['cast_w_gate_g']
                        )
                        y = paddle.cast(
                            res_pattern['mm_g'].result(1), 'float32'
                        )
                        res_pattern['cast_op'] = y.get_defining_op()
                        res_pattern['cast_op'].copy_attrs_from(
                            src_pattern['cast_w_gate_g']
                        )

                    for op in dense_main_program.global_block().ops:
                        if not op.has_attr("op_role"):
                            op.op_role = 1

            src_pattern_list.append(src_pattern)

    if len(src_pattern_list) == 0:
        return

    # # (2) Construct result pattern of ffn

    #     res_pattern_list = []
    #     for src_pattern in src_pattern_list:
    #         res_pattern = paddle.static.Program()
    #         with res_pattern.global_block():
    #             # # prepare fwd pattern
    #             # w_list = [src_pattern['mm1'].operand_source(1), src_pattern['mm2'].operand_source(1)]
    #             # _, fused_w = paddle._C_ops.concat_and_relocate_(w_list)

    #             # fused_o = paddle.matmul(src_pattern['mm1'].operand_source(0), fused_w, transpose_x=False, transpose_y=False)
    #             # out = paddle.incubate.nn.functional.swiglu(fused_o)

    #             # for op in res_pattern.global_block().ops:
    #             #     if not op.has_attr("op_role"):
    #             #         op.op_role = 0

    #             # prepare bwd pattern
    #             swiglu_op = res_pattern.global_block().ops[-1]
    #             swiglu_op.copy_attrs_from(src_pattern['swiglu'])
    #             matmul_op = res_pattern.global_block().ops[-2]
    #             matmul_op.copy_attrs_from(src_pattern['mm1'])

    #             fwd_inputs, fwd_outputs, stop_gradients = prepare_for_vjp(swiglu_op)
    #             paddle.framework.core.call_vjp(swiglu_op, fwd_inputs, fwd_outputs, [[src_pattern['swiglu_g'].operand_source(2)]], stop_gradients)
    #             swiglu_grad_op = res_pattern.global_block().ops[-1]
    #             swiglu_grad_op.copy_attrs_from(src_pattern['swiglu_g'])

    #             fwd_inputs, fwd_outputs, stop_gradients = prepare_for_vjp(matmul_op)
    #             paddle.framework.core.call_vjp(matmul_op, fwd_inputs, fwd_outputs, [[res_pattern.global_block().ops[-1].result(0)]], stop_gradients)
    #             matmul_grad_op = res_pattern.global_block().ops[-1]
    #             matmul_grad_op.copy_attrs_from(src_pattern['mm1_g'])

    #             cast_op = None
    #             if "cast_w_up_g" in src_pattern and "cast_w_gate_g" in src_pattern:
    #                 y = paddle.cast(matmul_grad_op.result(1), 'float32')
    #                 cast_op = res_pattern.global_block().ops[-1]
    #                 cast_op.copy_attrs_from(src_pattern['cast_w_gate_g'])

    #             for op in res_pattern.global_block().ops:
    #                 if not op.has_attr("op_role"):
    #                     op.op_role = 1

    #             # prepare opt pattern
    #             w_gate, w_up = paddle.split(fused_w, num_or_sections=2, axis=1)

    #             paddle._C_ops.assign_out_(w_gate, src_pattern['mm1'].operand_source(1))
    #             paddle._C_ops.assign_out_(w_up, src_pattern['mm2'].operand_source(1))

    #             if cast_op is None:
    #                 w_gate_g, w_up_g = paddle.split(matmul_grad_op.result(1), num_or_sections=2, axis=1)
    #             else:
    #                 w_gate_g, w_up_g = paddle.split(cast_op.result(0), num_or_sections=2, axis=1)

    #             for op in res_pattern.global_block().ops:
    #                 if not op.has_attr("op_role"):
    #                     op.op_role = 2

    #         res_pattern_list.append(res_pattern)

    #     # (3) Replace source pattern with result pattern: First, insert the operators from res_pattern into the original program, and then delete the operators from src_pattern in reverse order.
    #     print("dense_main_program: ", dense_main_program, flush=1)
    #     print("res_pattern_list: ", res_pattern_list, flush=1)
    #     for index in range(len(res_pattern_list)):
    #         src_pattern = src_pattern_list[index]
    #         res_pattern = res_pattern_list[index]
    #         # insert fwd pattern
    #         paddle.pir.set_insertion_point(src_pattern['swiglu'])

    #         # insert bwd pattern

    #         # insert opt pattern

    print("dense_main_program: ", dense_main_program, flush=1)
    # print("res_pattern_list: ", res_pattern_list, flush=1)


def fused_attention_qkv_pass(dense_mian_program):
    pass
