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
from paddle.base.framework import auto_complete_op_role
from paddle.distributed.passes.pass_base import PassContext, new_pass

from .process_group import get_process_group
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs
from .utils import dygraph_guard, get_pp_stage_by_pp_degree

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
        with auto_complete_op_role(program, op.op_role):
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
    with auto_complete_op_role(program, op.op_role):
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
            with auto_complete_op_role(program, ref_op_role):
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

            with auto_complete_op_role(program, ref_op_role):
                # reshard output to assign out input
                reshard_var_1 = paddle._C_ops.reshard_v2(
                    result, prev_var.dist_attr()
                )
                paddle.assign(reshard_var_1, prev_var)

            if old_dist_attr == result.dist_attr():
                continue

            reshard_var_2 = reshard_var_1
            if old_dist_attr != reshard_var_1.dist_attr():
                with auto_complete_op_role(program, ref_op_role):
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
                with auto_complete_op_role(program, op.op_role):
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

            with auto_complete_op_role(dist_program, ref_op_role):
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


# Replace the specific MoE-related dist op with the
# executable op in the dense program. In expert parallelism
# of the MoE model, the process mesh of each expert is
# different. Two specific apis are used to transform the
# input tensor's global process mesh to the experts' local
# process meshes, which will add two dist ops in the program.
# The following two functions are used to replace the two
# dist ops with the executable share_data_ ops.
def replace_moe_sub_mesh_tensors(op):
    cur_rank = paddle.distributed.get_rank()
    in_value = op.operand_source(0)
    out_value = None
    out_idx = -1
    for idx, val in enumerate(op.results()):
        val_mesh = val.dist_attr().process_mesh
        if cur_rank in val_mesh.process_ids:
            assert (
                out_value is None
            ), f'{op} has more than one results on rank {cur_rank}'
            out_value = val
            out_idx = idx

    paddle.pir.set_insertion_point(op)
    local_value = paddle._C_ops.share_data_(in_value)
    local_value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        out_value.type(), out_value.dist_attr()
    )
    local_value.set_type(local_value_type)
    out_value.replace_all_uses_with(local_value)

    op_dist_attr = op.dist_attr
    share_data_op = local_value.get_defining_op()
    share_data_op.dist_attr = (
        paddle.base.libpaddle.pir.create_op_dist_attribute(
            op_dist_attr.process_mesh,
            [op_dist_attr.operand(0).as_tensor_dist_attr()],
            [op_dist_attr.result(out_idx).as_tensor_dist_attr()],
        )
    )

    assert all(val.use_empty() for val in op.results())
    op.erase()


def replace_moe_global_mesh_tensor(op):
    cur_rank = paddle.distributed.get_rank()
    out_value = op.result(0)
    in_value = None
    in_idx = -1
    for idx, val in enumerate(op.operands_source()):
        val_mesh = val.dist_attr().process_mesh
        if cur_rank not in val_mesh.process_ids:
            continue
        assert (
            in_value is None
        ), f'{op} has more than one inputs on rank {cur_rank}'
        in_value = val
        in_idx = idx

    paddle.pir.set_insertion_point(op)
    local_value = paddle._C_ops.share_data_(in_value)
    # local_value = paddle.assign(in_value)
    local_value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
        out_value.type(), out_value.dist_attr()
    )
    local_value.set_type(local_value_type)
    out_value.replace_all_uses_with(local_value)

    op_dist_attr = op.dist_attr
    share_data_op = local_value.get_defining_op()
    share_data_op.dist_attr = (
        paddle.base.libpaddle.pir.create_op_dist_attribute(
            op_dist_attr.process_mesh,
            [op_dist_attr.operand(in_idx).as_tensor_dist_attr()],
            [op_dist_attr.result(0).as_tensor_dist_attr()],
        )
    )

    assert all(val.use_empty() for val in op.results())
    op.erase()


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
        if op.name() == "dist_op.moe_sub_mesh_tensors":
            replace_moe_sub_mesh_tensors(op)
            continue
        if op.name() == "dist_op.moe_global_mesh_tensor":
            replace_moe_global_mesh_tensor(op)
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
        res.get_defining_op().op_role = op.op_role
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
                reshape_var.get_defining_op().op_role = op.op_role
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


# ffn pattern:
# (o1) = "pd_op.matmul" (in, w1)
# (o2) = "pd_op.matmul" (in, w2)
# (o3) = "pd_op.swiglu" (o1, o2)
def is_ffn_pattern(op_list):
    if len(op_list) != 3:
        return False
    if op_list[0].name() != "pd_op.matmul":
        return False
    if op_list[1].name() != "pd_op.matmul":
        return False
    if op_list[2].name() != "pd_op.swiglu":
        return False
    return True


# qkv pattern:
# (o1) = "pd_op.matmul" (in, wq)
# (o2) = "pd_op.full_int_array" ()
# (o3) = "pd_op.reshape" (o1, o2)
# (o4) = "pd_op.matmul" (in, wk)
# (o5) = "pd_op.full_int_array" ()
# (o6) = "pd_op.reshape" (o4, o5)
# (o7) = "pd_op.matmul" (in, wv)
# (o8) = "pd_op.full_int_array" ()
# (o9) = "pd_op.reshape" (o7, o8)
def is_qkv_pattern(op_list):
    if len(op_list) != 9:
        return False
    if op_list[0].name() != "pd_op.matmul":
        return False
    if op_list[1].name() != "pd_op.full_int_array":
        return False
    if op_list[2].name() != "pd_op.reshape":
        return False
    if op_list[3].name() != "pd_op.matmul":
        return False
    if op_list[4].name() != "pd_op.full_int_array":
        return False
    if op_list[5].name() != "pd_op.reshape":
        return False
    if op_list[6].name() != "pd_op.matmul":
        return False
    if op_list[7].name() != "pd_op.full_int_array":
        return False
    if op_list[8].name() != "pd_op.reshape":
        return False
    return True


def get_param_op(program, param_name):
    all_ops = program.global_block().ops
    for i in range(len(all_ops)):
        if (
            all_ops[i].name() == "builtin.set_parameter"
            and all_ops[i].str_attr("parameter_name") == param_name
        ):
            return [all_ops[i], all_ops[i].operand_source(0).get_defining_op()]


def fuse_attention_ffn_qkv_pass(
    startup_program, main_program, concrete_program
):
    # 0. Prepare the data structure
    pir_param_names = []
    dy_param_names = []
    for i in range(len(concrete_program.parameters[1])):
        dy_param_names.append(concrete_program.parameters[0][i].name)
        pir_param_names.append(concrete_program.parameters[1][i].name)

    fused_w_pattern_map = {"ffn": [], "qkv": []}
    fused_w_name_map = {"ffn": [], "qkv": []}

    # 1. Traverse main_program, extract all ffn and qkv patterns.
    all_ops = main_program.global_block().ops
    for i in range(len(all_ops)):
        # check ffn pattern
        pat = all_ops[i : i + 3] if i + 3 <= len(all_ops) else all_ops[i:]
        if is_ffn_pattern(pat):
            fused_w_pattern_map['ffn'].append(pat)
            i = i + 3
            continue
        # check qkv pattern
        pat = all_ops[i : i + 9] if i + 9 <= len(all_ops) else all_ops[i:]
        if is_qkv_pattern(pat):
            fused_w_pattern_map['qkv'].append(pat)
            i = i + 9
            continue

    # 2. Replace all ffn and qkv patterns with fusion patterns, and record the weights after replacement.
    for pat in fused_w_pattern_map['ffn']:
        fusion_w_name = f"fused_{pat[0].operand_source(1).name}_{pat[1].operand_source(1).name}"
        fused_w_name_map["ffn"].append(
            {
                fusion_w_name: [
                    pat[0].operand_source(1).name,
                    pat[1].operand_source(1).name,
                ]
            }
        )
        fusion_w_dtype = pat[0].operand_source(1).dtype
        fusion_w_shape = pat[0].operand_source(1).shape
        fusion_w_shape[-1] *= 2
        fusion_w_process_mesh = pat[0].operand_source(1).process_mesh
        # Insert fusion parameter
        with paddle.static.program_guard(main_program, startup_program):
            fused_w = paddle.pir.core.create_parameter(
                dtype=fusion_w_dtype,
                shape=fusion_w_shape,
                name=fusion_w_name,
                process_mesh=fusion_w_process_mesh,
                placements=[
                    paddle.distributed.Replicate(),
                    paddle.distributed.Shard(1),
                ],
                initializer=paddle.nn.initializer.Constant(value=0),
            )
        # Insert dst pattern
        paddle.pir.set_insertion_point_after(pat[2])
        fused_o = paddle.matmul(
            pat[0].operand_source(0),
            fused_w,
            transpose_x=False,
            transpose_y=False,
        )
        fused_o.get_defining_op().copy_attrs_from(pat[0])
        out = paddle.incubate.nn.functional.swiglu(fused_o)
        out.get_defining_op().copy_attrs_from(pat[2])
        pat[2].result(0).replace_all_uses_with(out)

    for pat in fused_w_pattern_map['qkv']:
        fusion_w_name = f"fused_{pat[0].operand_source(1).name}_{pat[3].operand_source(1).name}_{pat[6].operand_source(1).name}"
        fused_w_name_map["qkv"].append(
            {
                fusion_w_name: [
                    pat[0].operand_source(1).name,
                    pat[3].operand_source(1).name,
                    pat[6].operand_source(1).name,
                ]
            }
        )
        fusion_w_dtype = pat[0].operand_source(1).dtype
        fusion_w_shape = pat[0].operand_source(1).shape
        fusion_w_shape[-1] *= 3
        fusion_w_process_mesh = pat[0].operand_source(1).process_mesh
        # insert fusion parameter
        with paddle.static.program_guard(main_program, startup_program):
            fused_w = paddle.pir.core.create_parameter(
                dtype=fusion_w_dtype,
                shape=fusion_w_shape,
                name=fusion_w_name,
                process_mesh=fusion_w_process_mesh,
                placements=[
                    paddle.distributed.Replicate(),
                    paddle.distributed.Shard(1),
                ],
                initializer=paddle.nn.initializer.Constant(value=0),
            )
        # insert dst pattern
        paddle.pir.set_insertion_point_after(pat[8])
        fused_o = paddle.matmul(
            pat[0].operand_source(0),
            fused_w,
            transpose_x=False,
            transpose_y=False,
        )
        fused_o.get_defining_op().copy_attrs_from(pat[0])
        out = paddle.reshape(
            fused_o,
            shape=[
                0,
                0,
                pat[2].result(0).shape[-2],
                pat[2].result(0).shape[-1] * 3,
            ],
        )
        out.get_defining_op().copy_attrs_from(pat[2])
        out_q, out_k, out_v = paddle.split(
            out,
            num_or_sections=[
                pat[2].result(0).shape[-1],
                pat[2].result(0).shape[-1],
                pat[2].result(0).shape[-1],
            ],
            axis=-1,
        )
        pat[2].result(0).replace_all_uses_with(out_q)
        pat[5].result(0).replace_all_uses_with(out_k)
        pat[8].result(0).replace_all_uses_with(out_v)

    # 3. Delete src pattern from origin program.
    del_ops = []
    for pat in fused_w_pattern_map['ffn']:
        for op in reversed(pat):
            del_ops.append(op)
            if op.name() == "pd_op.matmul":
                del_ops.append(op.operand_source(1).get_defining_op())
                del_ops.extend(
                    get_param_op(startup_program, op.operand_source(1).name)
                )
    for pat in fused_w_pattern_map['qkv']:
        for op in reversed(pat):
            del_ops.append(op)
            if op.name() == "pd_op.matmul":
                del_ops.append(op.operand_source(1).get_defining_op())
                del_ops.extend(
                    get_param_op(startup_program, op.operand_source(1).name)
                )
    for op in del_ops:
        op.erase()

    # 4. Initialize fused parameters and delete orignal parameters.
    for key, pat_list in fused_w_name_map.items():
        for pat in pat_list:
            for pir_param, dy_param_list in pat.items():
                # Retrieve the params of ffn and qkv patterns from concrete_program for fusion.
                concated_dy_param_list = []
                concated_dy_param_index = []
                for dy_param in dy_param_list:
                    param_index = dy_param_names.index(dy_param)
                    concated_dy_param_list.append(
                        concrete_program.parameters[0][param_index]
                    )
                    concated_dy_param_index.append(param_index)
                # Fuse params and init pir program fusion params.
                with dygraph_guard():
                    concated_param = paddle.concat(
                        x=[
                            obj._local_value() for obj in concated_dy_param_list
                        ],
                        axis=-1,
                    )
                pir_scope_param = (
                    paddle.static.global_scope().var(pir_param).get_tensor()
                )
                pir_scope_param._share_data_with(concated_param.get_tensor())
                # Pop and relase original params from concrete_program
                for index in concated_dy_param_index:
                    concrete_program.parameters[0].pop(index)
                    concrete_program.parameters[1].pop(index)
                for param in concated_dy_param_list:
                    param.get_tensor()._clear()

    return fused_w_name_map
