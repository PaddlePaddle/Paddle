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

import collections
import logging
import re

import paddle
from paddle.autograd.backward_utils import ValueDict
from paddle.base.log_helper import get_logger
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.distributed.passes.pass_base import PassContext, new_pass

from .process_group import get_process_group
from .reshard_funcs.base_reshard_func import (
    choose_reshard_func,
)
from .reshard_funcs.reshard_func_register import register_reshard_funcs
from .utils import (
    get_pp_stage_by_pp_degree,
    get_pp_stage_by_process_mesh,
    get_sub_process_mesh_by_program,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

register_reshard_funcs()

partition_skip_op_list = ["builtin.combine", "builtin.split"]
amp_ops = ["pd_op.check_finite_and_unscale_", "pd_op.update_loss_scaling_"]


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


def apply_partition_pass(program, pipeline_strategy=None):
    if pipeline_strategy.enable and pipeline_strategy.schedule_mode == "VPP":
        complete_chunk_id(program, pipeline_strategy)

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
            reshard_var = paddle._C_ops.reshard_v2(prev_var, operand_attr)
            operand.set_source(reshard_var)
            if ref_op_role is not None:
                insert_pos = paddle.pir.get_current_insertion_point()
                insert_pos.prev().op_role = ref_op_role

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
            reshard_var_1.get_defining_op().op_role = ref_op_role
            paddle.assign(reshard_var_1, prev_var)
            if ref_op_role is not None:
                insert_pos = paddle.pir.get_current_insertion_point()
                insert_pos.prev().op_role = ref_op_role
                paddle.pir.set_insertion_point_after(op)

            if old_dist_attr == result.dist_attr():
                continue
            reshard_var_2 = reshard_var_1
            if old_dist_attr != reshard_var_1.dist_attr():
                reshard_var_2 = paddle._C_ops.reshard_v2(result, old_dist_attr)
                reshard_var_2.get_defining_op().op_role = ref_op_role
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
                ref_op_role = op.op_role
                if ref_op_role is not None:
                    insert_pos = paddle.pir.get_current_insertion_point()
                    insert_pos.prev().op_role = ref_op_role


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
            ), f'There is no reshard function that matches src_dist_attr: {src_dist_attr} and dst_dist_attr: {dst_dist_attr}, {var.get_defining_op()}'
            paddle.pir.set_insertion_point(op)
            ref_op_role = op.op_role
            out_value = reshard_func.reshard(
                src_dist_attr,
                dst_dist_attr,
                op.operand_source(0),
                op.result(0).type(),
            )
            insert_pos = paddle.pir.get_current_insertion_point()
            insert_index = dist_program.global_block().ops.index(
                insert_pos.get_operation()
            )
            if ref_op_role is not None:
                op_offset = 1
                while (
                    dist_program.global_block()
                    .ops[insert_index - op_offset]
                    .op_role
                    is None
                ):
                    dist_program.global_block().ops[
                        insert_index - op_offset
                    ].op_role = ref_op_role
                    op_offset += 1
            if out_value is not None:
                op.result(0).replace_all_uses_with(out_value)
            if op.result(0).use_empty():
                op.erase()


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
    "pd_op.c_reducescatter",
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
        "VPP",
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


def _extract_seg_method(op, seg_method):
    regex = re.compile(seg_method, re.IGNORECASE)
    struct_name = (
        op.attrs()["struct_name"] if op.has_attr("struct_name") else "/"
    )
    m = regex.search(struct_name)
    if not m:
        return None
    return struct_name[m.start(0) :].split("/")[0]


def _get_seg_struct_names(ops, seg_method):
    fwd_start_op_index = 0
    for i, op in enumerate(ops):
        if _extract_seg_method(op, seg_method):
            fwd_start_op_index = i
            break

    total_op_num = len(ops)
    fwd_end_op_index = total_op_num - 1
    for i in reversed(range(total_op_num)):
        if ops[i].op_role == int(OpRole.Forward) and _extract_seg_method(
            ops[i], seg_method
        ):
            fwd_end_op_index = i
            break

    struct_names = collections.OrderedDict()
    for i in range(fwd_start_op_index, fwd_end_op_index + 1):
        struct_name = _extract_seg_method(ops[i], seg_method)
        if struct_name:
            struct_names[struct_name] = 1
        else:
            if ops[i].name() != "dist_op.reshard":
                raise ValueError(
                    f"The op {ops[i].name()} should only be created by reshard"
                )
            # reshard op belong to the next segment
            for j in range(i + 1, fwd_end_op_index + 1):
                struct_name = _extract_seg_method(ops[j], seg_method)
                if struct_name and struct_name not in struct_names:
                    ops[i].set_str_attr(
                        "struct_name", ops[j].attrs()["struct_name"]
                    )
                    break

    return list(struct_names.keys())


def _analyze_use_custom_mesh(ops, seg_method, pp_degree):
    non_use_custom_mesh = True
    seg_pp_stages = [-1]

    for op in ops:
        if op.op_role != int(OpRole.Forward):
            break

        if _extract_seg_method(op, seg_method) and "pd_op" in op.name():
            op_mesh = op.dist_attr.process_mesh
            pp_stage = get_pp_stage_by_process_mesh(op_mesh, pp_degree)
            if seg_pp_stages[-1] > pp_stage:
                non_use_custom_mesh = False
                break

    if not non_use_custom_mesh:
        _logger.info("Cannot Use Auto VPP")
    else:
        _logger.info("Using Auto VPP")

    return non_use_custom_mesh


def _set_process_mesh_and_chunk_id(
    op, process_mesh, chunk_id, non_use_custom_mesh=True
):
    def set_process_mesh(vars, attrs):
        for idx, (var, attr) in enumerate(zip(vars, attrs)):
            var_dist_attr = var.dist_attr()
            operand_attr = attr.as_tensor_dist_attr()
            if var_dist_attr is None:
                print("have none var dist attr", op.name())

            if var_dist_attr and var_dist_attr.process_mesh == op_mesh:
                tensor_dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        process_mesh,
                        var_dist_attr.dims_mapping,
                        var_dist_attr.partial_status,
                    )
                )

                var.update_dist_attr(tensor_dist_attr)

            if operand_attr.process_mesh == op_mesh:
                attrs[idx] = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        process_mesh,
                        operand_attr.dims_mapping,
                        operand_attr.partial_status,
                    )
                )

    op_dist_attr = op.dist_attr
    op_mesh = op_dist_attr.process_mesh
    op_input_attrs = op_dist_attr.operands()
    op_output_attrs = op_dist_attr.results()
    op_input_vars = op.operands_source()
    op_output_vars = op.results()

    if non_use_custom_mesh:
        set_process_mesh(op_input_vars, op_input_attrs)
        set_process_mesh(op_output_vars, op_output_attrs)

    op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
        process_mesh,
        op_input_attrs,
        op_output_attrs,
        chunk_id,
    )


def complete_chunk_id(dist_program, pipeline_strategy):
    if not pipeline_strategy.enable:
        return

    sub_process_meshes = get_sub_process_mesh_by_program(dist_program)
    pp_degree = pipeline_strategy.pp_degree
    vpp_degree = pipeline_strategy.vpp_degree
    seg_method = pipeline_strategy.vpp_seg_method
    schedule_mode = pipeline_strategy.schedule_mode
    num_chunks = pp_degree * vpp_degree

    if pp_degree < 2 and vpp_degree > 1:
        raise ValueError("VPP schedule mode only can be set in pipeline mode.")
    if vpp_degree > 1 and (not seg_method or schedule_mode != "VPP"):
        raise ValueError(
            "Please set right schedule_mode and vpp_seg_method for VPP."
        )
    if vpp_degree < 2:
        return

    ops = dist_program.global_block().ops
    seg_struct_names = _get_seg_struct_names(ops, seg_method)

    assert (
        len(seg_struct_names) % num_chunks == 0
    ), f"The number of layers[{seg_method}] ({len(seg_struct_names)}) should be divided by part number ({num_chunks})."

    # Step2: analysis whether the pp_stage is non-decreasing among segments
    # 1. if non_decreasing is True, the ops' process_mesh will be changed by vpp strategy
    # 2. if non_decreasing is False, the ops's process_mesh will not be changed.
    non_use_custom_mesh = _analyze_use_custom_mesh(ops, seg_method, pp_degree)

    # Step3: Get op index boundary, pp_stage, chunk_id, struct_names of each segment
    fwd_seg_pp_stages = [i % pp_degree for i in range(num_chunks)]
    fwd_seg_chunk_ids = [i // pp_degree for i in range(num_chunks)]
    fwd_seg_parts = [0]
    fwd_ops = [op for op in ops if op.op_role == int(OpRole.Forward)]

    for idx, fwd_op in enumerate(fwd_ops):
        if len(fwd_seg_parts) == len(seg_struct_names):
            break
        struct_name = _extract_seg_method(fwd_op, seg_method)
        if struct_name == seg_struct_names[len(fwd_seg_parts)]:
            fwd_seg_parts.append(idx)
    fwd_seg_parts.append(len(fwd_ops))

    backward_seg_pp_stages = [
        i % pp_degree for i in reversed(range(num_chunks))
    ]
    backward_seg_chunk_ids = [
        i // pp_degree for i in reversed(range(num_chunks))
    ]
    bwd_seg_parts = [0]
    bwd_ops = [op for op in ops if op.op_role == int(OpRole.Backward)]

    for idx, bwd_op in enumerate(bwd_ops):
        if len(bwd_seg_parts) == len(seg_struct_names):
            break
        struct_name = _extract_seg_method(bwd_op, seg_method)
        if (
            struct_name
            == seg_struct_names[len(seg_struct_names) - len(bwd_seg_parts)]
        ):
            bwd_seg_parts.append(idx)
    bwd_seg_parts.append(len(bwd_ops))

    # Step4: Set the process_mesh of each op
    for fwd_seg_id in range(len(fwd_seg_parts) - 1):
        start_idx = fwd_seg_parts[fwd_seg_id]
        end_idx = fwd_seg_parts[fwd_seg_id + 1]
        pp_stage = fwd_seg_pp_stages[fwd_seg_id]
        chunk_id = fwd_seg_chunk_ids[fwd_seg_id]
        struct_name = seg_struct_names[fwd_seg_id]
        process_mesh = sub_process_meshes[pp_stage]

        _logger.info(
            f"stage=[{pp_stage}], chunk_id=[{chunk_id}], layer_name=[{struct_name}]"
        )
        _logger.info(
            f"start op: [{fwd_ops[start_idx].name()}], end op: [{fwd_ops[end_idx - 1].name()}]"
        )

        for idx in range(start_idx, end_idx):
            _set_process_mesh_and_chunk_id(
                fwd_ops[idx],
                process_mesh,
                chunk_id,
                non_use_custom_mesh,
            )

    for bwd_seg_id in range(len(bwd_seg_parts) - 1):
        start_idx = bwd_seg_parts[bwd_seg_id]
        end_idx = bwd_seg_parts[bwd_seg_id + 1]
        pp_stage = backward_seg_pp_stages[bwd_seg_id]
        chunk_id = backward_seg_chunk_ids[bwd_seg_id]
        process_mesh = sub_process_meshes[pp_stage]

        for idx in range(start_idx, end_idx):
            _set_process_mesh_and_chunk_id(
                bwd_ops[idx],
                process_mesh,
                chunk_id,
                non_use_custom_mesh,
            )
