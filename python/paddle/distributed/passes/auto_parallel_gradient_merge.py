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
from __future__ import annotations

import paddle
from paddle.distributed.auto_parallel.static.process_group import (
    get_world_process_group,
)
from paddle.distributed.fleet.meta_optimizers.common import (
    OpRole,
)
from paddle.framework import (
    _current_expected_place_ as _get_device,
)

from .pass_base import PassBase, PassType, register_pass

world_process_group = get_world_process_group()


def _move_used_grad_op(used_grad_op, grad):
    move_to_opt_block_flag = True
    move_to_opt_ops = []
    cannot_move_op = ["pd_op.send_v2", "pd_op.send"]

    def find_move_op(backward_op):
        nonlocal move_to_opt_block_flag
        if not move_to_opt_block_flag or backward_op in move_to_opt_ops:
            return
        if backward_op.name() in cannot_move_op:
            move_to_opt_block_flag = False
            return
        if backward_op.num_operands() == 1:
            move_to_opt_block_flag = True
            move_to_opt_ops.append(backward_op)
        elif backward_op.name() == "pd_op.slice":
            move_to_opt_ops.append(backward_op)
            for i in range(0, backward_op.num_operands()):
                if not grad.is_same(backward_op.operand_source(i)):
                    move_to_opt_ops.append(
                        backward_op.operand_source(i).get_defining_op()
                    )
            move_to_opt_block_flag = True
        else:
            # NOTE(zhangwl):temp only consider one operand op
            move_to_opt_block_flag = False
            return
        for op_result in backward_op.results():
            for next_op in op_result.all_used_ops():
                if next_op.op_role != int(OpRole.Optimize):
                    find_move_op(next_op)

    find_move_op(used_grad_op)
    if move_to_opt_block_flag:
        for move_op in move_to_opt_ops:
            move_op.op_role = int(OpRole.Optimize)


def _pir_append_gradient_merge_backward_op(
    main_program,
    startup_program,
    params_grads,
):
    main_block = main_program.global_block()
    startup_block = startup_program.global_block()

    # {param: gradient_merge_var} to insert scale op and fill_constant op
    new_params_grads = []
    place = _get_device()
    if isinstance(place, paddle.framework.CUDAPlace):
        place = paddle.framework.CUDAPlace(
            paddle.distributed.ParallelEnv().dev_id
        )
    cur_place = paddle.base.libpaddle.Place()
    cur_place.set_place(place)

    for param, grad in params_grads:
        if grad is None:
            continue

        assert (
            not param.is_selected_row_type()
        ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

        grad_dtype = grad.dtype
        grad_type = grad.type()

        for op in grad.all_used_ops():
            if op.has_attr("master_grad_cast"):
                grad_dtype = op.result(0).dtype
                grad_type = op.result(0).type()

        # step1: create gradient_merge var and init with 0
        # Add persistable gradient variables in startup_program
        paddle.pir.set_insertion_point_to_block_end(startup_block)
        gradient_merge_var = paddle.full(
            shape=grad._local_shape, fill_value=0.0, dtype=grad_dtype
        )
        gradient_merge_var.persistable = True
        paddle.pir.set_insertion_point_after(
            gradient_merge_var.get_defining_op()
        )
        paddle._C_ops.set_persistable_value(
            gradient_merge_var, param.name + "@GRAD@MERGE"
        )

        # step2: Accumulate persistable gradient variables in main_program
        # NOTE(zhaoyingli): inplace operation must be 'a = a + b', cannot be 'a = b + a'
        grad_defining_op = grad.get_defining_op()
        paddle.pir.set_insertion_point_after(grad_defining_op)

        new_gradient_merge_var = main_block.add_kwarg(
            param.name + "@GRAD@MERGE", grad_type
        )
        new_gradient_merge_var.persistable = True
        new_gradient_merge_var.place_attr = cur_place
        new_gradient_merge_var_add = paddle._C_ops.add_(
            new_gradient_merge_var, grad
        )
        new_gradient_merge_var_add_op = (
            new_gradient_merge_var_add.get_defining_op()
        )
        new_gradient_merge_var_add_op.op_role = grad_defining_op.op_role

        new_gradient_merge_var_add_op.dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                grad_defining_op.dist_attr.process_mesh,
                grad_defining_op.dist_attr.operands(),
                grad_defining_op.dist_attr.results(),
                grad_defining_op.dist_attr.chunk_id,
            )
        )
        new_gradient_merge_var_add_op.set_bool_attr("grad_merge_add", True)

        # NOTE(zhangweilong): grad may in different device in auto_parallel, so need consider all_gather/all_reduce/split/... op
        for used_grad_op in grad.all_used_ops():
            _move_used_grad_op(used_grad_op, grad)

        opt_ops_use_grad = [
            op
            for op in grad.all_used_ops()
            if op.op_role == int(OpRole.Optimize)
        ]

        grad.replace_grad_users_with(
            new_gradient_merge_var, set(opt_ops_use_grad)
        )

        # reset gradient merge var to zero after finishing optimization
        paddle.pir.set_insertion_point_to_block_end(main_block)
        set_value = paddle.full(
            shape=[1], fill_value=float(0), dtype=grad_dtype
        )
        new_gradient_merge_var_zero = paddle._C_ops.set_value_with_tensor_(
            new_gradient_merge_var, set_value, [], [], [], [], [], []
        )

        set_value_op = new_gradient_merge_var_zero.get_defining_op()
        set_value_op.op_role = int(OpRole.Optimize)
        for id in range(1, set_value_op.num_operands()):
            op_input = set_value_op.operand_source(id)
            op_input.get_defining_op().op_role = int(OpRole.Optimize)

        # step3: Construct new_params_grads and grad_to_gradient_merge
        new_params_grads.append((param, new_gradient_merge_var))

    return new_params_grads


def _pir_move_reduce_to_backward_stage(main_program):
    pass


def _pir_remove_cast_for_master_grad(main_program, params_grads):
    for op in main_program.global_block().ops:
        if op.has_attr("master_grad_cast"):
            op.result(0).replace_all_uses_with(op.operand_source(0))
            op.erase()


def _find_trivial_optimizer_ops(block):
    optimizer_ops = []
    for op in block.ops:
        if "adam" in op.name() or "sgd" in op.name():
            optimizer_ops.append(op)
    return optimizer_ops


def _get_prev_op(block, optimizer_op):
    found = False
    for op in reversed(block.ops):
        if found:
            return op
        if op.id == optimizer_op.id:
            found = True
    return None


def _insert_scale_op_after(target_value, optimizer_op, scale, bias=0.0):
    scaled_grad = paddle._C_ops.scale_(target_value, scale, bias, False)

    scale_op = scaled_grad.get_defining_op()
    scale_op.op_role = int(OpRole.Optimize)

    full_op = scale_op.operand_source(1).get_defining_op()
    assert (
        full_op.name() == "pd_op.full"
    ), f"The defining op of the scale value should be `pd_op.full`, but got {full_op.name()}"
    full_op.op_role = int(OpRole.Optimize)

    if "adam" in optimizer_op.name():
        optimizer_op.operand(1).set_source(scaled_grad)
    elif "sgd" in optimizer_op.name():
        optimizer_op.operand(2).set_source(scaled_grad)


def _append_scale_op_before_comm(block, new_params_to_grads, k_steps):
    for op in reversed(block.ops):
        if op.op_role == int(OpRole.Backward):
            paddle.pir.set_insertion_point_after(op)
            break
        for _, new_grad in new_params_to_grads:
            new_grad = paddle._C_ops.scale_(new_grad, 1.0 / k_steps, 0.0, False)

            scale_op = new_grad.get_defining_op()
            scale_op.op_role = int(OpRole.Optimize)

            full_op = scale_op.operand_source(1).get_defining_op()
            assert (
                full_op.name() == "pd_op.full"
            ), f"The defining op of the scale value should be `pd_op.full`, but got {full_op.name()}"
            full_op.op_role = int(OpRole.Optimize)
    paddle.pir.set_insertion_point_to_block_end(block)


def _append_scale_op_after_comm(block, optimizer_ops, k_steps):
    for optimizer_op in optimizer_ops:
        target_value = None
        if "adam" in optimizer_op.name():  # adam and adamw are included
            target_value = optimizer_op.operand_source(1)
        elif "sgd" in optimizer_op.name():
            target_value = optimizer_op.operand_source(2)
        else:
            raise NotImplementedError(
                f"We yet support adamw, adam and sgd, but got {optimizer_op.name()}"
            )
        assert (
            target_value is not None
        ), "target_value is not expected to be None"
        insertion_point = target_value.get_defining_op()
        if insertion_point is None:
            # target_value is a gradient_merge_var, which hasn't defining_op
            # so we find the prev op of optimizer_op, inserting a scale op behind.
            insertion_point = _get_prev_op(block, optimizer_op)
        paddle.pir.set_insertion_point_after(insertion_point)
        _insert_scale_op_after(target_value, optimizer_op, 1.0 / k_steps)
    paddle.pir.set_insertion_point_to_block_end(block)


def _pir_append_scale_op(program, new_params_to_grads, k_steps):
    block = program.global_block()
    optimizer_ops = _find_trivial_optimizer_ops(block)
    if len(optimizer_ops) > 0:
        _append_scale_op_after_comm(block, optimizer_ops, k_steps)
    else:
        _append_scale_op_before_comm(block, new_params_to_grads, k_steps)


def _pir_parse_program(
    main_program,
    startup_program,
    params_grads,
    k_steps,
    avg,
    gradient_sync_after_accumulate,
):
    # step1: append gradient merge backward op to main_program
    new_params_to_grads = _pir_append_gradient_merge_backward_op(
        main_program, startup_program, params_grads
    )

    # step2: move back reduce op to backward stage
    if not gradient_sync_after_accumulate:
        _pir_move_reduce_to_backward_stage(main_program, params_grads)

    # _pir_remove_cast_for_master_grad(main_program, params_grads)

    # step3: append scale op
    if avg:
        _pir_append_scale_op(main_program, new_params_to_grads, k_steps)


@register_pass("auto_parallel_gradient_merge_pass")
class GradientMergePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("k_steps", -1)
        self.set_attr("avg", True)
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]

    def _check_self(self):
        if self.get_attr("k_steps") < 1:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        k_steps = self.get_attr("k_steps", -1)
        avg = self.get_attr("avg", False)
        params_grads = self.get_attr("params_grads")
        gradient_sync_after_accumulate = self.get_attr(
            "gradient_sync_after_accumulate", False
        )

        if self._in_pir_mode:
            with paddle.static.program_guard(main_program, startup_program):
                _pir_parse_program(
                    main_program,
                    startup_program,
                    params_grads,
                    k_steps,
                    avg,
                    gradient_sync_after_accumulate,
                )
        else:
            raise NotImplementedError(
                "auto_parallel_gradient_merge_pass() only support PIR now."
            )
