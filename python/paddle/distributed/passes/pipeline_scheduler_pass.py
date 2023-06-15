# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_lr_sched_op,
    is_optimize_op,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.fluid import core
from paddle.fluid.framework import Parameter, Program

from .pass_base import PassBase, register_pass

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


def _create_param(dst_block, src_var):
    copied_kwargs = {}
    copied_kwargs['trainable'] = src_var.trainable
    copied_kwargs['optimize_attr'] = src_var.optimize_attr
    copied_kwargs['regularizer'] = src_var.regularizer
    copied_kwargs['do_model_average'] = src_var.do_model_average
    copied_kwargs['need_clip'] = src_var.need_clip

    Parameter(
        block=dst_block,
        type=src_var.type,
        name=src_var.name,
        shape=src_var.shape,
        dtype=src_var.dtype,
        lod_level=src_var.lod_level,
        error_clip=src_var.error_clip,
        stop_gradient=src_var.stop_gradient,
        is_data=src_var.is_data,
        belong_to_optimizer=src_var.belong_to_optimizer,
        **copied_kwargs
    )


def _create_inter(dst_block, src_var):
    dst_block.create_var(
        type=src_var.type,
        name=src_var.name,
        shape=src_var.shape,
        dtype=src_var.dtype,
        lod_level=src_var.lod_level,
        persistable=src_var.persistable,
        error_clip=src_var.error_clip,
        stop_gradient=src_var.stop_gradient,
        is_data=src_var.is_data,
        belong_to_optimizer=src_var.belong_to_optimizer,
    )


def _create_var(src_block, dst_block, src_varname, force_create=False):
    if not force_create:
        src_var = src_block.var(src_varname)
    else:
        src_var = src_block._var_recursive(src_varname)
    if src_var.type in __not_shape_var_type__:
        persist = getattr(src_var, 'persistable', False)
        dst_block.create_var(
            type=src_var.type,
            name=src_var.name,
            persistable=persist,
            error_clip=src_var.error_clip,
            stop_gradient=src_var.stop_gradient,
            is_data=src_var.is_data,
            belong_to_optimizer=src_var.belong_to_optimizer,
        )
    else:
        if isinstance(src_var, Parameter):
            _create_param(dst_block, src_var)
        else:
            _create_inter(dst_block, src_var)


def _create_program(src_block, dst_block, src_op, force_create=False):
    dst_op_desc = dst_block.desc.append_op()
    dst_op_desc.copy_from(src_op.desc)
    for input_varname in src_op.input_arg_names:
        if src_block.has_var(input_varname) or (
            force_create and src_block._find_var_recursive(input_varname)
        ):
            _create_var(src_block, dst_block, input_varname, force_create)
    for output_varname in src_op.output_arg_names:
        if src_block.has_var(output_varname) or (
            force_create and src_block._find_var_recursive(output_varname)
        ):
            _create_var(src_block, dst_block, output_varname, force_create)


def _insert_sync_for_fthenb_1f1b(program):
    """
    This implementation refers to lots of Paddle/python/paddle/fluid/optimizer.py.
    The difference between this function with 'PipelineOptimizer' is that
    'send_v2' op and 'recv_v2' op have been inserted in program by 'reshard'.
    """

    for block in program.blocks:
        offset = 0
        first_optimize_index = None
        for index, op in enumerate(list(block.ops)):
            if is_optimize_op(op):
                first_optimize_index = index
                break

        # insert sync ops
        for index, op in enumerate(list(block.ops)):
            # NOTE: pipeline might hang when dynamic_shape is True
            if op.type in ['send_v2', 'recv_v2']:
                op._set_attr("dynamic_shape", False)
            # set send op on comm stream
            if op.type == 'send_v2':
                # step1: set 'use_calc_stream' False
                op._set_attr("use_calc_stream", False)
                op_role = op.attr('op_role')
                ring_id = op.attr('ring_id')
                # step2: insert 'c_sync_calc_stream' op before 'send_v2' op
                var_name = op.input_arg_names[0]
                var = block.var(var_name)
                block._insert_op_without_sync(
                    index=index + offset,
                    type="c_sync_calc_stream",
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={'op_role': op_role},
                )
                offset += 1
                # step3: insert 'c_sync_comm_stream' op after 'send_v2' op or
                # before the first optimize op
                if int(op_role) == int(OpRole.Backward):
                    index = first_optimize_index + offset
                    new_op_role = OpRole.Optimize
                else:
                    index = index + offset + 1
                    new_op_role = OpRole.Backward
                sync_comm_op = block._insert_op_without_sync(
                    index=index,
                    type="c_sync_comm_stream",
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={
                        'op_role': new_op_role,
                        'ring_id': ring_id,
                    },
                )
                # step4: If 'send_v2' op in forward parse, set 'pipeline_flag' to distinguish
                # whether the 'c_sync_comm_stream' op is inserted for pipeline.
                if int(op_role) == int(OpRole.Forward):
                    sync_comm_op._set_attr('pipeline_flag', '')
                    offset += 1
        block._sync_with_cpp()

        offset = 0
        backward_recv_index = None
        for index, op in enumerate(block.ops):
            if op.type == "recv_v2" and is_backward_op(op):
                backward_recv_index = index
                break
        if backward_recv_index is None:
            continue

        # replace 'c_sync_comm_stream' op with 'nop' op
        # use nop op for gc
        for index, op in enumerate(list(block.ops)):
            if index >= backward_recv_index:
                break
            if op.type == 'c_sync_comm_stream' and op.has_attr('pipeline_flag'):
                var_name = op.output_arg_names[0]
                var = block.var(var_name)
                block._remove_op(index + offset, sync=False)
                offset -= 1
                block._insert_op_without_sync(
                    index=backward_recv_index,
                    type="nop",
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={'op_role': OpRole.Backward},
                )
        block._sync_with_cpp()


def _program_for_fthenb_and_1f1b(program):
    lr_prog = Program()
    fwd_prog = Program()
    bwd_prog = Program()
    opt_prog = Program()

    for idx, src_block in enumerate(program.blocks):
        if idx == 0:
            lr_block = lr_prog.block(0)
            fwd_block = fwd_prog.block(0)
            bwd_block = bwd_prog.block(0)
            opt_block = opt_prog.block(0)
        else:
            lr_block = lr_prog._create_block(parent_idx=src_block.parent_idx)
            fwd_block = fwd_prog._create_block(parent_idx=src_block.parent_idx)
            bwd_block = bwd_prog._create_block(parent_idx=src_block.parent_idx)
            opt_block = opt_prog._create_block(parent_idx=src_block.parent_idx)
            lr_block._set_forward_block_idx(src_block.forward_block_idx)
            fwd_block._set_forward_block_idx(src_block.forward_block_idx)
            bwd_block._set_forward_block_idx(src_block.forward_block_idx)
            opt_block._set_forward_block_idx(src_block.forward_block_idx)

        # split the program based on the op_role
        for op in src_block.ops:
            if is_lr_sched_op(op):
                _create_program(src_block, lr_block, op)
            if is_forward_op(op):
                _create_program(src_block, fwd_block, op)
            elif is_backward_op(op):
                _create_program(src_block, bwd_block, op)
            elif is_optimize_op(op):
                _create_program(src_block, opt_block, op)
            else:
                raise ValueError(
                    "The op role: "
                    + str(op.attr('op_role'))
                    + " isn't one of LRSched, Forward, Backward or Optimizer."
                )

    lr_prog._sync_with_cpp()
    fwd_prog._sync_with_cpp()
    bwd_prog._sync_with_cpp()
    opt_prog._sync_with_cpp()

    lr_prog._rollback()
    fwd_prog._rollback()
    bwd_prog._rollback()
    opt_prog._rollback()

    return {
        "lr": lr_prog.desc,
        "forward": fwd_prog.desc,
        "backward": bwd_prog.desc,
        "optimizer": opt_prog.desc,
    }


@register_pass("pipeline_fthenb_scheduler")
class PipelineFThenBPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _create_job_list(self):
        job_list = []
        lr_job = core.job("lr")
        job_list.append(lr_job)
        for i in range(self._micro_batch_size):
            forward_job = core.job("forward")
            forward_job.set_micro_batch_id(i)
            job_list.append(forward_job)

        for i in range(self._micro_batch_size):
            backward_job = core.job("backward")
            backward_job.set_micro_batch_id(i)
            job_list.append(backward_job)

        opt_job = core.job("optimizer")
        job_list.append(opt_job)
        return job_list

    def _apply_single_impl(self, main_program, startup_program, context):
        self._micro_batch_size = self.get_attr("micro_batch_size")
        self._program = main_program

        _insert_sync_for_fthenb_1f1b(self._program)
        type_to_program = _program_for_fthenb_and_1f1b(self._program)
        job_list = self._create_job_list()

        plan = core.plan(job_list, type_to_program)
        context.set_attr("plan", plan)
