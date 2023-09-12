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

from collections import OrderedDict
from typing import List

from paddle.base import core
from paddle.base.framework import Parameter, Program
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_lr_sched_op,
    is_optimize_op,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


def list_to_ordered_dict(list_obj, ordered_dict=None):
    if ordered_dict is None:
        ordered_dict = OrderedDict()
    else:
        assert isinstance(ordered_dict, OrderedDict)
    for obj in list_obj:
        if obj not in ordered_dict:
            ordered_dict[obj] = True
    return ordered_dict


# The inputs of a program are the variables
# that first occur as the input of the op.
def get_inputs_of_program(program):
    visited_vars = set()
    input_vars = []
    for op in program.global_block().ops:
        for in_var_name in op.input_arg_names:
            if in_var_name not in visited_vars:
                input_vars.append(in_var_name)
                visited_vars.add(in_var_name)

        for out_var_name in op.output_arg_names:
            visited_vars.add(out_var_name)
    return input_vars


def get_outputs_of_program(program):
    output_vars = OrderedDict()
    for op in program.global_block().ops:
        list_to_ordered_dict(op.output_arg_names, output_vars)
    return list(output_vars.keys())


def prune_program(program, start_op_idx, end_op_idx):
    op_num = len(program.global_block().ops)
    if start_op_idx < 0:
        start_op_idx += op_num
    assert start_op_idx >= 0 and start_op_idx < op_num
    if end_op_idx < 0:
        end_op_idx += op_num
    assert end_op_idx >= 0 and end_op_idx <= op_num, end_op_idx
    assert start_op_idx < end_op_idx

    program = program.clone()
    for idx in range(op_num - 1, end_op_idx - 1, -1):
        program.global_block()._remove_op(idx, sync=False)
    for idx in range(start_op_idx - 1, -1, -1):
        program.global_block()._remove_op(idx, sync=False)
    program._sync_with_cpp()

    valid_vars = set()
    for op in program.global_block().ops:
        for in_var_name in op.input_arg_names:
            valid_vars.add(in_var_name)
        for out_var_name in op.output_arg_names:
            valid_vars.add(out_var_name)

    vars_to_remove = []
    for var in program.global_block().vars:
        if var not in valid_vars:
            vars_to_remove.append(var)

    for var in vars_to_remove:
        program.global_block()._remove_var(var, sync=False)
    program._sync_with_cpp()
    return program


def split_program(program, op_indices):
    """
    Split the program by op_indices.

    For examples, a program has 100 ops, and op_indices = [25, 60].
    Then the program is splitted into 3 parts, containing 25, 35 and 40
    ops respectively.

    The return values are a tuple with 3 elements: the splitted program
    list, the input var names of each splitted program, and the output
    var names of each splitted program.
    """
    assert op_indices, "op_indices cannot be empty"
    op_num = len(program.global_block().ops)
    assert op_num > 0, "program cannot be empty"

    op_indices = [idx if idx >= 0 else idx + op_num for idx in op_indices]

    if op_indices[0] != 0:
        op_indices = [0] + op_indices
    if op_indices[-1] != op_num:
        op_indices.append(op_num)

    for idx in range(len(op_indices) - 1):
        assert (
            op_indices[idx] < op_indices[idx + 1]
        ), "op_indices must be strictly sorted"

    splitted_programs = []
    for idx in range(len(op_indices) - 1):
        new_split = prune_program(program, op_indices[idx], op_indices[idx + 1])
        splitted_programs.append(new_split)

    num_split = len(splitted_programs)
    input_vars = [get_inputs_of_program(p) for p in splitted_programs]
    output_vars = [
        list_to_ordered_dict(get_outputs_of_program(p))
        for p in splitted_programs
    ]
    valid_output_vars = [OrderedDict() for _ in range(num_split)]
    valid_output_vars[-1] = output_vars[-1]
    for i in range(1, num_split):
        for in_var_name in input_vars[i]:
            for j in reversed(range(i)):
                if in_var_name in output_vars[j]:
                    valid_output_vars[j][in_var_name] = True
                    break
    valid_output_vars = [list(item.keys()) for item in valid_output_vars]
    return splitted_programs, input_vars, valid_output_vars


class OpInOutInfo:
    """
    Record unused buffer input_vars of op and other var_names except unused buffer input_vars
    """

    def __init__(self):
        self._is_build = False
        self._no_need_buffer_slots = set()
        self._other_arg_names_set = set()

    @property
    def is_build(self):
        return self._is_build

    def _get_op_attrs(self, op):
        inputs = {}
        for input_name in op.input_names:
            inputs[input_name] = op.input(input_name)
        outputs = {}
        for output_name in op.output_names:
            outputs[output_name] = op.output(output_name)
        attrs = {}
        for attr_name in op.attr_names:
            attrs[attr_name] = op.attr(attr_name)

        return inputs, outputs, attrs

    def build_info(self, op):
        inputs, outputs, attrs = self._get_op_attrs(op)
        self._no_need_buffer_slots = core.infer_no_need_buffer_slots(
            op.type, inputs, outputs, attrs
        )
        if len(self._no_need_buffer_slots) == 0:
            return

        for slot_name in op.input_names:
            if slot_name not in self._no_need_buffer_slots:
                for in_name in op.input(slot_name):
                    self._other_arg_names_set.add(in_name)

        for slot_name in op.output_names:
            if slot_name not in self._no_need_buffer_slots:
                for out_name in op.output(slot_name):
                    self._other_arg_names_set.add(out_name)

        self._is_build = True

    def is_needed(self, arg_name):
        return (
            len(self._no_need_buffer_slots) == 0
            or arg_name in self._other_arg_names_set
        )


def var_can_be_deleted(var_name, block):
    var = block._find_var_recursive(var_name)
    return var is not None and not var.persistable


def get_skip_gc_vars(program_list: List[Program]):
    """
    Get `skip_gc_vars` for every sub_program of program_list.

    A whole_program is split up into sub_programs according to the schedule mode,
    thus a sub_program's vars might be used as the op's input of the later sub_program,
    and these vars cannot be gc after executing current sub_program.
    """

    # step1: Get all vars of every sub_program of program_list that are non-persistable and not in op's no_need_buffer.
    required_vars = [set() for _ in range(len(program_list))]
    for idx, program in enumerate(program_list):
        for block in program.blocks:
            for op in block.ops:
                # NOTE(Ruibiao): Some vars maybe be the arguements of conditional_block op but no-need-buffer in the actual subblock, should not add them to the required_vars.
                if op.type == "conditional_block":
                    continue
                # NOTE(lizhiyu): In the PP, 'nop','c_sync_comm_stream' don't need to hold memory, because we have add special code for 'send_v2' when recording stream for gc.
                if op.type == "nop" or op.type == "c_sync_comm_stream":
                    continue
                op_info = OpInOutInfo()
                op_info.build_info(op)
                for arg_name in op.input_arg_names + op.output_arg_names:
                    if var_can_be_deleted(
                        arg_name, block
                    ) and op_info.is_needed(arg_name):
                        required_vars[idx].add(arg_name)

    # step2: Get the `skip_gc_vars` that vars of current sub_program might be used in the later sub_program
    suffixed_required_vars = set()
    skip_gc_vars = [set()] * len(program_list)
    for idx, vars_set in reversed(list(enumerate(required_vars))):
        if idx < len(required_vars) - 1:
            suffixed_required_vars = suffixed_required_vars.union(
                required_vars[idx + 1]
            )
        skip_gc_vars[idx] = vars_set & suffixed_required_vars

    return skip_gc_vars


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
        **copied_kwargs,
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
    This implementation refers to lots of Paddle/python/paddle/base/optimizer.py.
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
    """
    This implementation is for fthenb and 1f1b programs and is called in partial_programs function.
    """
    _insert_sync_for_fthenb_1f1b(program)

    lr_prog = Program()
    fwd_prog = Program()
    bwd_prog = Program()
    opt_prog = Program()

    # split the program based on the op_role
    def _split_ops(block):
        lr_ops = []
        fwd_ops = []
        bwd_ops = []
        opt_ops = []
        for op in src_block.ops:
            if is_lr_sched_op(op):
                lr_ops.append(op)
            elif is_forward_op(op):
                fwd_ops.append(op)
            elif is_backward_op(op):
                bwd_ops.append(op)
            elif is_optimize_op(op):
                opt_ops.append(op)
            else:
                raise ValueError(
                    "The op role: "
                    + str(op.attr('op_role'))
                    + " isn't one of LRSched, Forward, Backward or Optimizer."
                )
        return lr_ops, fwd_ops, bwd_ops, opt_ops

    def _add_ops_into_block(src_block, dst_block, ops):
        for op in ops:
            _create_program(src_block, dst_block, op)

    for idx, src_block in enumerate(program.blocks):
        lr_ops, fwd_ops, bwd_ops, opt_ops = _split_ops(src_block)
        if idx == 0:
            lr_block = lr_prog.block(0)
            _add_ops_into_block(src_block, lr_block, lr_ops)

            fwd_block = fwd_prog.block(0)
            _add_ops_into_block(src_block, fwd_block, fwd_ops)

            bwd_block = bwd_prog.block(0)
            _add_ops_into_block(src_block, bwd_block, bwd_ops)

            opt_block = opt_prog.block(0)
            _add_ops_into_block(src_block, opt_block, opt_ops)
        else:
            if len(lr_ops):
                lr_block = lr_prog._create_block(
                    parent_idx=src_block.parent_idx
                )
                lr_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, lr_block, lr_ops)

            if len(fwd_ops):
                fwd_block = fwd_prog._create_block(
                    parent_idx=src_block.parent_idx
                )
                fwd_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, fwd_block, fwd_ops)

            if len(bwd_ops):
                bwd_block = bwd_prog._create_block(
                    parent_idx=src_block.parent_idx
                )
                bwd_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, bwd_block, bwd_ops)

            if len(opt_ops):
                opt_block = opt_prog._create_block(
                    parent_idx=src_block.parent_idx
                )
                opt_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, opt_block, opt_ops)

        for fetch_op in src_block.ops:
            if fetch_op.type in ["fetch", "fetch_v2"]:
                in_name = fetch_op.input_arg_names[0]
                dst_block = None
                for block in [lr_block, fwd_block, bwd_block, opt_block]:
                    if block._find_var_recursive(in_name):
                        dst_block = block
                        break
                if dst_block:
                    _create_program(src_block, dst_block, fetch_op)

    lr_prog._sync_with_cpp()
    fwd_prog._sync_with_cpp()
    bwd_prog._sync_with_cpp()
    opt_prog._sync_with_cpp()

    lr_prog._rollback()
    fwd_prog._rollback()
    bwd_prog._rollback()
    opt_prog._rollback()

    # It MUST return in this order
    return [lr_prog, fwd_prog, bwd_prog, opt_prog]


def _add_event_dependency(recorder_op_desc, waiter_op_desc):
    '''
    Add the extra event dependcy of the two operators.
    This function mainly aims for the cross-programs in pipeline parallelism,
    especial for the 'send_v2' 'recv_v2' etc.
    '''
    if not recorder_op_desc.dist_attr.force_record_event:
        recorder_op_desc.dist_attr.force_record_event = True
    # NOTE(lizhiyu): Here is the copy of 'waiter_op_desc.dist_attr.events_to_wait' not the reference,
    #                because the type of 'events_to_wait' is 'const vector<string>&' while the type of
    #                'waiter_wait_list' is python list.
    waiter_wait_list = waiter_op_desc.dist_attr.events_to_wait
    if recorder_op_desc.dist_attr.event_to_record not in waiter_wait_list:
        waiter_wait_list.append(recorder_op_desc.dist_attr.event_to_record)
        waiter_op_desc.dist_attr.events_to_wait = waiter_wait_list
