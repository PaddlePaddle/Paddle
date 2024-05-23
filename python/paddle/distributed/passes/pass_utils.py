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

import logging
from collections import OrderedDict
from enum import Enum

from paddle.base import core
from paddle.base.framework import Parameter, Program
from paddle.distributed.auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
)
from paddle.distributed.auto_parallel.static.utils import (
    get_logger,
    is_backward_op,
    is_forward_op,
    is_optimize_op,
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping,
    use_new_executor,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]

logger = get_logger(logging.INFO)


# NOTE: Here stream is just a presentation with different name,
# it is up to executor to create the exact streams given the name.
class AutoParallelStreamType(Enum):
    CALC_STREAM = "default"
    MP_STREAM = "auto_parallel_mp"
    SHARDING_STREAM = "auto_parallel_sharding"


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


def set_skip_gc_vars(num_micro_batches, job_types, sub_programs, jobs):
    """
    Set `skip_gc_vars` for every job in jobs.

    A whole_program is split up into sub_programs according to the schedule mode,
    thus a sub_program's vars might be used as the op's input of the later sub_program,
    and these vars cannot be gc after executing current sub_program.
    """
    assert num_micro_batches >= 1, "num_micro_batches needs to be >= 1"
    type_to_program = dict(zip(job_types, sub_programs))

    # step1: Get all vars of every sub_program that are non-persistable and not in op's no_need_buffer.
    type_to_required_vars = {}
    for type, program in type_to_program.items():
        type_to_required_vars[type] = set()
        for block in program.blocks:
            for op in block.ops:
                if op.type in [
                    "c_sync_comm_stream",
                    "conditional_block",
                    "data",
                    "nop",
                    "while",
                ]:
                    continue

                op_info = OpInOutInfo()
                op_info.build_info(op)
                for arg_name in op.input_arg_names + op.output_arg_names:
                    if var_can_be_deleted(
                        arg_name, block
                    ) and op_info.is_needed(arg_name):
                        type_to_required_vars[type].add(arg_name)

    # step2: Set `skip_gc_vars` for each job
    suffixed_required_vars = [set() for i in range(num_micro_batches)]
    num_jobs = len(jobs)
    for job_id in reversed(range(num_jobs)):
        job = jobs[job_id]
        job_type = job.type()
        required_vars = type_to_required_vars[job_type]
        micro_batch_id = job.micro_batch_id()
        skip_gc_vars = required_vars & suffixed_required_vars[micro_batch_id]
        logger.debug(
            f"Skip gc vars for {job_type}-({micro_batch_id}): {skip_gc_vars}"
        )

        if job_type in ["backward", "backward_w"]:
            assert (
                len(skip_gc_vars) == 0
            ), f"When enabling pipeline parallelism strategy, the skip_gc_vars for {job_type} subprogram must be empty, but it is {skip_gc_vars}."

        job.set_skip_gc_vars(skip_gc_vars)
        suffixed_required_vars[micro_batch_id] |= required_vars

    return type_to_program


def shadow_var_between_sub_programs(sub_programs):
    """
    Add shadow_output and data op pair to share vars between sub_programs.
    """
    suffixed_shadow_arg_names = (
        set()
    )  # arg_names that are required in later sub_programs
    for sub_program in reversed(sub_programs):
        # step 1: parse shadow arguments
        block = sub_program.global_block()
        input_arg_names = set()
        output_arg_names = set()
        shadow_arg_names = set()
        for op in block.ops:
            for input_arg_name in op.input_arg_names:
                if var_can_be_deleted(input_arg_name, block):
                    # NOTE(zhangbo): In pir, transpose_grad op has only one input, Xshape is no longer the input.
                    if (
                        op.type == 'transpose2_grad'
                        and "XShape" in op.input_names
                    ):
                        if input_arg_name in op.input("XShape"):
                            continue
                    input_arg_names.add(input_arg_name)
                    # NOTE(Ruibiao): When translating these codes to pir, we can simplely set
                    # `shadow_arg_names=input_arg_names-output_arg_names` since the program
                    # in pir satisfies SSA form.
                    if input_arg_name not in output_arg_names:
                        shadow_arg_names.add(input_arg_name)
            for output_arg_name in op.output_arg_names:
                output_arg_names.add(output_arg_name)

        # step 2: add `shadow_output` op
        shadow_arg_names_for_suffixed_programs = (
            output_arg_names & suffixed_shadow_arg_names
        )
        for shadow_arg_name in shadow_arg_names_for_suffixed_programs:
            block.append_op(
                type="shadow_output",
                inputs={"x": shadow_arg_name},
                outputs={"out": shadow_arg_name},  # unused
                attrs={"name": shadow_arg_name},
            )

        # step3: add `data` op
        for shadow_arg_name in shadow_arg_names:
            shadow_var = block.var(shadow_arg_name)
            block._prepend_op(
                type="data",
                outputs={"out": shadow_arg_name},
                attrs={
                    "shape": shadow_var.shape,
                    "dtype": shadow_var.dtype,
                    "place": 2,  # GPUPlace
                    "name": shadow_arg_name,
                },
            )

        sub_program._sync_with_cpp()

        # step4: update suffixed_shadow_arg_names
        suffixed_shadow_arg_names -= shadow_arg_names_for_suffixed_programs
        suffixed_shadow_arg_names |= shadow_arg_names


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


def _insert_sync_for_fthenb_1f1b(program, dist_context=None):
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
                sync_calc_op = block._insert_op_without_sync(
                    index=index + offset,
                    type="c_sync_calc_stream",
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={'op_role': op_role},
                )
                offset += 1
                # step3: insert 'c_sync_comm_stream' op after 'send_v2' op or
                # before the first optimize op
                insert_index = None
                new_op_role = None
                if int(op_role) == int(OpRole.Backward):
                    insert_index = first_optimize_index + offset
                    new_op_role = OpRole.Optimize
                else:
                    insert_index = index + offset + 1
                    new_op_role = OpRole.Backward
                sync_comm_op = block._insert_op_without_sync(
                    index=insert_index,
                    type="c_sync_comm_stream",
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={
                        'op_role': new_op_role,
                        'ring_id': ring_id,
                    },
                )

                if dist_context:
                    dist_op = dist_context.get_dist_op_for_program(op)
                    if dist_op:
                        out_dist_attr = dist_op.dist_attr.get_input_dist_attr(
                            var_name
                        )
                        op_dist_attr = OperatorDistAttr()
                        op_dist_attr.process_mesh = (
                            dist_op.dist_attr.process_mesh
                        )
                        op_dist_attr.chunk_id = dist_op.dist_attr.chunk_id
                        op_dist_attr.set_input_dist_attr(
                            var_name, out_dist_attr
                        )
                        op_dist_attr.set_output_dist_attr(
                            var_name, out_dist_attr
                        )
                        dist_context.set_op_dist_attr_for_program(
                            sync_calc_op, op_dist_attr
                        )
                        dist_context.set_op_dist_attr_for_program(
                            sync_comm_op, op_dist_attr
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
                if not use_new_executor():
                    # NOTE: new executor will make sure gc are right without using nop op.
                    block._insert_op_without_sync(
                        index=backward_recv_index,
                        type="nop",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': OpRole.Backward},
                    )
        block._sync_with_cpp()


def _overlap_send_recv(program):
    """
    This function is used to replace the function '_insert_sync_for_fthenb_1f1b'.
    The finally target of this function is as follows:
        1. no need to insert the 'c_sync_calc' and 'c_sync_calc' operators
        2. 'send_v2' operator uses 'dist_attr.execution_stream' to set stream of its own.
        3. 'recv_v2' operator uses 'dist_attr.execution_stream' to set stream of its own.
    """
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'send_v2':
                op._set_attr("dynamic_shape", False)
                op._set_attr("use_calc_stream", True)
                ring_id = op.attr("ring_id")
                op.dist_attr.execution_stream = "send_stream_" + str(ring_id)
                op.dist_attr.stream_priority = 0
            elif op.type == 'recv_v2':
                op._set_attr("dynamic_shape", False)
                op._set_attr("use_calc_stream", True)
                op.dist_attr.execution_stream = "recv_stream"
                op.dist_attr.stream_priority = 0
            else:
                pass


def _add_ops_into_block(src_block, dst_block, ops):
    for op in ops:
        _create_program(src_block, dst_block, op)


def _is_fetch_op(op):
    return op.type in ["fetch", "fetch_v2"]


def _program_for_fthenb_and_1f1b(program, enable_send_recv_overlap=False):
    """
    This implementation is for fthenb and 1f1b programs and is called in partial_programs function.
    """
    if enable_send_recv_overlap:
        _overlap_send_recv(program)
    else:
        _insert_sync_for_fthenb_1f1b(program)

    fwd_prog = Program()
    bwd_prog = Program()
    opt_prog = Program()

    # split the program based on the op_role
    def _split_ops(block):
        fwd_ops = []
        bwd_ops = []
        opt_ops = []
        fetch_ops = []
        for op in block.ops:
            if _is_fetch_op(op):
                fetch_ops.append(op)
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
                    + " isn't one of Forward, Backward or Optimizer."
                )
        return fwd_ops, bwd_ops, opt_ops, fetch_ops

    for idx, src_block in enumerate(program.blocks):
        fwd_ops, bwd_ops, opt_ops, fetch_ops = _split_ops(src_block)
        if idx == 0:
            fwd_block = fwd_prog.block(0)
            _add_ops_into_block(src_block, fwd_block, fwd_ops)

            bwd_block = bwd_prog.block(0)
            _add_ops_into_block(src_block, bwd_block, bwd_ops)

            opt_block = opt_prog.block(0)
            _add_ops_into_block(src_block, opt_block, opt_ops)
        else:
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

        for fetch_op in fetch_ops:
            in_name = fetch_op.input_arg_names[0]
            dst_block = None
            for block in [fwd_block, bwd_block, opt_block]:
                if block._find_var_recursive(in_name):
                    dst_block = block
                    break
            if dst_block:
                _create_program(src_block, dst_block, fetch_op)

    fwd_prog._sync_with_cpp()
    bwd_prog._sync_with_cpp()
    opt_prog._sync_with_cpp()

    fwd_prog._roll_to_global_block()
    bwd_prog._roll_to_global_block()
    opt_prog._roll_to_global_block()

    # It MUST return in this order
    return [fwd_prog, bwd_prog, opt_prog]


def _program_for_vpp(
    program, num_model_chunks, dist_context, enable_send_recv_overlap=False
):
    if enable_send_recv_overlap:
        _overlap_send_recv(program)
    else:
        _insert_sync_for_fthenb_1f1b(program, dist_context)

    oprole_type = {0: "forward", 1: "backward", 2: "optimizer"}

    def _split_ops(block):
        type_to_ops = OrderedDict()
        chunk_ids = list(range(num_model_chunks))
        for type in oprole_type.values():
            if type == "optimizer":
                type_to_ops[type] = []
            else:
                chunk_ids = (
                    chunk_ids if type != "backward" else reversed(chunk_ids)
                )
                for chunk_id in chunk_ids:
                    type_to_ops[type + str(chunk_id)] = []
        type_to_ops["fetch"] = []

        for ip, op in enumerate(block.ops):
            if is_forward_op(op):
                type = oprole_type[0]
            elif is_backward_op(op):
                type = oprole_type[1]
            elif is_optimize_op(op):
                type = oprole_type[2]
            else:
                raise ValueError(
                    "The op role: "
                    + str(op.attr('op_role'))
                    + " isn't one of Forward, Backward or Optimizer."
                )

            dist_op = dist_context.get_dist_op_for_program(op)
            if _is_fetch_op(op):
                type_to_ops["fetch"].append(op)
            elif is_optimize_op(op):
                type_to_ops[type].append(op)
            elif op.type == "feed":
                type_to_ops[type + str(0)].append(op)
            elif op.type == "share_buffer":
                dist_pre_op = dist_context.get_dist_op_for_program(
                    block.ops[ip - 1]
                )
                type_to_ops[type + str(dist_pre_op.dist_attr.chunk_id)].append(
                    op
                )
            elif (
                dist_op
                and type + str(dist_op.dist_attr.chunk_id) in type_to_ops
            ):
                type_to_ops[type + str(dist_op.dist_attr.chunk_id)].append(op)
            else:
                raise ValueError(f"There is not dist_attr for op[{op.type}].")

        return type_to_ops

    type_to_program = OrderedDict()

    for ib, src_block in enumerate(program.blocks):
        type_to_ops = _split_ops(src_block)
        fetch_ops = type_to_ops.pop("fetch", [])
        dst_blocks = []

        if ib == 0:
            for type, ops in type_to_ops.items():
                type_to_program[type] = Program()
                dst_block = type_to_program[type].block(0)
                _add_ops_into_block(src_block, dst_block, ops)
                dst_blocks.append(dst_block)
        else:
            for type, ops in type_to_ops.items():
                if len(ops) > 0:
                    dst_block = type_to_program[type]._create_block(
                        parent_idx=src_block.parent_idx
                    )
                    dst_block._set_forward_block_idx(
                        src_block.forward_block_idx
                    )
                    _add_ops_into_block(src_block, dst_block, ops)
                    dst_blocks.append(dst_block)

        for fetch_op in fetch_ops:
            in_name = fetch_op.input('X')[0]
            fetch_block = None
            for dst_block in dst_blocks:
                if dst_block._find_var_recursive(in_name):
                    fetch_block = dst_block
                    break

            if fetch_block:
                _create_program(src_block, fetch_block, fetch_op)

    for prog in type_to_program.values():
        prog._sync_with_cpp()
        prog._roll_to_global_block()

    return list(type_to_program.keys()), list(type_to_program.values())


def _program_for_vpp_split_bwk(
    program,
    num_model_chunks,
    dist_context,
    enable_send_recv_overlap=False,
):
    if enable_send_recv_overlap:
        _overlap_send_recv(program)
    else:
        _insert_sync_for_fthenb_1f1b(program, dist_context)

    oprole_type = {
        0: "forward",
        1: "backward",
        2: "backward_b",
        3: 'backward_w',
        4: "optimizer",
    }

    def _split_ops(block):
        type_to_ops = OrderedDict()
        for type in oprole_type.values():
            chunk_ids = list(range(num_model_chunks))
            if type == "optimizer":
                type_to_ops[type] = []
            else:
                chunk_ids = (
                    chunk_ids if "backward" not in type else reversed(chunk_ids)
                )
                for chunk_id in chunk_ids:
                    type_to_ops[type + str(chunk_id)] = []
        type_to_ops["fetch"] = []

        dealed_op_idx = 0
        for ip, op in enumerate(block.ops):
            if ip < dealed_op_idx:
                continue
            if is_forward_op(op):
                type = oprole_type[0]
            elif is_backward_op(op):
                types = _get_backward_op_type(block, op, ip)
                dealed_op_idx = dealed_op_idx + len(types) - 1
            elif is_optimize_op(op):
                type = oprole_type[4]
            else:
                raise ValueError(
                    "The op role: "
                    + str(op.attr('op_role'))
                    + " isn't one of Forward, Backward or Optimizer."
                )

            dist_op = dist_context.get_dist_op_for_program(op)
            if _is_fetch_op(op):
                type_to_ops["fetch"].append(op)
            elif is_optimize_op(op):
                type_to_ops[type].append(op)
            elif op.type == "feed":
                type_to_ops[type + str(0)].append(op)
            elif op.type == "share_buffer":
                dist_pre_op = dist_context.get_dist_op_for_program(
                    block.ops[ip - 1]
                )
                type_to_ops[type + str(dist_pre_op.dist_attr.chunk_id)].append(
                    op
                )
            elif (
                dist_op
                and type + str(dist_op.dist_attr.chunk_id) in type_to_ops
                and not is_backward_op(op)
            ):
                type_to_ops[type + str(dist_op.dist_attr.chunk_id)].append(op)
            elif (
                dist_op
                and type + str(dist_op.dist_attr.chunk_id) in type_to_ops
                and is_backward_op(op)
            ):
                for i, type in enumerate(types):
                    type_to_ops[
                        "backward" + str(dist_op.dist_attr.chunk_id)
                    ].append(block.ops[ip + i])
                    type_to_ops[type + str(dist_op.dist_attr.chunk_id)].append(
                        block.ops[ip + i]
                    )
            else:
                raise ValueError(f"There is not dist_attr for op[{op.type}].")
            dealed_op_idx = dealed_op_idx + 1

        return type_to_ops

    type_to_program = OrderedDict()

    for ib, src_block in enumerate(program.blocks):
        type_to_ops = _split_ops(src_block)
        fetch_ops = type_to_ops.pop("fetch", [])
        dst_blocks = []

        if ib == 0:
            for type, ops in type_to_ops.items():
                type_to_program[type] = Program()
                dst_block = type_to_program[type].block(0)
                _add_ops_into_block(src_block, dst_block, ops)
                dst_blocks.append(dst_block)
        else:
            for type, ops in type_to_ops.items():
                if len(ops) > 0:
                    dst_block = type_to_program[type]._create_block(
                        parent_idx=src_block.parent_idx
                    )
                    dst_block._set_forward_block_idx(
                        src_block.forward_block_idx
                    )
                    _add_ops_into_block(src_block, dst_block, ops)
                    dst_blocks.append(dst_block)

        for fetch_op in fetch_ops:
            in_name = fetch_op.input('X')[0]
            fetch_block = None
            for dst_block in dst_blocks:
                if dst_block._find_var_recursive(in_name):
                    fetch_block = dst_block
                    break

            if fetch_block:
                _create_program(src_block, fetch_block, fetch_op)

    for prog in type_to_program.values():
        prog._sync_with_cpp()
        prog._roll_to_global_block()

    return list(type_to_program.keys()), list(type_to_program.values())


def _get_backward_op_type(block, cur_op, idx):
    # deal the ops pattern: [reshape2, reshape2, matmul_v2, reshape2, elementwise_add]
    def is_reshape_matmul_pattern(cur_op, idx, ops, ops_len):
        ops_pattern = [
            "reshape2",
            "reshape2",
            "matmul_v2",
            "reshape2",
            "elementwise_add",
        ]
        if cur_op.type == "reshape2":
            if idx + 4 < ops_len:
                ops_names = []
                for i in range(idx, idx + 5):
                    if not is_backward_op(ops[i]):
                        return False
                    if ops[i].type == "matmul_v2":
                        output_arg_names = ops[i].output_arg_names
                        name = output_arg_names[0].split("@")[0]
                        if not block._find_var_recursive(name):
                            return False
                        var = block._find_var_recursive(name)
                        if not var.is_parameter:
                            return False
                    ops_names.append(ops[i].type)
                if ops_names == ops_pattern:
                    return True
        return False

    # For the cur_op doesn't have output such as 'send_v2', it should be backward_b.
    if len(cur_op.output_arg_names) == 0:
        return ["backward_b"]

    if is_reshape_matmul_pattern(cur_op, idx, block.ops, len(block.ops)):
        return [
            "backward_w",
            "backward_w",
            "backward_w",
            "backward_w",
            "backward_w",
        ]
    for name in cur_op.output_arg_names:
        name = name.split("@")[0]
        if not block._find_var_recursive(name):
            return ["backward_b"]
        var = block._find_var_recursive(name)
        if not var.is_parameter:
            return ["backward_b"]

    return ["backward_w"]


def _program_for_zero_bubble(program, enable_send_recv_overlap=False):
    if enable_send_recv_overlap:
        _overlap_send_recv(program)
    else:
        _insert_sync_for_fthenb_1f1b(program)

    oprole_type = {
        0: "forward",
        1: "backward",
        2: "backward_b",
        3: 'backward_w',
        4: "optimizer",
    }

    def _split_ops(block):
        # split the program based on the op_role
        type_to_ops = OrderedDict()
        for type in oprole_type.values():
            type_to_ops[type] = []
        type_to_ops["fetch"] = []

        dealed_op_idx = 0
        for idx, op in enumerate(block.ops):
            if idx < dealed_op_idx:
                continue
            if _is_fetch_op(op):
                type_to_ops["fetch"].append(op)
            elif is_forward_op(op):
                type_to_ops["forward"].append(op)
            elif is_backward_op(op):
                types = _get_backward_op_type(block, op, idx)
                dealed_op_idx = dealed_op_idx + len(types) - 1
                for i, type in enumerate(types):
                    type_to_ops[type].append(block.ops[idx + i])
                    type_to_ops["backward"].append(block.ops[idx + i])
            elif is_optimize_op(op):
                type_to_ops["optimizer"].append(op)
            else:
                raise ValueError(
                    "The op role: "
                    + str(op.attr('op_role'))
                    + " isn't one of Forward, Backward or Optimizer."
                )
            dealed_op_idx = dealed_op_idx + 1
        return type_to_ops

    type_to_program = OrderedDict()
    for type in oprole_type.values():
        type_to_program[type] = Program()

    for idx, src_block in enumerate(program.blocks):
        type_to_ops = _split_ops(src_block)
        fwd_ops, bwd_ops, bwd_b_ops, bwd_w_ops, opt_ops, fetch_ops = (
            type_to_ops["forward"],
            type_to_ops["backward"],
            type_to_ops["backward_b"],
            type_to_ops["backward_w"],
            type_to_ops["optimizer"],
            type_to_ops["fetch"],
        )
        if idx == 0:
            fwd_block = type_to_program["forward"].block(0)
            _add_ops_into_block(src_block, fwd_block, fwd_ops)

            bwd_block = type_to_program["backward"].block(0)
            _add_ops_into_block(src_block, bwd_block, bwd_ops)

            bwd_block_b = type_to_program["backward_b"].block(0)
            _add_ops_into_block(src_block, bwd_block_b, bwd_b_ops)

            bwd_block_w = type_to_program["backward_w"].block(0)
            _add_ops_into_block(src_block, bwd_block_w, bwd_w_ops)

            opt_block = type_to_program["optimizer"].block(0)
            _add_ops_into_block(src_block, opt_block, opt_ops)
        else:
            if len(fwd_ops):
                fwd_block = type_to_program["forward"]._create_block(
                    parent_idx=src_block.parent_idx
                )
                fwd_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, fwd_block, fwd_ops)

            if len(bwd_ops):
                bwd_block = type_to_program["backward"]._create_block(
                    parent_idx=src_block.parent_idx
                )
                bwd_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, bwd_block, bwd_ops)

            if len(bwd_b_ops):
                bwd_block_b = type_to_program["backward_b"]._create_block(
                    parent_idx=src_block.parent_idx
                )
                bwd_block_b._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, bwd_block_b, bwd_b_ops)

            if len(bwd_w_ops):
                bwd_block_w = type_to_program["backward_w"]._create_block(
                    parent_idx=src_block.parent_idx
                )
                bwd_block_w._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, bwd_block_w, bwd_w_ops)

            if len(opt_ops):
                opt_block = type_to_program["optimizer"]._create_block(
                    parent_idx=src_block.parent_idx
                )
                opt_block._set_forward_block_idx(src_block.forward_block_idx)
                _add_ops_into_block(src_block, opt_block, opt_ops)

        for fetch_op in fetch_ops:
            in_name = fetch_op.input_arg_names[0]
            dst_block = None
            for block in [fwd_block, bwd_block_b, bwd_block_w, opt_block]:
                if block._find_var_recursive(in_name):
                    dst_block = block
                    break
            if dst_block:
                _create_program(src_block, dst_block, fetch_op)

    for prog in type_to_program.values():
        prog._sync_with_cpp()
        prog._roll_to_global_block()

    return list(type_to_program.keys()), list(type_to_program.values())


def _add_event_dependency(recorder_op, waiter_op):
    '''
    Add the extra event dependency of the two operators.
    This function mainly aims for the cross-programs in pipeline parallelism,
    especial for the 'send_v2' 'recv_v2' etc.
    '''
    if not recorder_op.dist_attr.force_record_event:
        recorder_op.dist_attr.force_record_event = True
    # NOTE(lizhiyu): Here is the copy of 'waiter_op.dist_attr.events_to_wait' not the reference,
    #                because the type of 'events_to_wait' is 'const vector<string>&' while the type of
    #                'waiter_wait_list' is python list.
    waiter_wait_list = waiter_op.dist_attr.events_to_wait
    if recorder_op.dist_attr.event_to_record not in waiter_wait_list:
        waiter_wait_list.append(recorder_op.dist_attr.event_to_record)
        waiter_op.dist_attr.events_to_wait = waiter_wait_list


def _insert_reshape_op(
    block,
    index,
    x,
    shape,
    op_role,
    chunk_id,
    dist_context,
    out=None,
    op_namescope="/",
):
    var_x = block.var(x[0])
    x_dist_attr = dist_context.get_tensor_dist_attr_for_program(var_x)

    if out is None:
        out = block.create_var(
            name=f"{x[0]}@reshape.out",
            dtype=var_x.dtype,
            persistable=False,
        )
        dist_context.set_tensor_dist_attr_for_program(out, x_dist_attr)

    x_shape = block.create_var(name=f"{x[0]}@reshape.xshape", dtype=var_x.dtype)
    dist_context.set_tensor_dist_attr_for_program(x_shape, x_dist_attr)

    reshape_op = block._insert_op_without_sync(
        index=index,
        type="reshape2",
        inputs={"X": x},
        outputs={"Out": out, "XShape": x_shape},
        attrs={
            "shape": shape,
            "op_role": op_role,
            'op_namescope': op_namescope,
        },
    )

    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
        reshape_op,
        process_mesh=x_dist_attr.process_mesh,
        ref_mapping=x_dist_attr.dims_mapping,
        ctx=dist_context,
        chunk_id=chunk_id,
    )

    return out


def split_matmul_grad_to_matmul(
    block, matmul_grad_id, dist_context, op_namescope="/"
):
    ops = block.ops
    matmul_grad_op = ops[matmul_grad_id]

    tran_x = matmul_grad_op.attr("trans_x")
    assert (
        not tran_x
    ), f"matmul_grad(id={matmul_grad_id}) with tran_x == True is not supported for spliting matmul_grad to matmul"
    tran_y = matmul_grad_op.attr("trans_y")
    assert (
        not tran_y
    ), f"matmul_grad(id={matmul_grad_id}) with tran_y == True is not supported for spliting matmul_grad to matmul"

    x = matmul_grad_op.input("X")
    y = matmul_grad_op.input("Y")
    out_grad = matmul_grad_op.input("Out@GRAD")
    x_grad = matmul_grad_op.output("X@GRAD")
    y_grad = matmul_grad_op.output("Y@GRAD")
    op_role = matmul_grad_op.attr("op_role")

    var_x = block.var(x[0])
    var_out_grad = block.var(out_grad[0])
    var_y_grad = block.var(y_grad[0])

    x_dims = var_x.shape
    out_grad_dims = var_out_grad.shape
    y_grad_dims = var_y_grad.shape

    assert len(x_dims) == len(
        out_grad_dims
    ), f"The rank of x must be equal to that of out_grad, but got x rank = {len(x_dims)} and out_grad rank = {len(out_grad_dims)}."
    if len(x_dims) > 2:
        assert (
            x_dims[0:2] == out_grad_dims[0:2]
        ), f"The first two dimensions of x must be equal to that of out_grad, but got x_dims:{x_dims} and out_grad_dims:{out_grad_dims}."
    new_x_dims = [x_dims[0] * x_dims[1]] + list(x_dims[2:])
    new_out_grad_dims = [out_grad_dims[0] * out_grad_dims[1]] + list(
        out_grad_dims[2:]
    )

    # NOTE(Ruibiao): Why insert reshape op here?
    # When the rank of input matrix is 3, MatmulGradKernel use reshape to fold the first two dimensions of x and out_grad (see FoldInitDims in matmul_grad_kernel_impl.h), and then calls blas.Matmul to calculate y_grad.
    # If we directly append matmul op to calculate y_grad without FoldInitDims, blas.BatchedGEMM is actually called in MatmulKernel, which has a larger cost than using blas.Matmul after dimension folding.
    # Therefore, we imitate MatmulGradKernel here by inserting reshape op before matmul.
    chunk_id = dist_context.get_op_dist_attr_for_program(
        matmul_grad_op
    ).chunk_id
    new_x = _insert_reshape_op(
        block,
        matmul_grad_id + 1,
        x,
        new_x_dims,
        op_role,
        chunk_id=chunk_id,
        dist_context=dist_context,
        op_namescope=op_namescope,
    )
    new_out_grad = _insert_reshape_op(
        block,
        matmul_grad_id + 2,
        out_grad,
        new_out_grad_dims,
        op_role,
        chunk_id=chunk_id,
        dist_context=dist_context,
        op_namescope=op_namescope,
    )
    new_y_grad = block.create_var(
        name=f"{y_grad[0]}@reshape.out",
        dtype=var_y_grad.dtype,
        persistable=False,
    )

    dist_context.set_tensor_dist_attr_for_program(
        new_y_grad,
        dist_context.get_tensor_dist_attr_for_program(var_y_grad),
    )

    matmul_grad_dist_attr = dist_context.get_op_dist_attr_for_program(
        matmul_grad_op
    )

    matmul_op = block._insert_op_without_sync(
        index=matmul_grad_id + 3,
        type="matmul_v2",
        inputs={"X": new_x, "Y": new_out_grad},
        outputs={"Out": new_y_grad},
        attrs={
            "trans_x": True,
            "trans_y": False,
            "op_role": op_role,
            'op_namescope': op_namescope,
        },
    )

    dist_context.set_op_dist_attr_for_program(matmul_op, matmul_grad_dist_attr)
    _insert_reshape_op(
        block,
        matmul_grad_id + 4,
        [new_y_grad.name],
        y_grad_dims,
        op_role,
        chunk_id=chunk_id,
        dist_context=dist_context,
        out=y_grad,
        op_namescope=op_namescope,
    )

    matmul_op = block._insert_op_without_sync(
        index=matmul_grad_id + 1,
        type="matmul_v2",
        inputs={"X": out_grad, "Y": y},
        outputs={"Out": x_grad},
        attrs={
            "trans_x": False,
            "trans_y": True,
            "op_role": op_role,
            'op_namescope': op_namescope,
        },
    )

    dist_context.set_op_dist_attr_for_program(matmul_op, matmul_grad_dist_attr)

    block._remove_op(matmul_grad_id, sync=False)
