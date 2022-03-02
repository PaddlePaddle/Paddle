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
# limitations under the License

import copy
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import framework as framework
from paddle.fluid import core, unique_name
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container
from paddle.distributed.auto_parallel.dist_context import DistributedContext, DistributedOperatorContext
from .dist_attribute import OperatorDistributedAttribute
from .process_group import new_process_group
from .utils import set_dist_op_desc_original_id
from .utils import print_program_with_dist_attr, is_forward_op, is_backward_op
from .operators.common import BACKWARD_ONLY_DIST_OPS

__varname_not_in_block__ = ["lod_tensor_blocking_queue_0"]
__not_shape_var_type__ = [
    core.VarDesc.VarType.READER, core.VarDesc.VarType.STEP_SCOPES
]


class Partitioner(object):
    """
    warning:: Partitioner is experimental and subject to change.

    Partitioner convert a program into another program.
    Given a serial program which has been auto completed with shard annotation, the Partitioner 
    convert the serial program into a "distributed" program. The Partitioner will  modify the serial
    program in following two ways, which is also the major difference between serial and distributed program:
        1. partition op: replace a serial op into its corresponding dist op infered from the shard annotation
        2. partition var: if a var is sharded, modify the shape of var according to its shard annotation

    Partitioner is supposed to be call by the auto parallel framework, and not supposed to be directly called by user.
    """

    def __init__(self, dist_context, rank_id=0):
        """
        Args:
            dist_context (paddle.fluid.DistributedContext): used to access the distributed_attr of var & op, every Partitioner object could maintain its own DistributedContext member, and partition program base on that shard scenario.
            rank_id (int): global rank id to which the partitioned distributed program belong.
        """
        if not isinstance(dist_context, DistributedContext):
            raise TypeError(
                "dist_context be paddle.fluid.DistributedContext, got %s here" %
                type(dist_context))

        self._dist_context = dist_context
        self._rank_id = rank_id
        self._serial2dist_varname_mapping = {}
        self._dist_varname_suffix = ""

    def partition(self, serial_main_program, serial_startup_program,
                  params_grads):
        if not isinstance(serial_main_program, (Program)):
            raise TypeError(
                "main_program be paddle.fluid.framework.program, got %s here" %
                type(serial_main_program))

        # check if shard annotated serial program valid
        if not self._is_valid_annotated_program(serial_main_program):
            raise RuntimeError(
                "Not all vars or ops are annotated in main program !")

        # init distop helper
        dist_op_context = self._dist_context.dist_op_context
        dist_op_context.varname_mapping = self._serial2dist_varname_mapping
        dist_op_context.rank_id = self._rank_id

        # partition startup program
        if serial_startup_program == None:
            partitioned_startup_prog = None
        else:
            partitioned_startup_prog = self.partition_startup_program(
                serial_main_program, serial_startup_program)
        dist_op_context.dst_startup_program = partitioned_startup_prog

        # partition main program
        partitioned_main_prog, partitioned_params_grads = self.partition_main_program(
            serial_main_program, params_grads)

        return partitioned_main_prog, partitioned_startup_prog, partitioned_params_grads

    def partition_startup_program(self, serial_main_program,
                                  serial_startup_program):

        if not isinstance(serial_startup_program, (Program)):
            raise TypeError(
                "dist_context be paddle.fluid.framework.program, got %s here" %
                type(serial_startup_program))

        partitioned_startup_prog = fluid.Program()
        ref_block = serial_main_program.global_block()
        target_block = partitioned_startup_prog.global_block()
        var2shape = {}
        temp_varname_map = {}

        # tensors
        for var in serial_startup_program.list_vars():
            assert var.persistable
            new_name = var.name + self._dist_varname_suffix
            temp_varname_map[var.name] = new_name
            target_shape = _partition_var(self._dist_context, ref_block,
                                          target_block, var.name, new_name)
            var2shape[new_name] = target_shape

        # ops
        for op in serial_startup_program.global_block().ops:
            # TODO if var not belong to this rank, should be filtered
            output_vars = op.desc.output_arg_names()
            assert len(
                output_vars
            ) == 1, "initializer should output only ONE variable, but got [{}]".format(
                str(op.desc))
            assert temp_varname_map[output_vars[
                0]] in var2shape, "try to initialize [{}] which is not a persistable var".format(
                    output_vars[0])
            new_op_desc = target_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            new_op_desc._rename_output(output_vars[0],
                                       temp_varname_map[output_vars[0]])
            new_op_desc._set_attr("shape",
                                  var2shape[temp_varname_map[output_vars[0]]])
            target_block._sync_with_cpp()

            # set distribute atrribute
            new_op = target_block.ops[-1]
            assert new_op.type == new_op_desc.type()
            assert new_op.desc == new_op_desc
            output_var = target_block.var(output_vars[0])
            output_var_attr = self._dist_context.get_tensor_dist_attr_for_program(
                output_var)
            op_attr = OperatorDistributedAttribute()
            op_attr.process_mesh = output_var_attr.process_mesh
            op_attr.set_output_dims_mapping(output_var.name,
                                            output_var_attr.dims_mapping)
            op_attr.set_input_dims_mapping(output_var.name,
                                           output_var_attr.dims_mapping)
            self._dist_context.set_op_dist_attr_for_program(new_op, op_attr)

        return partitioned_startup_prog

    def partition_main_program(self, serial_main_program, params_and_grads):
        """
        1. partition variables
        2. replace local op with corresponding dist op
        """

        partitioned_main_prog = fluid.Program()
        dist_op_context = self._dist_context.dist_op_context
        dist_op_context.dst_main_program = partitioned_main_prog

        for idx in range(self._dist_context.block_state.nblock):
            ref_block = serial_main_program.blocks[idx]

            if idx == 0:
                target_block = partitioned_main_prog.blocks[0]
            else:
                target_block = partitioned_main_prog._create_block(
                    parent_idx=ref_block.parent_idx)
                assert ref_block.idx == target_block.idx
                target_block._set_forward_block_idx(ref_block.forward_block_idx)
            dist_op_context.work_block = target_block
            self.partition_block(ref_block, target_block)

        partitioned_main_prog.current_block_idx = 0

        partitioned_params_and_grads = []
        for p, g in params_and_grads:
            assert p.name in self._serial2dist_varname_mapping
            dist_p = self._get_dist_var_by_serial_var(p, partitioned_main_prog)
            if g is None:
                dist_g = None
            else:
                assert g.name in self._serial2dist_varname_mapping
                dist_g = self._get_dist_var_by_serial_var(g,
                                                          partitioned_main_prog)
            partitioned_params_and_grads.append((dist_p, dist_g))

        return partitioned_main_prog, partitioned_params_and_grads

    def partition_block(self, ref_block, target_block):

        dist_op_context = self._dist_context.dist_op_context
        serial_ops = ref_block.ops

        # init mapping
        forward_op_id2forward_op = {}
        for idx in range(len(serial_ops)):
            if is_forward_op(serial_ops[idx]):
                forward_op_id2forward_op[serial_ops[idx].desc.id(
                )] = serial_ops[idx]

        # partiiton
        for op in serial_ops:

            # partititon input variables
            for serial_input_varname in op.desc.input_arg_names():
                if serial_input_varname not in self._serial2dist_varname_mapping:
                    new_varname = serial_input_varname + self._dist_varname_suffix
                    if ref_block.has_var(serial_input_varname):
                        _partition_var(self._dist_context, ref_block,
                                       target_block, serial_input_varname,
                                       new_varname)
                    else:
                        assert serial_input_varname in __varname_not_in_block__

                    self._serial2dist_varname_mapping[
                        serial_input_varname] = new_varname

            # partition output vars
            for serial_output_varname in op.desc.output_arg_names():
                if serial_output_varname not in self._serial2dist_varname_mapping:
                    new_varname = serial_output_varname + self._dist_varname_suffix
                    _partition_var(self._dist_context, ref_block, target_block,
                                   serial_output_varname, new_varname)
                    self._serial2dist_varname_mapping[
                        serial_output_varname] = new_varname

            # partition op
            op_dist_attr = self._dist_context.get_op_dist_attr_for_program(op)
            if is_forward_op(op) or op_dist_attr.is_recompute:
                kinputs, koutputs = dist_op_context.prepare_context(op)
                dist_op_forward_impl = _get_dist_op_forward_implement(
                    op, self._dist_context)
                dist_op_forward_impl.forward(self._dist_context, **kinputs,
                                             **koutputs)

            elif is_backward_op(op):
                kinputs, koutputs = dist_op_context.prepare_context(op)
                dist_op_backward_impl = _get_dist_op_backward_implement(
                    op, self._dist_context, forward_op_id2forward_op)
                dist_op_backward_impl.backward(self._dist_context, **kinputs,
                                               **koutputs)
            else:
                raise NotImplementedError(
                    "partitioner only support forward op and backward op, but got {}".
                    format(str(op)))

    def _is_valid_annotated_program(self, program):

        # TODO (ZJ-LIANG) should check all block
        ops = program.global_block().ops
        vars_ = program.list_vars()
        op_dist_attrs = [
            self._dist_context.get_op_dist_attr_for_program(op) for op in ops
        ]
        var_dist_attrs = [
            self._dist_context.get_tensor_dist_attr_for_program(var)
            for var in vars_ if (var.type not in __not_shape_var_type__)
        ]

        all_ops_annotated = all(dist_attr is not None
                                for dist_attr in op_dist_attrs)
        all_vars_annotated = all(dist_attr is not None
                                 for dist_attr in var_dist_attrs)

        return all_ops_annotated and all_vars_annotated

    def _get_dist_var_by_serial_var(self, serial_var, partitioned_main_prog):

        block_idx = serial_var.block.idx
        target_block = partitioned_main_prog.blocks[block_idx]
        dist_var_name = self._serial2dist_varname_mapping[serial_var.name]
        assert target_block.has_var(dist_var_name)
        return target_block.var(dist_var_name)


def _get_dist_shape(var, dist_attr):

    var_shape = var.shape
    mapping = dist_attr.dims_mapping
    mesh = dist_attr.process_mesh.topology
    if mapping == []:
        return var_shape

    assert len(var_shape) == len(
        mapping
    ), "variable shape [{}] and dim_mapping [{}] is NOT match !".format(
        var_shape, mapping)
    new_shape = []
    for idx in range(len(var_shape)):
        if var_shape[idx] == -1 or mapping[idx] == -1:
            new_shape.append(var_shape[idx])
        else:
            assert var_shape[idx] % mesh[mapping[
                idx]] == 0, "un-event partition: var_shape[idx]=[{}], mesh[{}]".format(
                    var_shape[idx], mesh[mapping[idx]])
            new_shape.append(var_shape[idx] // mesh[mapping[idx]])

    return new_shape


def _partition_parameter(dist_context, src_var, dst_block, dst_varname,
                         dst_shape):
    # NOTE hack to copied Parameter
    # not initialized parameter, need to initialize it
    copied_kwargs = {}
    copied_kwargs['trainable'] = src_var.trainable
    copied_kwargs['optimize_attr'] = src_var.optimize_attr
    copied_kwargs['regularizer'] = src_var.regularizer
    copied_kwargs['do_model_average'] = src_var.do_model_average
    copied_kwargs['need_clip'] = src_var.need_clip

    param = Parameter(
        block=dst_block,
        type=src_var.type,
        name=dst_varname,
        shape=dst_shape,
        dtype=src_var.dtype,
        lod_level=src_var.lod_level,
        error_clip=src_var.error_clip,
        stop_gradient=src_var.stop_gradient,
        is_data=src_var.is_data,
        belong_to_optimizer=src_var.belong_to_optimizer,
        **copied_kwargs)

    # set dist attr uid
    # distributed_attr_uid = src_var.desc.get_distributed_attr_uid()
    # param.desc.set_distributed_attr_uid(distributed_attr_uid)
    dist_attr = copy.deepcopy(
        dist_context.get_tensor_dist_attr_for_program(src_var))
    assert dist_attr is not None
    dist_context.set_tensor_dist_attr_for_program(param, dist_attr)


def _partition_intermediate_var(dist_context, src_var, dst_block, dst_varname,
                                dst_shape):
    var = dst_block.create_var(
        type=src_var.type,
        name=dst_varname,
        shape=dst_shape,
        dtype=src_var.dtype,
        lod_level=src_var.lod_level,
        persistable=src_var.persistable,
        error_clip=src_var.error_clip,
        stop_gradient=src_var.stop_gradient,
        is_data=src_var.is_data,
        belong_to_optimizer=src_var.belong_to_optimizer)

    # set dist attr uid
    # distributed_attr_uid = src_var.desc.get_distributed_attr_uid()
    # var.desc.set_distributed_attr_uid(distributed_attr_uid)
    dist_attr = copy.deepcopy(
        dist_context.get_tensor_dist_attr_for_program(src_var))
    assert dist_attr is not None
    dist_context.set_tensor_dist_attr_for_program(var, dist_attr)


def _partition_var(dist_context, src_block, dst_block, src_varname,
                   dst_varname):
    """
    partition include: split + replicate
    """
    src_var = src_block.var(src_varname)

    if src_var.type in __not_shape_var_type__:
        dst_block.create_var(
            type=src_var.type,
            name=dst_varname,
            persistable=True,
            stop_gradient=True)
        target_shape = None
    else:
        dist_attr = dist_context.get_tensor_dist_attr_for_program(src_var)
        target_shape = _get_dist_shape(src_var, dist_attr)

        if isinstance(src_var, Parameter):
            _partition_parameter(dist_context, src_var, dst_block, dst_varname,
                                 target_shape)
        else:
            _partition_intermediate_var(dist_context, src_var, dst_block,
                                        dst_varname, target_shape)
    return target_shape


def _get_dist_op_backward_implement(backward_op, dist_context,
                                    forward_op_id2forward_op):
    dist_op_context = dist_context.dist_op_context
    if backward_op.desc.id() in dist_op_context.grad_op_id_to_op_id:
        forward_op_id = dist_op_context.grad_op_id_to_op_id[backward_op.desc.id(
        )]
        forward_op = forward_op_id2forward_op[forward_op_id]
        forward_op_dist_attr = dist_context.get_op_dist_attr_for_program(
            forward_op)
        dist_op_impl_container = get_distributed_operator_impl_container(
            forward_op_dist_attr.impl_type)
        dist_op_impl = dist_op_impl_container.get_impl(
            forward_op_dist_attr.impl_idx)
        return dist_op_impl

    # # NOTE trick for dist ops that only have backward implement
    if backward_op.type in BACKWARD_ONLY_DIST_OPS:
        op_dist_attr = dist_context.get_op_dist_attr_for_program(backward_op)
        assert op_dist_attr.impl_idx >= 0
        dist_op_impl = get_distributed_operator_impl_container(
            op_dist_attr.impl_type).get_impl(op_dist_attr.impl_idx)
        return dist_op_impl

    dist_op = get_distributed_operator_impl_container("default")
    return dist_op.get_impl(0)


def _get_dist_op_forward_implement(forward_op, dist_context):
    dist_attr = dist_context.get_op_dist_attr_for_program(forward_op)
    dist_op_impl_container = get_distributed_operator_impl_container(
        dist_attr.impl_type)
    dist_op_impl = dist_op_impl_container.get_impl(dist_attr.impl_idx)
    return dist_op_impl
