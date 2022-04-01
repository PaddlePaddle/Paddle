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
from functools import reduce

import paddle
import paddle.fluid.core as core
from paddle.utils import unique_name
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Program, OpProtoHolder
from paddle.distributed.fleet.meta_optimizers.common import OpRole
import paddle.fluid.layers.utils as utils
from ..collective import _get_global_env
from .dist_context import DistributedContext
from .dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from .process_group import new_process_group, ProcessGroup, _g_process_group_map

# NOTE: If op in _g_special_ops, it will not be resharded. 
_g_special_ops = ['check_finite_and_unscale', 'update_loss_scaling']


def get_var_with_recursion(var_name, block, program):
    """Get var in the parent block if not found in the current block"""
    var = None
    if var_name in block.vars:
        var = block.vars[var_name]
    else:
        parent_block = program.blocks[block.parent_idx]
        if var_name in parent_block.vars:
            var = parent_block.vars[var_name]
    assert var is not None
    return var


class AllGatherOpDesc:
    """
    Describe the allgather op in the reshard phase.

    Args:
        group (list): Process group.
    """

    def __init__(self, group):
        self._group = group
        self._desc = "all_gather"

    @property
    def group(self):
        return self._group

    @property
    def desc(self):
        return self._desc

    def __repr__(self):
        return f"op: {self._desc}, group: {self._group}."


class SendOpDesc:
    """
    Describe the send op in the reshard phase.

    Args:
        partition_index (list): The index of partition in complete tensor.
        dst (int): The destination process to receive.
    """

    def __init__(self, partition_index, dst):
        self._dst = dst
        self._partition_index = partition_index
        self._desc = "send"

    @property
    def partition_index(self):
        return self._partition_index

    @property
    def dst(self):
        return self._dst

    @property
    def desc(self):
        return self._desc

    def __repr__(self):
        return f"op: {self._desc}, partition_index: {self._partition_index}, dst: {self._dst}."


class RecvOpDesc:
    """
    Describe the recv op in the reshard op.

    Args:
        partition_index (list): The index of partition in complete tensor.
        src (int): The source process to send.
    """

    def __init__(self, partition_index, src):
        self._src = src
        self._partition_index = partition_index
        self._desc = "recv"

    @property
    def partition_index(self):
        return self._partition_index

    @property
    def src(self):
        return self._src

    @property
    def desc(self):
        return self._desc

    def __repr__(self):
        return f"op: {self._desc}, partition_index: {self._partition_index}, src: {self._src}."


class SliceOpDesc:
    """
    Describe the slice op in the reshard phase.

    Args:
        starts (list): It represents starting indices of corresponding axis in ``axes``.
        ends (list):  It represents ending indices of corresponding axis in ``axes``.
        axes (list):  Axes that `starts` and `ends` apply to .
    """

    def __init__(self, starts, ends, axes):
        self._starts = starts
        self._ends = ends
        self._axes = axes
        self._desc = "slice"

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    @property
    def axes(self):
        return self._axes

    @property
    def desc(self):
        return self._desc

    def __repr__(self):
        return f"op: {self._desc}, starts: {self._starts}, ends: {self._ends}, axes: {self._axes}."


class ConcatOpDesc:
    """
    Describe the concat op in the reshard phase.

    Args:
        partition_index_list (list): The list contains all partition index.
    """

    def __init__(self, partition_index_list):
        self._partition_index_list = partition_index_list
        self._desc = "concat"

    @property
    def partition_index_list(self):
        return self._partition_index_list

    @property
    def desc(self):
        return self._desc

    def __repr__(self):
        return f"op: {self._desc}, partition_index_list: {self._partition_index_list}."


class Inserter:
    """Insert op required in the reshard process."""

    @staticmethod
    def insert_send_op(block, idx, tensor, dst, op_role):
        """Insert send op into block at the given index."""
        op_type = 'send_v2'
        block._insert_op(
            idx,
            type=op_type,
            inputs={'X': [tensor]},
            attrs={
                'ring_id': 0,
                'peer': dst,
                'use_calc_stream': True,
                'op_role': op_role
            })

    @staticmethod
    def insert_recv_op(block, idx, tensor, src, op_role):
        """Insert recv op into block at the given index."""
        op_type = 'recv_v2'
        block._insert_op(
            idx,
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [tensor]},
            attrs={
                'ring_id': 0,
                'peer': src,
                'out_shape': tensor.shape,
                'dtype': tensor.dtype,
                'use_calc_stream': True,
                'op_role': op_role
            })

    @staticmethod
    def insert_concat_op(block, idx, tensors, axis, op_role):
        """Insert concat op into block at the given block."""
        inputs = {'X': tensors}
        attrs = {}
        attrs['axis'] = axis
        attrs['op_role'] = op_role
        helper = LayerHelper('concat', **locals())
        with paddle.static.program_guard(block.program):
            out = helper.create_variable_for_type_inference(
                dtype=helper.input_dtype())
        block._insert_op(
            idx,
            type='concat',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs)
        return out

    @staticmethod
    def insert_slice_op(block, idx, tensor, starts, ends, axes, new_var_name,
                        op_role):
        """Insert slice op into block at the given block."""
        inputs = {'Input': tensor}
        infer_flags = list(1 for i in range(len(axes)))
        attrs = {
            "axes": axes,
            "starts": starts,
            "ends": ends,
            "infer_flags": infer_flags,
            'op_role': op_role
        }
        helper = LayerHelper('slice', **locals())
        out = block.create_var(
            name=new_var_name, dtype=tensor.dtype, type=tensor.type)
        block._insert_op(
            idx,
            type="slice",
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs)
        return out

    @staticmethod
    def insert_split_op(block, idx, tensor, num_or_sections, op_role):
        """Insert split op into block at the given index."""
        helper = LayerHelper('split', **locals())
        input_shape = tensor.shape
        inputs = {'X': tensor}
        attrs = {'num': num_or_sections, 'axis': 0, 'op_role': op_role}
        with paddle.static.program_guard(block.program):
            outs = [
                helper.create_variable_for_type_inference(
                    dtype=helper.input_dtype()) for i in range(num_or_sections)
            ]
        block._insert_op(
            idx,
            type="split",
            inputs=inputs,
            outputs={'Out': outs},
            attrs=attrs)
        return outs

    @staticmethod
    def insert_fill_constant_op(block, idx, op_role):
        """Insert fill constant op into block at the given index."""
        helper = LayerHelper("fill_constant", **locals())
        with paddle.static.program_guard(block.program):
            out = helper.create_variable_for_type_inference(dtype="int32")
        inputs = {}
        attrs = {'force_cpu': False}
        attrs['str_value'] = str(int("1"))
        attrs['value'] = int("1")
        attrs['dtype'] = out.dtype
        attrs['op_role'] = op_role
        utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=[0], op_type='fill_constant')
        block._insert_op(
            idx,
            type='fill_constant',
            inputs=inputs,
            outputs={'Out': [out]},
            attrs=attrs)
        out.stop_gradient = True
        return out

    @staticmethod
    def insert_allgather_op(block, idx, tensor, ranks, op_role):
        """Insert allgather op into block at the given index."""
        tensor_list = []
        group = new_process_group(ranks)
        idx_offset = 0

        # instant process group before insert allgather op.
        if not group.is_instantiate():
            # insert fill_constant op
            fill_constant_out = Inserter.insert_fill_constant_op(block, idx,
                                                                 op_role)
            fill_constant_out.stop_gradient = True

            # insert c_allreduce_sum op
            block._insert_op(
                idx + 1,
                type="c_allreduce_sum",
                inputs={'X': [fill_constant_out]},
                outputs={'Out': [fill_constant_out]},
                attrs={
                    'ring_id': 0,
                    'use_calc_stream': True,
                    'op_role': op_role
                })

            # insert c_sync_calc_stream op
            block._insert_op(
                idx + 2,
                type="c_sync_calc_stream",
                inputs={'X': [fill_constant_out]},
                outputs={'Out': [fill_constant_out]},
                attrs={'op_role': op_role})
            idx_offset = 3

        # insert c_allgather op
        op_type = 'c_allgather'
        helper = LayerHelper(op_type, **locals())
        with paddle.static.program_guard(block.program):
            allgather_out = helper.create_variable_for_type_inference(
                dtype=tensor.dtype)
        block._insert_op(
            idx + idx_offset,
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [allgather_out]},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'nranks': group.nranks,
                'op_role': op_role
            })
        idx_offset += 1

        # insert split op
        split_out = Inserter.insert_split_op(
            block, idx + idx_offset, allgather_out, group.nranks, op_role)
        idx_offset += 1
        tensor_list.extend(split_out)
        return tensor_list, idx_offset

    @staticmethod
    def concat_partitions_with_op(partition_tensor_list, tensor,
                                  partition_index, block, idx, op_role):
        """Concat the tensors and insert concat op."""
        if not partition_tensor_list:
            partition_tensor_list.append((tensor, partition_index))
        else:
            i = 0
            has_concat = False
            while i < len(partition_tensor_list):
                concat_axis, first_order, new_partition = Resharder.compute_concat_info(
                    partition_tensor_list[i][1], partition_index)
                if concat_axis != -1:
                    has_concat = True
                    _ = Inserter.insert_concat_op(block, idx[0], [partition_tensor_list[i][0], tensor], concat_axis, op_role) \
                        if first_order == 0 else \
                        Inserter.insert_concat_op(block, idx[0], [tensor, partition_tensor_list[i][0]], concat_axis, op_role)
                    partition_tensor_list.pop(i)
                    idx[0] += 1
                    Inserter.concat_partitions_with_op(partition_tensor_list, _,
                                                       new_partition, block,
                                                       idx, op_role)
                    break
                i += 1
            if not has_concat:
                partition_tensor_list.append((tensor, partition_index))


class Remover:
    """Remove var and op in the reshard process."""

    @staticmethod
    def remove_no_need_ops(auto_parallel_main_prog, dist_context, rank_id):
        """Remove no need ops in the main program"""
        not_remove_op_ref = [
            "create_py_reader", "create_double_buffer_reader", "read"
        ]

        # NOTE: The nested sub block is not be supported now.
        remove_block_order = []
        for block_idx in Resharder.while_block_info:
            remove_block_order.append(block_idx)

        for block_idx, block in enumerate(auto_parallel_main_prog.blocks):
            if block_idx not in remove_block_order:
                remove_block_order.append(block_idx)

        # the sub block should be removed first
        for block_idx in remove_block_order:
            remove_op_idx = []
            block = auto_parallel_main_prog.blocks[block_idx]
            ops = block.ops
            vars = block.vars
            for idx, op in enumerate(ops):
                if op.type == "read":
                    dim_list = []
                    for var_name in op.output_arg_names:
                        dim_list.extend(
                            get_var_with_recursion(
                                var_name, block, auto_parallel_main_prog).shape)
                    for i in range(idx, -1, -1):
                        if ops[i].type == "create_py_reader":
                            ops[i]._set_attr("shape_concat", dim_list)
                            break
                    continue

                # replace the input and output of c_sync_comm_stream op when in pipeline scene.
                if op.type == "c_sync_comm_stream":
                    need_save = []
                    for var_name in op.input_arg_names:
                        process_mesh = dist_context.get_tensor_dist_attr_for_program(
                            get_var_with_recursion(
                                var_name, block,
                                auto_parallel_main_prog)).process_mesh
                        if rank_id in process_mesh.processes:
                            need_save.append(var_name)
                    if not need_save:
                        remove_op_idx.append(idx)
                        continue

                    proto = OpProtoHolder.instance().get_op_proto(op.type)
                    op.desc.set_input(proto.inputs[0].name, need_save)
                    op.desc.set_output(proto.outputs[0].name, need_save)
                    continue

                # judge the other op whether should be removed.
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                if op_dist_attr is not None:
                    op_process_mesh = op_dist_attr.process_mesh
                    if rank_id not in op_process_mesh.processes and op.type not in not_remove_op_ref:
                        remove_op_idx.append(idx)

            for idx in remove_op_idx[::-1]:
                block._remove_op(idx)

    @staticmethod
    def remove_no_need_vars(auto_parallel_main_prog, dist_params_grads):
        """Remove no need vars in the main program"""
        for block_idx, block in enumerate(auto_parallel_main_prog.blocks):
            remove_vars = set()
            ops = block.ops
            vars = block.vars
            need_vars = set()
            for op in ops:
                for var_name in op.input_arg_names:
                    if var_name in vars:
                        need_vars.add(var_name)
                for var_name in op.output_arg_names:
                    if var_name in vars:
                        need_vars.add(var_name)
            for var in vars:
                if var not in need_vars:
                    remove_vars.add(var)

            # change dist_params_grads, the optimize op just in block 0.
            if block_idx == 0:
                param_grad_map = {}
                for op in ops:
                    if int(op.attr('op_role')) == int(OpRole.Optimize):
                        if "Param" in op.input_names and "Grad" in op.input_names:
                            param_name = op.input("Param")[0]
                            grad_name = op.input("Grad")[0]
                            param_grad_map[param_name] = grad_name

                need_remove_idx = []
                for idx, item in enumerate(dist_params_grads):
                    if item[0].name not in param_grad_map.keys():
                        need_remove_idx.append(idx)

                for idx in need_remove_idx[::-1]:
                    dist_params_grads.pop(idx)

                idx = 0
                while idx < len(dist_params_grads):
                    param_name = dist_params_grads[idx][0].name
                    grad_name = dist_params_grads[idx][1].name
                    if grad_name != param_grad_map[param_name]:
                        dist_params_grads[idx] = (
                            vars[param_name], vars[param_grad_map[param_name]])
                    idx += 1

            for var in remove_vars:
                block._remove_var(var)

    @staticmethod
    def remove_no_need_in_main(auto_parallel_main_prog, dist_context, rank_id,
                               dist_params_grads):
        """Remove no need vars and ops in the main program."""
        Remover.remove_no_need_ops(auto_parallel_main_prog, dist_context,
                                   rank_id)
        Resharder.change_while_op_input_and_output(auto_parallel_main_prog,
                                                   dist_context)
        Remover.remove_no_need_vars(auto_parallel_main_prog, dist_params_grads)

    @staticmethod
    def remove_no_need_in_startup(auto_parallel_main_prog,
                                  auto_parallel_startup_prog):
        """Remove no need vars and ops in the startup program."""
        main_input_vars = set()
        main_ops = auto_parallel_main_prog.global_block().ops
        for op in main_ops:
            for var_name in op.input_arg_names:
                main_input_vars.add(var_name)

        startup_block = auto_parallel_startup_prog.global_block()
        startup_output_vars = set()
        startup_ops = startup_block.ops
        for op in startup_ops:
            # skip c_sync_comm_stream op
            if op.type == "c_sync_comm_stream":
                continue
            for var_name in op.output_arg_names:
                startup_output_vars.add(var_name)

        need_vars = set()
        for var_name in startup_output_vars:
            if var_name in main_input_vars:
                need_vars.add(var_name)

        startup_ops = startup_block.ops
        actual_need_vars = set()
        for idx, op in enumerate(startup_ops):
            is_need_op = False
            if op.type == "c_sync_comm_stream":
                continue
            for var_name in op.output_arg_names:
                if var_name in need_vars:
                    is_need_op = True
                    break
            if is_need_op:
                for var_name in op.output_arg_names:
                    actual_need_vars.add(var_name)
                for var_name in op.input_arg_names:
                    actual_need_vars.add(var_name)

        remove_vars = set()
        for var_name in startup_block.vars:
            if var_name not in actual_need_vars:
                remove_vars.add(var_name)
        for var in remove_vars:
            startup_block._remove_var(var)

        remove_op_idx = []
        vars = startup_block.vars
        for idx, op in enumerate(startup_block.ops):
            is_no_need_op = False
            if op.type == "c_sync_comm_stream":
                var_names = []
                for var_name in op.input_arg_names:
                    if var_name in vars:
                        var_names.append(var_name)
                if not var_names:
                    remove_op_idx.append(idx)
                else:
                    proto = OpProtoHolder.instance().get_op_proto(op.type)
                    op.desc.set_input(proto.inputs[0].name, var_names)
                    op.desc.set_output(proto.outputs[0].name, var_names)
                continue

            for var_name in op.output_arg_names:
                if var_name not in vars:
                    is_no_need_op = True
                    break
            if is_no_need_op:
                remove_op_idx.append(idx)
        for idx in remove_op_idx[::-1]:
            startup_block._remove_op(idx)


class Resharder:
    """
    Reshard tensor in the program according to its distributed attribute and corresponding op distributed attribute.

    Args:
        auto_parallel_main_prog (Program): An auto parallel main program.
        auto_parallel_startup_prog (Program): An auto parallel startup program.
        rank_id (int): The process id.
        dist_context (DistributedContext): The distributed context of this rank.
        dist_params_grads (list): The list contains the tuple of param and grad.
        batch_size (int): The batch size. Default: None.
    """
    while_block_info = {}

    def __init__(self,
                 auto_parallel_main_prog,
                 auto_parallel_startup_prog,
                 rank_id,
                 dist_context,
                 dist_params_grads,
                 batch_size=None):
        assert isinstance(auto_parallel_main_prog, Program), "The type of auto_parallel_main_prog should be Program, " \
                                            "but got {}.".format(type(auto_parallel_main_prog))
        assert isinstance(auto_parallel_main_prog, Program), "The type of auto_parallel_startup_prog should be Program, " \
                                            "but got {}.".format(type(auto_parallel_startup_prog))
        assert isinstance(rank_id, int), "The type of rank_id should be int, " \
                                            "but got {}.".format(type(rank_id))
        assert isinstance(dist_context, DistributedContext), "The type of dist_context should be DistributedContext, " \
                                            "but got {}.".format(type(dist_context))
        if batch_size is not None:
            assert isinstance(batch_size, int), "The type of batch_size should be int, " \
                                                "but got {}.".format(type(batch_size))

        self._auto_parallel_main_prog = auto_parallel_main_prog
        self._auto_parallel_startup_prog = auto_parallel_startup_prog
        self._rank_id = rank_id
        self._dist_context = dist_context
        self._dist_params_grads = dist_params_grads
        self._batch_size = batch_size
        self._has_sent = {}
        self._has_recv = {}
        self._has_allgather = {}

    @property
    def auto_parallel_main_prog(self):
        return self._auto_parallel_main_prog

    @property
    def auto_parallel_startup_prog(self):
        return self._auto_parallel_startup_prog

    @property
    def rank_id(self):
        return self._rank_id

    @property
    def dist_context(self):
        return self._dist_context

    @property
    def dist_params_grads(self):
        return self._dist_params_grads

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_sent(self):
        return self._has_sent

    @property
    def has_recv(self):
        return self._has_recv

    @property
    def has_allgather(self):
        return self._has_allgather

    @staticmethod
    def compute_partition_shape(complete_shape, dims_mapping, process_shape):
        """Compute the shape of partition."""
        partition_shape = []
        for idx, item in enumerate(complete_shape):
            if dims_mapping[idx] == -1:
                partition_shape.append(item)
            else:
                partition_shape.append(item // process_shape[dims_mapping[idx]])

        return partition_shape

    @staticmethod
    def compute_process_index(process, process_group, process_shape):
        """Compute the index of process_shape corresponding to the process."""
        relative_process = process_group.index(process)
        process_index = []
        product = reduce(lambda x, y: x * y, process_shape)

        for i in range(len(process_shape)):
            idx = relative_process // (product // process_shape[i])
            product = product // process_shape[i]
            relative_process = relative_process - relative_process // product * product
            process_index.append(idx)

        return process_index

    @staticmethod
    def compute_partition_index(process, complete_shape, dims_mapping,
                                process_shape, process_group):
        """Compute the partition index in complete tensor."""
        partition_shape = Resharder.compute_partition_shape(
            complete_shape, dims_mapping, process_shape)
        process_index = Resharder.compute_process_index(process, process_group,
                                                        process_shape)
        partition_index = []

        for i in range(len(complete_shape)):
            if dims_mapping[i] == -1:
                partition_index.append([0, partition_shape[i]])
            else:
                partition_index.append([
                    process_index[dims_mapping[i]] * partition_shape[i],
                    (process_index[dims_mapping[i]] + 1) * partition_shape[i]
                ])

        return partition_index

    @staticmethod
    def compute_concat_info(partition_index_x, partition_index_y):
        """Judge whether two partition can be concatenated and compute concatenated partition index."""
        differ_count = 0
        concat_axis = -1
        first_order = 0
        new_partition = []

        for idx, item in enumerate(partition_index_x):
            if item != partition_index_y[idx]:
                differ_count += 1
                if item[1] == partition_index_y[idx][0] and item[
                        0] < partition_index_y[idx][1]:
                    concat_axis = idx
                    new_partition.append([item[0], partition_index_y[idx][1]])
                elif item[0] == partition_index_y[idx][1] and item[
                        1] > partition_index_y[idx][0]:
                    first_order = 1
                    concat_axis = idx
                    new_partition.append([partition_index_y[idx][0], item[1]])
            else:
                new_partition.append(item)

        if differ_count == 1:
            return concat_axis, first_order, new_partition
        else:
            return -1, first_order, new_partition

    @staticmethod
    def compute_complete_shape(slice_shape, process_shape, dims_mapping):
        """compute the complete shape of the slice tensor  with its process mesh and dims mapping"""
        complete_shape = []
        for idx, item in enumerate(slice_shape):
            if dims_mapping[idx] == -1:
                complete_shape.append(item)
            else:
                complete_shape.append(item * process_shape[dims_mapping[idx]])
        return complete_shape

    @staticmethod
    def concat_partitions(partition_index_list, partition_index):
        """Concat the given partitions without inserting concat op."""
        if not partition_index_list:
            partition_index_list.append(partition_index)
        else:
            i = 0
            has_concat = False
            while i < len(partition_index_list):
                concat_axis, _, new_partition = Resharder.compute_concat_info(
                    partition_index_list[i], partition_index)
                if concat_axis != -1:
                    has_concat = True
                    partition_index_list.pop(i)
                    Resharder.concat_partitions(partition_index_list,
                                                new_partition)
                    break
                i += 1
            if not has_concat:
                partition_index_list.append(partition_index)

    @staticmethod
    def change_while_op_input_and_output(auto_parallel_main_prog, dist_context):
        """Change while op input and output after the corresponding sub block ops removed"""
        for sub_block_idx in Resharder.while_block_info:
            sub_block = auto_parallel_main_prog.blocks[sub_block_idx]
            parent_while_op_id = Resharder.while_block_info[sub_block_idx][
                "op_id"]
            parent_block = auto_parallel_main_prog.blocks[sub_block.parent_idx]

            sub_block_op_inputs = set()
            sub_block_op_outputs = []
            for op in sub_block.ops:
                # skip the input and output of operators inserted in the reshard phase
                dist_op = dist_context.get_dist_op_for_program(op)
                if dist_op:
                    for var_name in op.output_arg_names:
                        if var_name not in sub_block_op_outputs:
                            sub_block_op_outputs.append(var_name)
                    for var_name in op.input_arg_names:
                        sub_block_op_inputs.add(var_name)

            # find the while op
            while_op = None
            for op in parent_block.ops:
                if op.desc.id() == parent_while_op_id and op.type == "while":
                    while_op = op
                    break

            assert while_op is not None

            # find the actual input and output of while op
            proto = OpProtoHolder.instance().get_op_proto(while_op.type)
            new_X = []
            for var_name in while_op.input("X"):
                if var_name in sub_block_op_inputs:
                    new_X.append(var_name)
            assert new_X
            while_op.desc.set_input(proto.inputs[0].name, new_X)

            new_Out = []
            for var_name in while_op.output("Out"):
                for output_name in sub_block_op_outputs[::-1]:
                    if output_name.find(var_name) != -1:
                        new_Out.append(output_name)
            assert new_Out
            while_op.desc.set_output(proto.outputs[0].name, new_Out)

    def is_overlapped(self, shape_x, shape_y):
        """Judge whether two partitions intersect on the specified dimension."""
        overlapped = False
        if (shape_y[0] <= shape_x[0] < shape_y[1]) or (
                shape_x[0] <= shape_y[0] < shape_x[1]):
            overlapped = True
        return overlapped

    def is_unshard(self, dims_mapping):
        for dim in dims_mapping:
            if dim != -1:
                return False
        return True

    def is_special_op(self, op):
        global _g_special_ops
        if op.type in _g_special_ops:
            return True
        return False

    def is_condition_replicative(self, op):
        assert op.type == "while"
        sub_block = self.auto_parallel_main_prog.blocks[op.attr("sub_block").id]
        dist_op = self.dist_context.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr

        # the dims mapping of condition tensor should be replicative
        for var_name in op.input("Condition"):
            var = get_var_with_recursion(var_name, sub_block,
                                         self.auto_parallel_main_prog)
            dist_tensor = self.dist_context.get_dist_tensor_for_program(var)
            tensor_dist_attr = dist_tensor.dist_attr
            var_dims_mapping = tensor_dist_attr.dims_mapping
            for dim in var_dims_mapping:
                if dim != -1:
                    return False

        return True

    def need_reshard(self,
                     dist_tensor,
                     dist_op,
                     actual_process_mesh,
                     op_input=True):
        """Judge the tensor whether needs to be resharded."""
        is_reshard = False
        tensor_dist_attr = dist_tensor.dist_attr
        tensor_name = dist_tensor.serial_tensor.name
        tensor_dims_mapping = tensor_dist_attr.dims_mapping
        tensor_process_mesh = tensor_dist_attr.process_mesh
        op_dist_attr = dist_op.dist_attr
        op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(tensor_name)
        op_process_mesh = actual_process_mesh
        if op_input:
            op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                tensor_name)
            if all(
                    map(lambda x: x is not None, [
                        tensor_dims_mapping, tensor_process_mesh,
                        op_input_dims_mapping, op_process_mesh
                    ])):
                # dims_mapping
                if tensor_dims_mapping != op_input_dims_mapping:
                    if dist_op.serial_op.type == "while":
                        sub_block = self.auto_parallel_main_prog.blocks[
                            dist_op.serial_op.attr("sub_block").id]
                        for op in sub_block.ops:
                            for var_name in op.input_arg_names:
                                if var_name == tensor_name:
                                    dist_op_attr = self.dist_context.get_dist_op_for_program(
                                        op).dist_attr
                                    var_dims_mapping = dist_op_attr.get_input_dims_mapping(
                                        var_name)
                                    if var_dims_mapping != tensor_dims_mapping:
                                        is_reshard = True
                                        break
                    else:
                        is_reshard = True
                # process_mesh
                if tensor_process_mesh != op_process_mesh:
                    # when processes length is not the same, the dims mapping must be replicative now
                    if len(tensor_process_mesh.processes) != len(
                            op_process_mesh.processes):
                        assert self.is_unshard(tensor_dims_mapping)
                        assert self.is_unshard(op_input_dims_mapping)
                    else:
                        if dist_tensor.serial_tensor.dtype == paddle.bool:
                            raise ValueError(
                                "Bool var is not supported reshard.")

                        # for while op, it should find the process mesh of op actually used the tensor as input
                        if dist_op.serial_op.type == "while":
                            sub_block = self.auto_parallel_main_prog.blocks[
                                dist_op.serial_op.attr("sub_block").id]
                            for op in sub_block.ops:
                                for var_name in op.input_arg_names:
                                    if var_name == tensor_name:
                                        dist_op_attr = self.dist_context.get_dist_op_for_program(
                                            op).dist_attr
                                        process_mesh = dist_op_attr.process_mesh
                                        if process_mesh == op_process_mesh:
                                            is_reshard = True
                                            break
                        else:
                            is_reshard = True
        else:
            op_output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                tensor_name)
            if all(
                    map(lambda x: x is not None, [
                        tensor_dims_mapping, tensor_process_mesh,
                        op_output_dims_mapping, op_process_mesh
                    ])):
                if tensor_process_mesh != op_process_mesh:
                    if dist_tensor.serial_tensor.dtype == paddle.bool:
                        raise ValueError("Bool var is not supported reshard.")
                    is_reshard = True
                if tensor_dims_mapping != op_output_dims_mapping:
                    raise ValueError(
                        "It is not supported that tensor dims mapping is different from op output dims mapping."
                    )

        return is_reshard

    def get_process_meshes(self, op):
        """Get all process meshes when op has sub block."""
        assert op.has_attr("sub_block")
        sub_block = self.auto_parallel_main_prog.blocks[op.attr("sub_block").id]
        ops = sub_block.ops
        op_process_mesh = self.dist_context.get_dist_op_for_program(
            op).dist_attr.process_mesh
        process_meshes = []
        for op in ops:
            dist_op = self.dist_context.get_dist_op_for_program(op)
            if not dist_op:
                continue
            process_mesh = dist_op.dist_attr.process_mesh
            if process_mesh not in process_meshes and process_mesh != op_process_mesh:
                process_meshes.append(process_mesh)

        if not process_meshes:
            process_meshes.append(op_process_mesh)

        return process_meshes

    def get_op_process_meshes(self, op):
        process_meshes = []
        dist_op = self.dist_context.get_dist_op_for_program(op)
        op_process_mesh = dist_op.dist_attr.process_mesh
        for process_mesh in self.dist_context.process_meshes:
            if set(process_mesh.processes) & (
                    set(op_process_mesh.processes)
            ) and len(process_mesh.processes) <= len(op_process_mesh.processes):
                process_meshes.append(process_mesh)

        # it means the process mesh is not a union when process meshes is null
        if not process_meshes:
            process_meshes.append(op_process_mesh)

        return process_meshes

    def get_while_op_actual_process_mesh(self, op):
        """Get the while op actual Process mesh corresponding to rank"""
        assert op.type == "while"
        while_op_process_mesh = self.dist_context.get_dist_op_for_program(
            op).dist_attr.process_mesh
        sub_block = self.auto_parallel_main_prog.blocks[op.attr("sub_block").id]
        ops = sub_block.ops
        actual_process_mesh = None
        for op in ops:
            dist_op = self.dist_context.get_dist_op_for_program(op)
            if not dist_op:
                continue
            process_mesh = dist_op.dist_attr.process_mesh
            if process_mesh == while_op_process_mesh:
                continue
            if self.rank_id in process_mesh.processes:
                raw_process_mesh = process_mesh
                break

        if actual_process_mesh is None and self.rank_id in while_op_process_mesh.processes:
            actual_process_mesh = while_op_process_mesh

        assert actual_process_mesh is not None
        return actual_process_mesh

    def find_op_desc_seq(self, dist_tensor, dist_op, actual_process_mesh):
        """
        Find the op description sequence to reshard the source tensor for matching the op requirement.

        Args:
            dist_tensor (DistributedTensor): A distributed tensor.
            dist_op (DistributedOperator): A distributed operator.
            actual_process_mesh (ProcessMesh): The actual op process mesh.

        Returns:
            Dict, the dict represents the required op description sequence corresponding to process, The key of dict is
            process and value is a list containing op description.
        """
        tensor_dist_attr = dist_tensor.dist_attr
        source_tensor = dist_tensor.serial_tensor
        tensor_name = source_tensor.name
        source_dims_mapping = tensor_dist_attr.dims_mapping
        source_process_mesh = tensor_dist_attr.process_mesh
        source_process_group = source_process_mesh.processes
        source_process_shape = source_process_mesh.topology

        op_dist_attr = dist_op.dist_attr
        target_process_mesh = actual_process_mesh
        target_dims_mapping = op_dist_attr.get_input_dims_mapping(tensor_name)
        target_process_group = target_process_mesh.processes
        target_process_shape = target_process_mesh.topology

        if source_tensor.shape[0] < 0:
            new_shape = list(source_tensor.shape)
            new_shape[0] = self.batch_size
            source_tensor.desc.set_shape(new_shape)

        complete_shape = Resharder.compute_complete_shape(
            source_tensor.shape, source_process_shape, source_dims_mapping)
        op_desc_seq = {}

        # TODO: if the target process group has the same process with source process group
        if set(target_process_group).intersection(set(
                source_process_group)) and set(target_process_group).difference(
                    set(source_process_group)):
            pass

        # in the different process group, it will use send, recv, concat and slice op
        elif target_process_group != source_process_group:
            partition_process_mapping_list = []
            for source_process in source_process_group:
                source_partition_index = Resharder.compute_partition_index(source_process, complete_shape, source_dims_mapping, \
                                                                source_process_shape, source_process_group)
                if not partition_process_mapping_list:
                    partition_process_mapping_list.append(
                        [source_partition_index, [source_process], [False]])
                else:
                    partition_list = list(
                        [item[0] for item in partition_process_mapping_list])
                    process_list = list(
                        [item[1] for item in partition_process_mapping_list])
                    has_used = list(
                        [item[2] for item in partition_process_mapping_list])
                    if partition_list.count(source_partition_index) == 1:
                        index = partition_list.index(source_partition_index)
                        process_list[index].append(source_process)
                        has_used[index].append(False)
                    else:
                        partition_process_mapping_list.append([
                            source_partition_index, [source_process], [False]
                        ])

            for target_process in target_process_group:
                has_sent = []
                target_partition_index = Resharder.compute_partition_index(
                    target_process, complete_shape, target_dims_mapping,
                    target_process_shape, target_process_group)
                partition_index_list = []
                all_partition_index_list = []
                for source_process in source_process_group:
                    source_partition_index = Resharder.compute_partition_index(
                        source_process, complete_shape, source_dims_mapping,
                        source_process_shape, source_process_group)
                    to_send_process = None
                    if all(_ for _ in list(map(self.is_overlapped, source_partition_index, target_partition_index))) \
                            and source_partition_index not in has_sent:
                        idx = list([
                            item[0] for item in partition_process_mapping_list
                        ]).index(source_partition_index)
                        has_used = list([
                            item[2] for item in partition_process_mapping_list
                        ])[idx]
                        process_list = list([
                            item[1] for item in partition_process_mapping_list
                        ])[idx]
                        i = 0
                        while i < len(has_used):
                            if not has_used[i]:
                                to_send_process = process_list[i]
                                has_used[i] = True
                                break
                            i += 1
                        if i == len(has_used):
                            has_used = list(map(lambda x: False, has_used))
                            to_send_process = process_list[0]
                            has_used[0] = True
                        assert to_send_process is not None, "Failed to find the send process."

                        if to_send_process not in op_desc_seq.keys():
                            op_desc_seq[to_send_process] = []
                        if target_process not in op_desc_seq.keys():
                            op_desc_seq[target_process] = []
                        all_partition_index_list.append(source_partition_index)

                        # append send and recv op desc
                        send_op_desc = SendOpDesc(source_partition_index,
                                                  target_process)
                        recv_op_desc = RecvOpDesc(source_partition_index,
                                                  to_send_process)
                        op_desc_seq[to_send_process].append(send_op_desc)
                        op_desc_seq[target_process].append(recv_op_desc)
                        has_sent.append(source_partition_index)
                        Resharder.concat_partitions(partition_index_list,
                                                    source_partition_index)

                # append concat op desc
                op_desc_seq[target_process].append(
                    ConcatOpDesc(all_partition_index_list))

                # append slice op desc
                slice_starts = []
                slice_ends = []
                slices_axes = []
                concatenated_partition_index = partition_index_list[0]
                for idx, item in enumerate(concatenated_partition_index):
                    slice_starts.append(target_partition_index[idx][0] - item[
                        0])
                    slice_ends.append(target_partition_index[idx][1] - item[0])
                    slices_axes.append(idx)
                op_desc_seq[target_process].append(
                    SliceOpDesc(slice_starts, slice_ends, slices_axes))

        # in the same process group, it will use allgahther and slice op
        else:
            partition_index_list = []
            all_partition_index_list = []
            process_index = []
            for source_process in source_process_group:
                source_partition_index = Resharder.compute_partition_index(
                    source_process, complete_shape, source_dims_mapping,
                    source_process_shape, source_process_group)
                if source_partition_index not in partition_index_list:
                    partition_index_list.append(source_partition_index)
                    process_index.append(
                        [[source_process, ], source_partition_index])
                else:
                    process_index[partition_index_list.index(
                        source_partition_index)][0].append(source_process)

            for i in range(len(process_index[0][0])):
                group = []
                for j in range(len(process_index)):
                    group.append(process_index[j][0][i])
                    if i == 0:
                        all_partition_index_list.append(process_index[j][1])
                for process in group:
                    # append slice op desc
                    slice_starts = []
                    slice_ends = []
                    slices_axes = []
                    target_partition_index = Resharder.compute_partition_index(
                        process, complete_shape, target_dims_mapping,
                        target_process_shape, target_process_group)
                    for idx, item in enumerate(target_partition_index):
                        slice_starts.append(item[0])
                        slice_ends.append(item[1])
                        slices_axes.append(idx)

                    slice_op_desc = SliceOpDesc(
                        starts=slice_starts, ends=slice_ends, axes=slices_axes)
                    op_desc_seq[process] = [AllGatherOpDesc(group=group),
                                            ConcatOpDesc(partition_index_list=all_partition_index_list), slice_op_desc] \
                        if len(group) > 1 else [slice_op_desc]

        return op_desc_seq

    def parse_op_desc(self, block, op_desc_seq, var_name, reshard_op,
                      actual_process_mesh):
        """Parse op desc sequence and insert op in the block"""
        tensor_list = []
        partition_tensor_list = []
        if self.rank_id not in op_desc_seq.keys():
            return
        op_desc_list = op_desc_seq[self.rank_id]

        idx = None
        for index, op in list(enumerate(block.ops)):
            if op.desc.id == reshard_op.desc.id:
                idx = index
                break
        assert idx is not None, "The op for reshard cannot be found in the rank {} program.".format(
            self.rank_id)

        matched_op = block.ops[idx]
        source_tensor = get_var_with_recursion(var_name, block,
                                               self.auto_parallel_main_prog)
        for op_desc in op_desc_list:
            if isinstance(op_desc, AllGatherOpDesc):  # noqa: F401
                if var_name not in self.has_allgather.keys():
                    self.has_allgather[var_name] = []
                if not self.has_allgather[
                        var_name] or op_desc.group not in list(
                            map(lambda x: x[0], self.has_allgather[var_name])):
                    tensor_list, idx_offset = Inserter.insert_allgather_op(
                        block, idx, source_tensor, op_desc.group,
                        reshard_op.attr('op_role'))
                    idx += idx_offset
                    tensor_name_list = [var.name for var in tensor_list]
                    self.has_allgather[var_name].append(
                        [op_desc.group, tensor_name_list])
                else:
                    for item in self.has_allgather[var_name]:
                        if op_desc.group == item[0]:
                            tensor_list = [
                                get_var_with_recursion(
                                    var_name, block,
                                    self.auto_parallel_main_prog)
                                for var_name in item[1]
                            ]
                            break
                assert tensor_list, "The result of parsing allgather op should not be None."

            elif isinstance(op_desc, SendOpDesc):
                if var_name not in self.has_sent.keys():
                    self.has_sent[var_name] = []
                if op_desc.dst not in self.has_sent[var_name]:
                    Inserter.insert_send_op(block, idx, source_tensor,
                                            op_desc.dst,
                                            reshard_op.attr('op_role'))
                    idx += 1
                    self.has_sent[var_name].append(op_desc.dst)

            elif isinstance(op_desc, RecvOpDesc):
                if var_name not in self.has_recv.keys():
                    self.has_recv[var_name] = {}
                if op_desc.src not in self.has_recv[var_name].keys():
                    partition_index = op_desc.partition_index
                    shape = []
                    for index in partition_index:
                        shape.append(index[1] - index[0])
                    recv_tensor = block.create_var(
                        name=unique_name.generate(var_name + "@recv"),
                        shape=shape,
                        dtype=source_tensor.dtype,
                        type=source_tensor.type)
                    Inserter.insert_recv_op(block, idx, recv_tensor,
                                            op_desc.src,
                                            reshard_op.attr('op_role'))
                    tensor_list.append(recv_tensor)
                    idx += 1
                    self.has_recv[var_name][op_desc.src] = recv_tensor
                else:
                    tensor_list.append(self.has_recv[var_name][op_desc.src])

            elif isinstance(op_desc, ConcatOpDesc):
                partition_index_list = op_desc.partition_index_list
                idx_list = [idx]
                for index, tensor in enumerate(tensor_list):
                    Inserter.concat_partitions_with_op(
                        partition_tensor_list, tensor,
                        partition_index_list[index], block, idx_list,
                        reshard_op.attr('op_role'))
                idx = idx_list[0]

            elif isinstance(op_desc, SliceOpDesc):
                assert len(
                    partition_tensor_list) == 1 or not partition_tensor_list
                to_slice_tensor = partition_tensor_list[0][0] if len(
                    partition_tensor_list) == 1 else source_tensor
                new_name = unique_name.generate(var_name + "@RESHARD")
                target_tensor = Inserter.insert_slice_op(
                    block,
                    idx,
                    to_slice_tensor,
                    starts=op_desc.starts,
                    ends=op_desc.ends,
                    axes=op_desc.axes,
                    new_var_name=new_name,
                    op_role=reshard_op.attr('op_role'))

                tensor_attr = TensorDistributedAttribute()
                process_mesh = actual_process_mesh
                dims_mapping = self.dist_context.get_op_dist_attr_for_program(
                    matched_op).get_input_dims_mapping(var_name)
                tensor_attr.dims_mapping = dims_mapping
                tensor_attr.process_mesh = process_mesh
                self.dist_context.set_tensor_dist_attr_for_program(
                    target_tensor, tensor_attr)

                if op.type == "while":
                    # var_reshard_mapping means the while op input need be changed to 
                    if "var_reshard_mapping" not in Resharder.while_block_info[
                            op.attr("sub_block").id].keys():
                        Resharder.while_block_info[op.attr("sub_block").id][
                            "var_reshard_mapping"] = {}
                    Resharder.while_block_info[op.attr("sub_block").id][
                        "var_reshard_mapping"][var_name] = target_tensor.name

                # rename op input name according to new name
                for op in block.ops:
                    for name in op.input_arg_names:
                        op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
                            op)
                        if name == var_name and op_dist_attr is not None:
                            if op.desc.id() == matched_op.desc.id():
                                op.desc._rename_input(name, target_tensor.name)
                                op_dist_attr.set_input_dims_mapping(
                                    target_tensor.name, dims_mapping)
                                op_dist_attr.set_input_dist_attr(name, None)
                                continue

                            # NOTE: For op whose process mesh is a union, its input will not be renamed by other op reshard result now which means that it will have more reshard operation.
                            op_process_mesh = op_dist_attr.process_mesh
                            op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                                var_name)
                            if op_process_mesh == process_mesh and op_input_dims_mapping == dims_mapping:
                                op.desc._rename_input(name, target_tensor.name)
                                op_dist_attr.set_input_dims_mapping(
                                    target_tensor.name, dims_mapping)
                                op_dist_attr.set_input_dist_attr(name, None)

    def reshard(self):
        for block_idx, block in enumerate(self.auto_parallel_main_prog.blocks):
            if block_idx in Resharder.while_block_info:
                if "var_reshard_mapping" in Resharder.while_block_info[
                        block_idx]:
                    var_reshard_mapping = Resharder.while_block_info[block_idx][
                        "var_reshard_mapping"]
                    for op in block.ops:
                        for var_name in op.input_arg_names:
                            if var_name in var_reshard_mapping:
                                op.desc._rename_input(
                                    var_name, var_reshard_mapping[var_name])
                                dist_op = self.dist_context.get_dist_op_for_program(
                                    op)
                                op_dist_attr = dist_op.dist_attr
                                if op_dist_attr.process_mesh == Resharder.while_block_info[
                                        block_idx]["actual_process_mesh"]:
                                    dims_mapping = op_dist_attr.get_input_dims_mapping(
                                        var_name)
                                    op_dist_attr.set_input_dims_mapping(
                                        var_reshard_mapping[var_name],
                                        dims_mapping)
                                    op_dist_attr.set_input_dist_attr(var_name,
                                                                     None)

                        # the outputs also need to be renamed when the output name is the same with input name
                        for var_name in op.output_arg_names:
                            if var_name in var_reshard_mapping:
                                op.desc._rename_output(
                                    var_name, var_reshard_mapping[var_name])
                                dist_op = self.dist_context.get_dist_op_for_program(
                                    op)
                                op_dist_attr = dist_op.dist_attr
                                if op_dist_attr.process_mesh == Resharder.while_block_info[
                                        block_idx]["actual_process_mesh"]:
                                    dims_mapping = op_dist_attr.get_output_dims_mapping(
                                        var_name)
                                    op_dist_attr.set_output_dims_mapping(
                                        var_reshard_mapping[var_name],
                                        dims_mapping)
                                    op_dist_attr.set_output_dist_attr(var_name,
                                                                      None)

            idx = 0
            while idx < len(block.ops):
                pre_op_count = len(block.ops)
                op = block.ops[idx]

                if self.is_special_op(op):
                    idx += 1
                    continue

                dist_op = self.dist_context.get_dist_op_for_program(op)
                if dist_op is not None:
                    process_meshes = []
                    if op.type == "while":
                        if not self.is_condition_replicative(op):
                            raise ValueError(
                                "Please check the condition due to the dims mapping is not replicative."
                            )
                        process_meshes = self.get_process_meshes(op)
                        assert process_meshes
                        if op.attr("sub_block"
                                   ).id not in Resharder.while_block_info:
                            Resharder.while_block_info[op.attr("sub_block")
                                                       .id] = {}
                        Resharder.while_block_info[op.attr("sub_block").id][
                            "op_id"] = op.desc.id()
                        Resharder.while_block_info[op.attr("sub_block").id][
                            "actual_process_mesh"] = self.get_while_op_actual_process_mesh(
                                op)
                    else:
                        process_meshes = self.get_op_process_meshes(op)
                    input_vars = None
                    if op.type == "while":
                        input_var_names = op.input("X")
                    else:
                        input_var_names = op.input_arg_names
                    idx_offset = 0
                    for var_name in op.input_arg_names:
                        # skip lod_tensor_blocking_queue_0
                        if var_name == "lod_tensor_blocking_queue_0":
                            continue
                        var = get_var_with_recursion(
                            var_name, block, self.auto_parallel_main_prog)
                        dist_tensor = self.dist_context.get_dist_tensor_for_program(
                            var)
                        for process_mesh in process_meshes:
                            if dist_tensor is not None and self.need_reshard(
                                    dist_tensor, dist_op, process_mesh):
                                reshard_op_desc = self.find_op_desc_seq(
                                    dist_tensor, dist_op, process_mesh)
                                self.parse_op_desc(block, reshard_op_desc,
                                                   var_name, op, process_mesh)
                                cur_op_count = len(block.ops)
                                idx_offset = idx_offset + cur_op_count - pre_op_count
                                pre_op_count = cur_op_count
                    idx = idx + idx_offset + 1
                else:
                    idx += 1

            # insert send and recv op if output process mesh is different from tensor process mesh
            idx = 0
            # skip reader and ops whose process mesh is union
            skip_ops = [
                "create_py_reader", "create_double_buffer_reader", "read",
                "while", "write_to_array", "read_from_array"
            ]
            global _g_special_ops
            skip_ops += _g_special_ops
            while idx < len(block.ops):
                pre_op_count = len(block.ops)
                op = block.ops[idx]
                dist_op = self.dist_context.get_dist_op_for_program(op)
                if dist_op is not None and op.type not in skip_ops:
                    for var_name in op.output_arg_names:
                        var = get_var_with_recursion(
                            var_name, block, self.auto_parallel_main_prog)
                        dist_tensor = self.dist_context.get_dist_tensor_for_program(
                            var)
                        process_mesh = dist_op.dist_attr.process_mesh
                        if dist_tensor is not None and self.need_reshard(
                                dist_tensor, dist_op, process_mesh, False):
                            for index, item in enumerate(
                                    dist_op.dist_attr.process_mesh.processes):
                                recv_rank = dist_tensor.dist_attr.process_mesh.processes[
                                    index]
                                if self.rank_id == item:
                                    Inserter.insert_send_op(block, idx + 1, var,
                                                            recv_rank,
                                                            op.attr('op_role'))
                                if self.rank_id == recv_rank:
                                    Inserter.insert_recv_op(block, idx + 1, var,
                                                            item,
                                                            op.attr('op_role'))
                            cur_op_count = len(block.ops)
                            idx_offset = idx_offset + cur_op_count - pre_op_count
                            pre_op_count = cur_op_count
                    idx = idx + idx_offset + 1
                else:
                    idx += 1

        # remove no need vars and ops in the main program
        Remover.remove_no_need_in_main(self.auto_parallel_main_prog,
                                       self.dist_context, self.rank_id,
                                       self.dist_params_grads)

        # remove no need vars and ops in the startip program
        Remover.remove_no_need_in_startup(self.auto_parallel_main_prog,
                                          self.auto_parallel_startup_prog)

        # reset some variable when remove operation ended
        Resharder.while_block_info = {}
