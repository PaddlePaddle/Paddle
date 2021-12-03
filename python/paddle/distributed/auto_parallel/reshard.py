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
import paddle.fluid.layers.utils as utils
from ..collective import _get_global_env
from .dist_context import DistributedContext
from .dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from .process_group import new_process_group, ProcessGroup, _g_process_group_map


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
        partition_index_list (list): A list contains all partition index.
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


def _compute_partition_shape(complete_shape, dims_mapping, process_shape):
    """Compute the shape of partition."""
    partition_shape = []
    for idx, item in enumerate(complete_shape):
        if dims_mapping[idx] == -1:
            partition_shape.append(item)
        else:
            partition_shape.append(item // process_shape[dims_mapping[idx]])

    return partition_shape


def _compute_process_index(process, process_group, process_shape):
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


def _compute_partition_index(process, complete_shape, dims_mapping,
                             process_shape, process_group):
    """Compute the partition index in complete tensor."""
    partition_shape = _compute_partition_shape(complete_shape, dims_mapping,
                                               process_shape)
    process_index = _compute_process_index(process, process_group,
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


def _compute_concat_info(partition_index_x, partition_index_y):
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


def _concat_partitions(partition_index_list, partition_index):
    """Concat the given partitions without inserting concat op."""
    if not partition_index_list:
        partition_index_list.append(partition_index)
    else:
        i = 0
        has_concat = False
        while i < len(partition_index_list):
            concat_axis, _, new_partition = _compute_concat_info(
                partition_index_list[i], partition_index)
            if concat_axis != -1:
                has_concat = True
                partition_index_list.pop(i)
                _concat_partitions(partition_index_list, new_partition)
                break
            i += 1
        if not has_concat:
            partition_index_list.append(partition_index)


def _is_overlapped(shape_x, shape_y):
    """Judge whether two partitions intersect on the specified dimension."""
    overlapped = False
    if (shape_y[0] <= shape_x[0] < shape_y[1]) or (
            shape_x[0] <= shape_y[0] < shape_x[1]):
        overlapped = True
    return overlapped


def _need_reshard(dist_tensor, dist_op):
    """Judge the tensor whether needs to be resharded."""
    is_reshard = False
    tensor_dist_attr = dist_tensor.dist_attr
    tensor_name = dist_tensor.serial_tensor.name
    tensor_dims_mapping = tensor_dist_attr.dims_mapping
    tensor_process_mesh = tensor_dist_attr.process_mesh
    op_dist_attr = dist_op.dist_attr
    op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(tensor_name)
    op_process_mesh = op_dist_attr.process_mesh
    if all(
            map(lambda x: x is not None, [
                tensor_dims_mapping, tensor_process_mesh, op_input_dims_mapping,
                op_process_mesh
            ])):
        if tensor_dims_mapping != op_input_dims_mapping or tensor_process_mesh != op_process_mesh:
            is_reshard = True
    return is_reshard


def _compute_complete_shape(slice_shape, process_shape, dims_mapping):
    """compute the complete shape of the slice tensor  with its process mesh and dims mapping"""
    complete_shape = []
    for idx, item in enumerate(slice_shape):
        if dims_mapping[idx] == -1:
            complete_shape.append(item)
        else:
            complete_shape.append(item * process_shape[dims_mapping[idx]])
    return complete_shape


def find_op_desc_seq(dist_tensor, dist_op):
    """
    Find the op description sequence to reshard the source tensor for matching the op requirement.

    Args:
        dist_tensor (DistributedTensor): A distributed tensor.
        dist_op (DistributedOperator): A distributed operator.

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
    target_process_mesh = op_dist_attr.process_mesh
    target_dims_mapping = op_dist_attr.get_input_dims_mapping(tensor_name)
    target_process_group = target_process_mesh.processes
    target_process_shape = target_process_mesh.topology

    complete_shape = _compute_complete_shape(
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
            source_partition_index = _compute_partition_index(source_process, complete_shape, source_dims_mapping, \
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
                    partition_process_mapping_list.append(
                        [source_partition_index, [source_process], [False]])

        for target_process in target_process_group:
            has_sent = []
            target_partition_index = _compute_partition_index(
                target_process, complete_shape, target_dims_mapping,
                target_process_shape, target_process_group)
            partition_index_list = []
            all_partition_index_list = []
            for source_process in source_process_group:
                source_partition_index = _compute_partition_index(
                    source_process, complete_shape, source_dims_mapping,
                    source_process_shape, source_process_group)
                to_send_process = None
                if all(_ for _ in list(map(_is_overlapped, source_partition_index, target_partition_index))) \
                        and source_partition_index not in has_sent:
                    idx = list([
                        item[0] for item in partition_process_mapping_list
                    ]).index(source_partition_index)
                    has_used = list(
                        [item[2]
                         for item in partition_process_mapping_list])[idx]
                    process_list = list(
                        [item[1]
                         for item in partition_process_mapping_list])[idx]
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
                    _concat_partitions(partition_index_list,
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
                slice_starts.append(target_partition_index[idx][0] - item[0])
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
            source_partition_index = _compute_partition_index(
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
                target_partition_index = _compute_partition_index(
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


def _insert_send_op(block, idx, tensor, dst):
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
        })


def _insert_recv_op(block, idx, tensor, src):
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
        })


def _insert_concat_op(block, idx, tensors, axis):
    """Insert concat op into block at the given block."""
    inputs = {'X': tensors}
    attrs = {}
    attrs['axis'] = axis
    helper = LayerHelper('concat', **locals())
    with paddle.static.program_guard(block.program):
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    block._insert_op(
        idx, type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out


def _insert_slice_op(block, idx, tensor, starts, ends, axes, new_var_name):
    """Insert slice op into block at the given block."""
    inputs = {'Input': tensor}
    infer_flags = list(1 for i in range(len(axes)))
    attrs = {
        "axes": axes,
        "starts": starts,
        "ends": ends,
        "infer_flags": infer_flags
    }
    helper = LayerHelper('slice', **locals())
    out = block.create_var(
        name=new_var_name,
        dtype=tensor.dtype,
        type=core.VarDesc.VarType.LOD_TENSOR)
    block._insert_op(
        idx, type="slice", inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out


def _insert_split_op(block, idx, tensor, num_or_sections):
    """Insert split op into block at the given index."""
    helper = LayerHelper('split', **locals())
    input_shape = tensor.shape
    inputs = {'X': tensor}
    attrs = {'num': num_or_sections, "axis": 0}
    with paddle.static.program_guard(block.program):
        outs = [
            helper.create_variable_for_type_inference(
                dtype=helper.input_dtype()) for i in range(num_or_sections)
        ]
    block._insert_op(
        idx, type="split", inputs=inputs, outputs={'Out': outs}, attrs=attrs)
    return outs


def _insert_allgather_op(block, idx, tensor, ranks):
    """Insert allgather op into block at the given index."""

    def _insert_fill_constant_op(block, idx):
        """Insert fill constant op into block at the given index."""
        helper = LayerHelper("fill_constant", **locals())
        with paddle.static.program_guard(block.program):
            out = helper.create_variable_for_type_inference(dtype="int32")
        inputs = {}
        attrs = {'force_cpu': False}
        attrs['str_value'] = str(int("1"))
        attrs['value'] = int("1")
        attrs['dtype'] = out.dtype
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

    tensor_list = []
    group = new_process_group(ranks)
    idx_offset = 0

    # instant process group before insert allgather op.
    if not group.is_instantiate():
        # insert fill_constant op
        fill_constant_out = _insert_fill_constant_op(block, idx)
        fill_constant_out.stop_gradient = True

        # insert c_allreduce_sum op
        block._insert_op(
            idx + 1,
            type="c_allreduce_sum",
            inputs={'X': [fill_constant_out]},
            outputs={'Out': [fill_constant_out]},
            attrs={'ring_id': 0,
                   'use_calc_stream': True})

        # insert c_sync_calc_stream op
        block._insert_op(
            idx + 2,
            type="c_sync_calc_stream",
            inputs={'X': [fill_constant_out]},
            outputs={'Out': [fill_constant_out]})
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
            'nranks': group.nranks
        })
    idx_offset += 1

    # insert split op
    split_out = _insert_split_op(block, idx + idx_offset, allgather_out,
                                 group.nranks)
    idx_offset += 1
    tensor_list.extend(split_out)
    return tensor_list, idx_offset


def _concat_partitions_with_op(partition_tensor_list, tensor, partition_index,
                               block, idx):
    """Concat the tensors and insert concat op."""
    if not partition_tensor_list:
        partition_tensor_list.append((tensor, partition_index))
    else:
        i = 0
        has_concat = False
        while i < len(partition_tensor_list):
            concat_axis, first_order, new_partition = _compute_concat_info(
                partition_tensor_list[i][1], partition_index)
            if concat_axis != -1:
                has_concat = True
                _ = _insert_concat_op(block, idx[0], [partition_tensor_list[i][0], tensor], concat_axis) \
                    if first_order == 0 else \
                    _insert_concat_op(block, idx[0], [tensor, partition_tensor_list[i][0]], concat_axis)
                partition_tensor_list.pop(i)
                idx[0] += 1
                _concat_partitions_with_op(partition_tensor_list, _,
                                           new_partition, block, idx)
                break
            i += 1
        if not has_concat:
            partition_tensor_list.append((tensor, partition_index))


HAS_SENT = {}
HAS_RECV = {}
HAS_ALLGATHER = {}


def parse_op_desc(program, rank_id, op_desc_seq, var_name, reshard_op,
                  dist_context):
    """Parse op desc sequence and insert op in the block"""
    global HAS_SENT
    global HAS_RECV
    global HAS_ALLGATHER
    tensor_list = []
    partition_tensor_list = []
    if rank_id not in op_desc_seq.keys():
        return
    op_desc_list = op_desc_seq[rank_id]
    block = program.global_block()
    assert var_name in block.vars.keys(
    ), "The {} cannot be found in the {} program.".format(var_name, rank_id)

    idx = None
    for index, op in list(enumerate(block.ops)):
        if op.desc.id == reshard_op.desc.id:
            idx = index
            break
    assert idx is not None, "The op for reshard cannot be found in the rank {} program.".format(
        rank_id)

    matched_op = block.ops[idx]
    source_tensor = block.vars[var_name]
    for op_desc in op_desc_list:
        if isinstance(op_desc, AllGatherOpDesc):  # noqa: F401
            if var_name not in HAS_ALLGATHER.keys():
                HAS_ALLGATHER[var_name] = []
            if not HAS_ALLGATHER[var_name] or op_desc.group not in list(
                    map(lambda x: x[0], HAS_ALLGATHER[var_name])):
                tensor_list, idx_offset = _insert_allgather_op(
                    block, idx, source_tensor, op_desc.group)
                idx += idx_offset
                tensor_name_list = [var.name for var in tensor_list]
                HAS_ALLGATHER[var_name].append(
                    [op_desc.group, tensor_name_list])
            else:
                for item in HAS_ALLGATHER[var_name]:
                    if op_desc.group == item[0]:
                        tensor_list = [
                            program.global_block().vars[var_name]
                            for var_name in item[1]
                        ]
                        break
            assert tensor_list, "The result of parsing allgather op should not be None."

        elif isinstance(op_desc, SendOpDesc):
            if var_name not in HAS_SENT.keys():
                HAS_SENT[var_name] = []
            if op_desc.dst not in HAS_SENT[var_name]:
                _insert_send_op(block, idx, source_tensor, op_desc.dst)
                idx += 1
                HAS_SENT[var_name].append(op_desc.dst)

        elif isinstance(op_desc, RecvOpDesc):
            if var_name not in HAS_RECV.keys():
                HAS_RECV[var_name] = {}
            if op_desc.src not in HAS_RECV[var_name].keys():
                partition_index = op_desc.partition_index
                shape = []
                for index in partition_index:
                    shape.append(index[1] - index[0])
                recv_tensor = block.create_var(
                    name=unique_name.generate(var_name + "@recv"),
                    shape=shape,
                    dtype=source_tensor.dtype)
                _insert_recv_op(block, idx, recv_tensor, op_desc.src)
                tensor_list.append(recv_tensor)
                idx += 1
                HAS_RECV[var_name][op_desc.src] = recv_tensor
            else:
                tensor_list.append(HAS_RECV[var_name][op_desc.src])

        elif isinstance(op_desc, ConcatOpDesc):
            partition_index_list = op_desc.partition_index_list
            idx_list = [idx]
            for index, tensor in enumerate(tensor_list):
                _concat_partitions_with_op(partition_tensor_list, tensor,
                                           partition_index_list[index], block,
                                           idx_list)
            idx = idx_list[0]

        elif isinstance(op_desc, SliceOpDesc):
            assert len(partition_tensor_list) == 1 or not partition_tensor_list
            to_slice_tensor = partition_tensor_list[0][0] if len(
                partition_tensor_list) == 1 else source_tensor
            new_name = unique_name.generate(var_name + "@RESHARD")
            target_tensor = _insert_slice_op(
                block,
                idx,
                to_slice_tensor,
                starts=op_desc.starts,
                ends=op_desc.ends,
                axes=op_desc.axes,
                new_var_name=new_name)

            tensor_attr = TensorDistributedAttribute()
            process_mesh = dist_context.get_op_dist_attr_for_program(
                matched_op).process_mesh
            dims_mapping = dist_context.get_op_dist_attr_for_program(
                matched_op).get_input_dims_mapping(var_name)
            tensor_attr.dims_mapping = dims_mapping
            tensor_attr.process_mesh = process_mesh
            dist_context.set_tensor_dist_attr_for_program(target_tensor,
                                                          tensor_attr)

            # rename op input name according to new name
            for op in block.ops:
                for name in op.input_arg_names:
                    op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                    if name == var_name and op_dist_attr is not None:
                        op_process_mesh = op_dist_attr.process_mesh
                        op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                            var_name)
                        if op_process_mesh == process_mesh and op_input_dims_mapping == dims_mapping:
                            op.desc._rename_input(name, target_tensor.name)
                            op_dist_attr.set_input_dims_mapping(
                                target_tensor.name, dims_mapping)
                            op_dist_attr.set_input_dist_attr(name, None)


def _remove_no_need_ops(auto_parallel_main_prog, dist_context, rank_id):
    """Remove no need ops in the main program"""
    not_remove_op_ref = [
        "create_py_reader", "create_double_buffer_reader", "read"
    ]
    remove_op_idx = []
    block = auto_parallel_main_prog.global_block()
    ops = block.ops
    vars = block.vars
    for idx, op in enumerate(ops):
        # handle read op in the pipeline scene specially, it will be removed in the future.
        if op.type == "read":
            dim_list = []
            for var_name in op.output_arg_names:
                dim_list.extend(vars[var_name].shape)
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
                    vars[var_name]).process_mesh
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


def _remove_no_need_vars(auto_parallel_main_prog):
    """Remove no need vars in the main program"""
    remove_vars = set()
    block = auto_parallel_main_prog.global_block()
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
    for var in remove_vars:
        block._remove_var(var)


def remove_no_need_in_main(auto_parallel_main_prog, dist_context, rank_id):
    """Remove no need vars and ops in the main program."""
    _remove_no_need_ops(auto_parallel_main_prog, dist_context, rank_id)
    _remove_no_need_vars(auto_parallel_main_prog)


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


def reshard(auto_parallel_main_prog, auto_parallel_startup_prog, rank_id,
            dist_context):
    """
    Reshard tensor in the program according to its dist attr and corresponding op dist attr.

    Args:
        auto_parallel_main_prog (Program): An auto parallel main program.
        auto_parallel_startup_prog (Program): An auto parallel startup program.
        rank_id (int): The process id.
    """
    assert isinstance(auto_parallel_main_prog, Program), "The type of auto_parallel_main_prog should be Program, " \
                                         "but got {}.".format(type(auto_parallel_main_prog))
    assert isinstance(auto_parallel_main_prog, Program), "The type of auto_parallel_startup_prog should be Program, " \
                                         "but got {}.".format(type(auto_parallel_startup_prog))
    assert isinstance(rank_id, int), "The type of rank_id should be int, " \
                                         "but got {}.".format(type(rank_id))
    assert isinstance(dist_context, DistributedContext), "The type of dist_context should be DistributedContext, " \
                                         "but got {}.".format(type(dist_context))

    block = auto_parallel_main_prog.global_block()
    idx = 0
    while idx < len(block.ops):
        pre_op_count = len(block.ops)
        op = block.ops[idx]
        dist_op = dist_context.get_dist_op_for_program(op)
        if dist_op is not None:
            idx_offset = 0
            for var_name in op.input_arg_names:
                # skip lod_tensor_blocking_queue_0
                if var_name == "lod_tensor_blocking_queue_0":
                    continue
                var = block.vars[var_name]
                dist_tensor = dist_context.get_dist_tensor_for_program(var)
                if dist_tensor is not None and _need_reshard(dist_tensor,
                                                             dist_op):
                    reshard_op_desc = find_op_desc_seq(dist_tensor, dist_op)
                    parse_op_desc(auto_parallel_main_prog, rank_id,
                                  reshard_op_desc, var_name, op, dist_context)
                    cur_op_count = len(block.ops)
                    idx_offset = idx_offset + cur_op_count - pre_op_count
                    pre_op_count = cur_op_count
            idx = idx + idx_offset + 1
        else:
            idx += 1

    # remove no need vars and ops in the main program
    remove_no_need_in_main(auto_parallel_main_prog, dist_context, rank_id)

    # remove no need vars and ops in the startip program
    remove_no_need_in_startup(auto_parallel_main_prog,
                              auto_parallel_startup_prog)
