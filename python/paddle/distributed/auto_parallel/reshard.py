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
from .cost import build_comm_desc, CommContext
from .cost import AllgatherOpCost, SendOpCost
from .cost import SliceOpCost, SplitOpCost, ConcatOpCost
from .cluster import Cluster
from .utils import print_program_with_dist_attr, _is_gradient_clip_op

# NOTE: If op in _g_special_ops or _g_gradient_clip_ops, it will not be resharded.
_g_special_ops = ['check_finite_and_unscale', 'update_loss_scaling']
_g_gradient_clip_ops = [
    "sum", "sqrt", "fill_constant", "elementwise_max", "elementwise_div"
]


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
        shape (list): The tensor shape.
        is_bool (bool): Whether allgather bool data. Default: False.
    """

    def __init__(self, group, shape, is_bool=False):
        self._group = group
        self._desc = "all_gather"
        self._shape = shape
        self._is_bool = is_bool

    @property
    def is_bool(self):
        return self._is_bool

    @property
    def group(self):
        return self._group

    @property
    def desc(self):
        return self._desc

    @property
    def shape(self):
        return self._shape

    def __repr__(self):
        return f"op: {self._desc}, group: {self._group}, shape: {self._shape}, is_bool: {self._is_bool}."


class SendOpDesc:
    """
    Describe the send op in the reshard phase.

    Args:
        partition_index (list): The index of partition in complete tensor.
        src (int): The source process to send.
        dst (int): The destination process to receive.
        is_bool (bool): Whether send bool data. Default: False.
    """

    def __init__(self, partition_index, src, dst, is_bool=False):
        self._dst = dst
        self._partition_index = partition_index
        self._desc = "send"
        self._shape = []
        self._is_bool = is_bool
        self._src = src

    @property
    def src(self):
        return self._src

    @property
    def is_bool(self):
        return self._is_bool

    @property
    def partition_index(self):
        return self._partition_index

    @property
    def dst(self):
        return self._dst

    @property
    def desc(self):
        return self._desc

    @property
    def shape(self):
        if not self._shape:
            for item in self.partition_index:
                self._shape.append(item[1] - item[0])
        return self._shape

    def __repr__(self):
        return f"op: {self._desc}, partition_index: {self._partition_index}, dst: {self._dst}, shape: {self._shape}, is_bool: {self._is_bool}."


class RecvOpDesc:
    """
    Describe the recv op in the reshard op.

    Args:
        partition_index (list): The index of partition in complete tensor.
        src (int): The source process to send.
        dst (int): The destination process to receive.
        is_bool (bool): Whether receive bool data. Default: False.
    """

    def __init__(self, partition_index, src, dst, is_bool=False):
        self._src = src
        self._partition_index = partition_index
        self._desc = "recv"
        self._shape = []
        self._is_bool = is_bool
        self._dst = dst

    @property
    def dst(self):
        return self._dst

    @property
    def is_bool(self):
        return self._is_bool

    @property
    def partition_index(self):
        return self._partition_index

    @property
    def src(self):
        return self._src

    @property
    def desc(self):
        return self._desc

    @property
    def shape(self):
        if not self._shape:
            for item in self.partition_index:
                self._shape.append(item[1] - item[0])
        return self._shape

    def __repr__(self):
        return f"op: {self._desc}, partition_index: {self._partition_index}, dst: {self._dst}, shape: {self._shape}, is_bool: {self._is_bool}."


class SliceOpDesc:
    """
    Describe the slice op in the reshard phase.

    Args:
        starts (list): It represents start indices of corresponding axis in ``axes``.
        ends (list):  It represents end indices of corresponding axis in ``axes``.
        axes (list):  Axes that `starts` and `ends` apply to.
        shape (list): The shape of the tensor to be sliced.
    """

    def __init__(self, starts, ends, axes, shape=None):
        self._starts = starts
        self._ends = ends
        self._axes = axes
        self._desc = "slice"
        self._shape = shape

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

    @property
    def shape(self):
        return self._shape

    def __repr__(self):
        if self._shape is not None:
            return f"op: {self._desc}, starts: {self._starts}, ends: {self._ends}, axes: {self._axes}, shape: {self._shape}."
        else:
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
    def insert_cast_op(block, idx, tensor, op_role, tensor_type):
        # to avoid name conflict with framework
        new_var_name = paddle.fluid.unique_name.generate_with_ignorable_key(
            ".".join(["cast@RESHARD", 'tmp']))
        out = block.create_var(name=new_var_name,
                               dtype=tensor_type,
                               type=tensor.type,
                               lod_level=tensor.lod_level)
        block._insert_op(idx,
                         type='cast',
                         inputs={'X': [tensor]},
                         outputs={'Out': [out]},
                         attrs={
                             'in_dtype': tensor.dtype,
                             'out_dtype': out.dtype,
                             'op_role': op_role
                         })
        return out

    @staticmethod
    def insert_send_op(block, idx, tensor, src, dst, op_role):
        """Insert send op into block at the given index."""
        op_type = 'send_v2'
        # use pair comm group
        process_group = new_process_group([src, dst])
        block._insert_op(idx,
                         type=op_type,
                         inputs={'X': [tensor]},
                         attrs={
                             'ring_id': process_group.id,
                             'peer': process_group.ranks.index(dst),
                             'use_calc_stream': True,
                             'op_role': op_role,
                             'dynamic_shape': True
                         })

    @staticmethod
    def insert_recv_op(block, idx, tensor, src, dst, op_role):
        """Insert recv op into block at the given index."""
        op_type = 'recv_v2'
        # use pair group
        process_group = new_process_group([src, dst])
        block._insert_op(idx,
                         type=op_type,
                         inputs={'X': [tensor]},
                         outputs={'Out': [tensor]},
                         attrs={
                             'ring_id': process_group.id,
                             'peer': process_group.ranks.index(src),
                             'out_shape': tensor.shape,
                             'dtype': tensor.dtype,
                             'use_calc_stream': True,
                             'op_role': op_role,
                             'dynamic_shape': True
                         })

    @staticmethod
    def insert_reset_lod_op(block, idx, X, Y, op_role):
        """Insert reset_lod op into block at the given index."""

        new_var_name = paddle.fluid.unique_name.generate_with_ignorable_key(
            ".".join(["reset_lod@RESHARD", 'tmp']))
        reset_lod_out = block.create_var(name=new_var_name,
                                         shape=X.shape,
                                         type=X.type,
                                         dtype=X.dtype,
                                         lod_level=X.lod_level)

        block._insert_op(idx,
                         type="lod_reset",
                         inputs={
                             'X': X,
                             'Y': Y
                         },
                         outputs={'Out': reset_lod_out},
                         attrs={'op_role': op_role})
        return reset_lod_out

    @staticmethod
    def insert_concat_op(block, idx, tensors, axis, op_role):
        """Insert concat op into block at the given block."""
        inputs = {'X': tensors}
        attrs = {}
        attrs['axis'] = axis
        attrs['op_role'] = op_role
        # to avoid name conflict with framework
        helper = LayerHelper('concat@RESHARD', **locals())
        with paddle.static.program_guard(block.program):
            out = block.create_var(
                name=paddle.fluid.unique_name.generate_with_ignorable_key(
                    ".".join([helper.name, 'tmp'])),
                dtype=tensors[0].dtype,
                shape=None,
                lod_level=tensors[0].lod_level,
                type=tensors[0].type,
                persistable=False,
                stop_gradient=False)
        block._insert_op(idx,
                         type='concat',
                         inputs=inputs,
                         outputs={'Out': [out]},
                         attrs=attrs)
        return out

    @staticmethod
    def insert_slice_op(block, idx, tensor, starts, ends, axes, new_var_name,
                        op_role):
        """Insert slice op into block at the given block."""
        # This is a hack to insert split op to get slice tensor
        # 1. [128, 128] => [64, 128]: split
        # 2. [128, 128] => [128, 128]: assign
        # 3. [128, 128] => [64, 64]: slice, it will replaced by multi split
        global_shape = tensor.shape
        slice_shape = [ends[i] - starts[i] for i in range(len(starts))]
        diff_dims = []
        for index, item in enumerate(slice_shape):
            if item != global_shape[index]:
                diff_dims.append(index)

        # use assign
        if len(diff_dims) == 0:
            out = block.create_var(name=new_var_name,
                                   dtype=tensor.dtype,
                                   type=tensor.type,
                                   shape=slice_shape,
                                   lod_level=tensor.lod_level)
            inputs = {'X': [tensor]}
            outputs = {"Out": [out]}
            attrs = {"in_place": False}
            block._insert_op(idx,
                             type="assign",
                             inputs=inputs,
                             outputs=outputs,
                             attrs=attrs)
            return out

        # use split once
        elif len(diff_dims) == 1:
            diff_dim = diff_dims[0]
            num_or_sections = global_shape[diff_dim] // slice_shape[diff_dim]
            axis = diff_dim
            cur_idx = starts[diff_dim] // slice_shape[diff_dim]
            input_shape = global_shape
            inputs = {'X': tensor}
            attrs = {'num': num_or_sections, 'axis': axis, 'op_role': op_role}
            new_shape = []
            for index, item in enumerate(tensor.shape):
                if index != axis:
                    new_shape.append(item)
                else:
                    new_shape.append(item // num_or_sections)
            with paddle.static.program_guard(block.program):
                outs = [
                    block.create_var(name=paddle.fluid.unique_name.
                                     generate_with_ignorable_key(".".join(
                                         ['split@RESHARD', 'tmp'])),
                                     dtype=tensor.dtype,
                                     shape=None,
                                     type=tensor.type,
                                     persistable=False,
                                     lod_level=tensor.lod_level,
                                     stop_gradient=False)
                    for i in range(num_or_sections)
                ]
                out = outs[cur_idx]
            op = block._insert_op(idx,
                                  type="split",
                                  inputs=inputs,
                                  outputs={'Out': outs},
                                  attrs=attrs)
            return out

        # use slice
        else:
            inputs = {'Input': tensor}
            infer_flags = list(1 for i in range(len(axes)))
            attrs = {
                "axes": axes,
                "starts": starts,
                "ends": ends,
                "infer_flags": infer_flags,
                'op_role': op_role
            }
            out = block.create_var(name=new_var_name,
                                   dtype=tensor.dtype,
                                   type=tensor.type,
                                   lod_level=tensor.lod_level)
            block._insert_op(idx,
                             type="slice",
                             inputs=inputs,
                             outputs={'Out': [out]},
                             attrs=attrs)

            return out

    @staticmethod
    def insert_split_op(block, idx, tensor, num_or_sections, op_role, axis=0):
        """Insert split op into block at the given index."""
        helper = LayerHelper('split@RESHARD', **locals())
        input_shape = tensor.shape
        inputs = {'X': tensor}
        attrs = {'num': num_or_sections, 'axis': axis, 'op_role': op_role}
        new_shape = []
        for index, item in enumerate(tensor.shape):
            if index != axis:
                new_shape.append(item)
            else:
                new_shape.append(item // num_or_sections)
        with paddle.static.program_guard(block.program):
            outs = [
                block.create_var(
                    name=paddle.fluid.unique_name.generate_with_ignorable_key(
                        ".".join([helper.name, 'tmp'])),
                    dtype=tensor.dtype,
                    shape=None,
                    lod_level=tensor.lod_level,
                    type=tensor.type,
                    persistable=False,
                    stop_gradient=False) for i in range(num_or_sections)
            ]
        block._insert_op(idx,
                         type="split",
                         inputs=inputs,
                         outputs={'Out': outs},
                         attrs=attrs)
        return outs

    @staticmethod
    def insert_fill_constant_op(block, idx, op_role):
        """Insert fill constant op into block at the given index."""
        # to avoid name conflict with framework
        helper = LayerHelper('fill_constant@RESHARD', **locals())
        # use paddle.int64 as dtype
        with paddle.static.program_guard(block.program):
            out = block.create_var(
                name=paddle.fluid.unique_name.generate_with_ignorable_key(
                    ".".join([helper.name, 'tmp'])),
                dtype=paddle.int64,
                shape=None,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
        inputs = {}
        attrs = {'force_cpu': False}
        attrs['str_value'] = str(int("1"))
        attrs['value'] = int("1")
        attrs['dtype'] = out.dtype
        attrs['op_role'] = op_role
        utils.get_shape_tensor_inputs(inputs=inputs,
                                      attrs=attrs,
                                      shape=[0],
                                      op_type='fill_constant')
        block._insert_op(idx,
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
            fill_constant_out = Inserter.insert_fill_constant_op(
                block, idx, op_role)
            fill_constant_out.stop_gradient = True

            # insert c_allreduce_sum op
            block._insert_op(idx + 1,
                             type="c_allreduce_sum",
                             inputs={'X': [fill_constant_out]},
                             outputs={'Out': [fill_constant_out]},
                             attrs={
                                 'ring_id': 0,
                                 'use_calc_stream': True,
                                 'op_role': op_role
                             })

            # insert c_sync_calc_stream op
            block._insert_op(idx + 2,
                             type="c_sync_calc_stream",
                             inputs={'X': [fill_constant_out]},
                             outputs={'Out': [fill_constant_out]},
                             attrs={'op_role': op_role})
            idx_offset = 3

        # insert c_allgather op
        op_type = 'c_allgather'
        # to avoid name conflict with framework
        helper = LayerHelper(op_type + "@RESHARD", **locals())
        with paddle.static.program_guard(block.program):
            allgather_out = block.create_var(
                name=paddle.fluid.unique_name.generate_with_ignorable_key(
                    ".".join([helper.name, 'tmp'])),
                dtype=tensor.dtype,
                shape=None,
                lod_level=tensor.lod_level,
                type=tensor.type,
                persistable=False,
                stop_gradient=False)
        block._insert_op(idx + idx_offset,
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
        split_out = Inserter.insert_split_op(block, idx + idx_offset,
                                             allgather_out, group.nranks,
                                             op_role)
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
    def remove_no_need_vars(auto_parallel_main_prog, dist_params_grads,
                            feed_var_names):
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
                if var in feed_var_names:
                    continue
                block._remove_var(var)

    @staticmethod
    def remove_no_need_in_main(auto_parallel_main_prog, dist_context, rank_id,
                               dist_params_grads):
        """Remove no need vars and ops in the main program."""
        Remover.remove_no_need_ops(auto_parallel_main_prog, dist_context,
                                   rank_id)
        Resharder.change_while_op_input_and_output(auto_parallel_main_prog,
                                                   dist_context)
        # 'feed_var_names' cannot be removed from auto_parallel_main_prog
        feed_var_names = []
        for var in sum(list(dist_context.serial_feed_vars.values()), []):
            feed_var_names.append(var.name)
        Remover.remove_no_need_vars(auto_parallel_main_prog, dist_params_grads,
                                    feed_var_names)

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
        if auto_parallel_startup_prog is not None:
            assert isinstance(auto_parallel_main_prog, Program), "The type of auto_parallel_startup_prog should be Program or None, " \
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
        # to avoid reshard repeatly
        self._has_resharded = {}

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
                if item[1] == partition_index_y[idx][
                        0] and item[0] < partition_index_y[idx][1]:
                    concat_axis = idx
                    new_partition.append([item[0], partition_index_y[idx][1]])
                elif item[0] == partition_index_y[idx][
                        1] and item[1] > partition_index_y[idx][0]:
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
                if dist_op or (op.type == "slice" and not dist_op) or (
                        op.type == "split"
                        and not dist_op) or (op.type == "assign"
                                             and not dist_op):
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

            if while_op is None:
                continue

            # find the actual input and output of while op
            proto = OpProtoHolder.instance().get_op_proto(while_op.type)
            new_X = []
            for var_name in while_op.input("X"):
                if var_name in sub_block_op_inputs:
                    new_X.append(var_name)
            assert new_X
            new_X.sort()
            while_op.desc.set_input(proto.inputs[0].name, new_X)

            new_Out = []
            for var_name in while_op.output("Out"):
                for output_name in sub_block_op_outputs[::-1]:
                    if output_name.find(var_name) != -1:
                        if output_name not in new_Out:
                            new_Out.append(output_name)
            assert new_Out
            while_op.desc.set_output(proto.outputs[0].name, new_Out)

    def is_overlapped(self, shape_x, shape_y):
        """Judge whether two partitions intersect on the specified dimension."""
        overlapped = False
        if (shape_y[0] <= shape_x[0] < shape_y[1]) or (shape_x[0] <= shape_y[0]
                                                       < shape_x[1]):
            overlapped = True
        return overlapped

    def is_unshard(self, dims_mapping):
        for dim in dims_mapping:
            if dim != -1:
                return False
        return True

    def is_special_op(self, op):
        global _g_special_ops, _g_gradient_clip_ops
        if op.type in _g_special_ops:
            return True
        if _is_gradient_clip_op(op) and op.type in _g_gradient_clip_ops:
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

    def need_reshard(self, dist_tensor, dist_attr, op_input=True, dist_op=None):
        """Judge the tensor whether needs to be resharded."""
        is_reshard = False
        tensor_dist_attr = dist_tensor.dist_attr
        tensor_dims_mapping = tensor_dist_attr.dims_mapping
        tensor_process_mesh = tensor_dist_attr.process_mesh

        # dist_attr is [process_mesh, dims_mapping] and process_mesh is not a union
        op_process_mesh = dist_attr[0]

        if op_input:
            op_input_dims_mapping = dist_attr[1]
            if all(
                    map(lambda x: x, [
                        tensor_dims_mapping, tensor_process_mesh,
                        op_input_dims_mapping, op_process_mesh
                    ])):
                # judge whether need reshard by dims_mapping
                if tensor_dims_mapping != op_input_dims_mapping:
                    if tensor_process_mesh not in self.dist_context.process_meshes:
                        # assert whether -1 when union.
                        for item in tensor_dims_mapping:
                            if item != -1:
                                raise ValueError(
                                    "The dim must be -1 when tensor process mesh is a union."
                                )
                        # tensor process_mesh: [0, 1, 2, 3], dims_mapping: [-1, -1]
                        # op process_mesh: [4, 5], dims_mapping: [0, -1]
                        # reshard is not supported such as above
                        if not is_reshard:
                            return is_reshard
                        else:
                            raise ValueError(
                                "it is not supported that tensor process mesh is a union and needs reshard."
                            )
                    is_reshard = True

                # judge whether need reshard by process_mesh
                if tensor_process_mesh != op_process_mesh:
                    is_reshard = True
        else:
            op_output_dims_mapping = dist_attr[1]
            if all(
                    map(lambda x: x, [
                        tensor_dims_mapping, tensor_process_mesh,
                        op_output_dims_mapping, op_process_mesh
                    ])):
                if tensor_dims_mapping != op_output_dims_mapping:
                    raise ValueError(
                        "It is not supported that tensor dims mapping is different from op output dims mapping."
                    )
                if tensor_process_mesh != op_process_mesh:
                    is_reshard = True

        return is_reshard

    def get_op_process_meshes(self, op):
        """Get sub process meshes of the given op if op process mesh is a union."""
        process_meshes = []
        dist_op = self.dist_context.get_dist_op_for_program(op)
        op_process_mesh = dist_op.dist_attr.process_mesh

        for process_mesh in self.dist_context.process_meshes:
            if set(process_mesh.processes) & (set(
                    op_process_mesh.processes)) and len(
                        process_mesh.processes) < len(
                            op_process_mesh.processes):
                process_meshes.append(process_mesh)

        # it means the process mesh is not a union when process meshes is null
        if not process_meshes:
            process_meshes.append(op_process_mesh)

        return process_meshes

    def find_op_desc_seq(self, dist_tensor, dist_attr, serial=False):
        """
        Find the op description sequence to reshard the source tensor for matching the op requirement.

        Args:
            dist_tensor (DistributedTensor): A distributed tensor.
            dist_attr (list): A list contains process_mesh and dims_mapping such as [process_mesh, dims_mapping].
            serial (bool): If serial is true, the dist tensor and dist op come from serial program. Otherwise, they come from auto program.

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

        target_process_mesh = dist_attr[0]
        target_dims_mapping = dist_attr[1]
        target_process_group = target_process_mesh.processes
        target_process_shape = target_process_mesh.topology

        if source_tensor.shape[0] < 0:
            assert source_tensor.shape[0] == -1
            new_shape = list(source_tensor.shape)
            new_shape[0] = self.batch_size
            source_tensor.desc.set_shape(new_shape)

        complete_shape = Resharder.compute_complete_shape(
            source_tensor.shape, source_process_shape,
            source_dims_mapping) if not serial else source_tensor.shape
        op_desc_seq = {}

        # TODO: if the target process group has the same process with source process group
        if set(target_process_group).intersection(set(
                source_process_group)) and set(target_process_group).difference(
                    set(source_process_group)):
            pass

        elif target_process_group != source_process_group:
            partition_process_mapping_list = []
            for source_process in source_process_group:
                # get partition index of source process
                source_partition_index = Resharder.compute_partition_index(source_process, complete_shape, source_dims_mapping, \
                                                                source_process_shape, source_process_group)
                if not partition_process_mapping_list:
                    # the item in partition_process_mapping_list is source_partition_index, which processes and whether has been used
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
                # has_sent means the source_partition_index has been sent to target_process
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
                        is_bool = (
                            dist_tensor.serial_tensor.dtype == paddle.bool)
                        send_op_desc = SendOpDesc(source_partition_index,
                                                  to_send_process,
                                                  target_process,
                                                  is_bool=is_bool)
                        recv_op_desc = RecvOpDesc(source_partition_index,
                                                  to_send_process,
                                                  target_process,
                                                  is_bool=is_bool)
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
                to_slice_tensor_shape = []

                for idx, item in enumerate(concatenated_partition_index):
                    slice_starts.append(target_partition_index[idx][0] -
                                        item[0])
                    slice_ends.append(target_partition_index[idx][1] - item[0])
                    slices_axes.append(idx)
                    to_slice_tensor_shape.append(item[1] - item[0])

                op_desc_seq[target_process].append(
                    SliceOpDesc(slice_starts,
                                slice_ends,
                                slices_axes,
                                shape=to_slice_tensor_shape))

        # in the same process group, it will use allgahther and slice op.
        else:
            # NOTE: It just supports even partition scene.
            partition_index_list = []
            all_partition_index_list = []
            process_index = []
            for source_process in source_process_group:
                source_partition_index = Resharder.compute_partition_index(
                    source_process, complete_shape, source_dims_mapping,
                    source_process_shape, source_process_group)
                if source_partition_index not in partition_index_list:
                    partition_index_list.append(source_partition_index)
                    process_index.append([[
                        source_process,
                    ], source_partition_index])
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

                    to_slice_tensor_shape = dist_tensor.global_sizes()
                    slice_op_desc = SliceOpDesc(starts=slice_starts,
                                                ends=slice_ends,
                                                axes=slices_axes,
                                                shape=to_slice_tensor_shape)
                    allgather_shape = None if not serial else dist_tensor.local_sizes(
                        rank=process)
                    op_desc_seq[process] = [AllGatherOpDesc(group=group, shape=allgather_shape, is_bool=(source_tensor.dtype == paddle.bool)),
                                            ConcatOpDesc(partition_index_list=all_partition_index_list), slice_op_desc] \
                        if len(group) > 1 else [slice_op_desc]

        return op_desc_seq

    def parse_op_desc(self, block, op_desc_seq, var_name, reshard_op,
                      dist_attr):
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
                if not self.has_allgather[var_name] or op_desc.group not in list(
                        map(lambda x: x[0], self.has_allgather[var_name])):
                    if op_desc.is_bool:
                        # for bool data allgather, cast to int64 -> allgather -> cast bool
                        out_cast = Inserter.insert_cast_op(
                            block, idx, source_tensor,
                            reshard_op.attr('op_role'), paddle.int64)
                        tensor_list, idx_offset = Inserter.insert_allgather_op(
                            block, idx + 1, out_cast, op_desc.group,
                            reshard_op.attr('op_role'))
                        idx += idx_offset
                        tensor_name_list = []
                        for var in tensor_list:
                            out_cast = Inserter.insert_cast_op(
                                block, idx, var, reshard_op.attr('op_role'),
                                paddle.bool)
                            tensor_name_list.append(out_cast.name)
                            idx += 1
                        self.has_allgather[var_name].append(
                            [op_desc.group, tensor_name_list])
                    else:
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
                    if op_desc.is_bool:
                        out_cast = Inserter.insert_cast_op(
                            block, idx, source_tensor,
                            reshard_op.attr('op_role'), paddle.int64)
                        Inserter.insert_send_op(block, idx + 1, out_cast,
                                                op_desc.src, op_desc.dst,
                                                reshard_op.attr('op_role'))
                        idx += 2
                    else:
                        Inserter.insert_send_op(block, idx, source_tensor,
                                                op_desc.src, op_desc.dst,
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
                    if op_desc.is_bool:
                        # for bool data, recv int64 -> cast to bool
                        recv_tensor = block.create_var(
                            name=unique_name.generate(var_name + "@recv"),
                            shape=shape,
                            lod_level=source_tensor.lod_level,
                            dtype=paddle.int64,
                            type=source_tensor.type)
                        Inserter.insert_recv_op(block, idx, recv_tensor,
                                                op_desc.src, op_desc.dst,
                                                reshard_op.attr('op_role'))
                        out_cast = Inserter.insert_cast_op(
                            block, idx + 1, recv_tensor,
                            reshard_op.attr('op_role'), paddle.bool)
                        tensor_list.append(out_cast)
                        idx += 2
                        self.has_recv[var_name][op_desc.src] = out_cast
                    else:
                        recv_tensor = block.create_var(
                            name=unique_name.generate(var_name + "@recv"),
                            shape=shape,
                            lod_level=source_tensor.lod_level,
                            dtype=source_tensor.dtype,
                            type=source_tensor.type)
                        Inserter.insert_recv_op(block, idx, recv_tensor,
                                                op_desc.src, op_desc.dst,
                                                reshard_op.attr('op_role'))

                        # for lod tensor, need reset lod after received
                        if recv_tensor.lod_level != 0:
                            set_lod = False
                            # use data lod to reset tensor lod
                            for tmp_block in self.auto_parallel_main_prog.blocks:
                                for tmp_var_name in tmp_block.vars:
                                    tmp_var = tmp_block.vars[tmp_var_name]
                                    if tmp_var.is_data and tmp_var.lod_level == recv_tensor.lod_level:
                                        reset_lod_out = Inserter.insert_reset_lod_op(
                                            block, idx + 1, recv_tensor,
                                            tmp_var, reshard_op.attr('op_role'))
                                        tensor_list.append(reset_lod_out)
                                        idx += 2
                                        self.has_recv[var_name][
                                            op_desc.src] = reset_lod_out
                                        set_lod = True
                                        break
                                if set_lod:
                                    break
                            assert set_lod is True
                        else:
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

                process_mesh = dist_attr[0]
                dims_mapping = dist_attr[1]

                tensor_attr = TensorDistributedAttribute()
                tensor_attr.dims_mapping = dims_mapping
                tensor_attr.process_mesh = process_mesh
                self.dist_context.set_tensor_dist_attr_for_program(
                    target_tensor, tensor_attr)

                if matched_op.type == "while":
                    # var_reshard_mapping means the while op input need be changed to
                    if "var_reshard_mapping" not in Resharder.while_block_info[
                            op.attr("sub_block").id].keys():
                        Resharder.while_block_info[op.attr(
                            "sub_block").id]["var_reshard_mapping"] = {}
                    if var_name not in Resharder.while_block_info[op.attr(
                            "sub_block").id]["var_reshard_mapping"].keys():
                        Resharder.while_block_info[op.attr("sub_block").id][
                            "var_reshard_mapping"][var_name] = []
                    Resharder.while_block_info[op.attr("sub_block").id][
                        "var_reshard_mapping"][var_name].append(
                            [dist_attr, target_tensor.name])

                # rename op input name according to new name
                for op in block.ops:
                    # just for while op
                    while_op_X_append = []
                    for name in op.input_arg_names:
                        op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
                            op)
                        if name == var_name and op_dist_attr is not None:
                            if op.desc.id() == matched_op.desc.id():
                                if matched_op.type == "while":
                                    old_name = name
                                    new_name = target_tensor.name
                                    assert old_name != new_name
                                    op_input_dist_attr = op_dist_attr.get_input_dist_attr(
                                        old_name)
                                    op_dist_attr.set_input_dist_attr(
                                        new_name, op_input_dist_attr)
                                    op_dist_attr.set_input_dims_mapping(
                                        new_name, dims_mapping)
                                    if old_name in op_dist_attr._inputs_dist_attrs:
                                        op_dist_attr.del_input_dist_attr(
                                            old_name)
                                    while_op_X_append.append(new_name)
                                    continue
                                else:
                                    op.desc._rename_input(
                                        name, target_tensor.name)
                                    old_name = name
                                    new_name = target_tensor.name
                                    assert old_name != new_name
                                    op_input_dist_attr = op_dist_attr.get_input_dist_attr(
                                        old_name)
                                    op_dist_attr.set_input_dist_attr(
                                        new_name, op_input_dist_attr)
                                    op_dist_attr.set_input_dims_mapping(
                                        new_name, dims_mapping)
                                    op_dist_attr.del_input_dist_attr(old_name)
                                    continue

                            op_process_mesh = op_dist_attr.process_mesh
                            op_input_dims_mapping = op_dist_attr.get_input_dims_mapping(
                                var_name)
                            # NOTE: For op whose process mesh is a union, its input will not be renamed by other op reshard result now which means that it will have more reshard operation.
                            if op_process_mesh == process_mesh and op_input_dims_mapping == dims_mapping:
                                op.desc._rename_input(name, target_tensor.name)
                                old_name = name
                                new_name = target_tensor.name
                                assert old_name != new_name
                                op_input_dist_attr = op_dist_attr.get_input_dist_attr(
                                    old_name)
                                op_dist_attr.set_input_dist_attr(
                                    new_name, op_input_dist_attr)
                                op_dist_attr.set_input_dims_mapping(
                                    new_name, dims_mapping)
                                op_dist_attr.del_input_dist_attr(old_name)

                    # for while op, the input X should reset
                    if while_op_X_append:
                        proto = OpProtoHolder.instance().get_op_proto(op.type)
                        op.desc.set_input(proto.inputs[0].name,
                                          op.input("X") + while_op_X_append)

    def _get_while_op_input_attrs(self, op, var_name):
        # NOTE: Multi while loop is not supported
        assert op.type == "while"
        sub_block = self.auto_parallel_main_prog.blocks[op.attr("sub_block").id]
        ops = sub_block.ops
        input_attrs = []

        for op in ops:
            dist_op = self.dist_context.get_dist_op_for_program(op)
            if not dist_op:
                continue
            dist_attr = dist_op.dist_attr
            for name in op.input_arg_names:
                if name == var_name:
                    process_mesh = dist_attr.process_mesh
                    input_dims_mapping = dist_attr.get_input_dims_mapping(
                        var_name)
                    has_exist = False
                    for input_attr in input_attrs:
                        if process_mesh == input_attr[
                                0] and input_dims_mapping == input_attr[1]:
                            has_exist = True
                            break
                    if not has_exist:
                        input_attrs.append([process_mesh, input_dims_mapping])
        return input_attrs

    def _get_common_op_input_attrs(self, op, var_name):
        process_meshes = []
        dist_op = self.dist_context.get_dist_op_for_program(op)
        dist_attr = dist_op.dist_attr
        op_process_mesh = dist_attr.process_mesh
        for process_mesh in self.dist_context.process_meshes:
            if set(process_mesh.processes) & (set(
                    op_process_mesh.processes)) and len(
                        process_mesh.processes) < len(
                            op_process_mesh.processes):
                process_meshes.append(process_mesh)

        # it means that the process mesh is not a union when process meshes is none
        if not process_meshes:
            process_meshes.append(op_process_mesh)

        input_dims_mapping = dist_attr.get_input_dims_mapping(var_name)
        input_attrs = []
        for process_mesh in process_meshes:
            input_attrs.append([process_mesh, input_dims_mapping])

        return input_attrs

    def get_op_input_attrs(self, op, var_name):
        op_input_attrs = []

        if op.type == "while":
            op_input_attrs = self._get_while_op_input_attrs(op, var_name)
        else:
            op_input_attrs = self._get_common_op_input_attrs(op, var_name)

        assert op_input_attrs

        return op_input_attrs

    def _remove_global_process_mesh(self):
        """Remove global process mesh from dist_context.process_meshes"""
        processes = set()
        process_mesh_count = len(self.dist_context.process_meshes)
        if process_mesh_count > 1:
            global_process_mesh_idx = None
            for process_mesh in self.dist_context.process_meshes:
                for process in process_mesh.processes:
                    processes.add(process)
            for idx, process_mesh in enumerate(
                    self.dist_context.process_meshes):
                if len(set(process_mesh.processes)) == len(processes):
                    global_process_mesh_idx = idx
                    break
            if global_process_mesh_idx is not None:
                self.dist_context.process_meshes.pop(idx)

    def _change_subblock_op_input_and_output(self, block_idx, block):
        if "var_reshard_mapping" in Resharder.while_block_info[block_idx]:
            var_reshard_mapping = Resharder.while_block_info[block_idx][
                "var_reshard_mapping"]
            for op in block.ops:
                for var_name in op.input_arg_names:
                    if var_name in var_reshard_mapping:
                        # in while sub block, the union process mesh is not split before reshard sub block
                        dist_op = self.dist_context.get_dist_op_for_program(op)
                        dist_attr = dist_op.dist_attr
                        target_name = None
                        for item in var_reshard_mapping[var_name]:
                            if dist_attr.process_mesh == item[0][
                                    0] and dist_attr.get_input_dims_mapping(
                                        var_name) == item[0][1]:
                                target_name = item[1]
                                break
                        if target_name is None:
                            continue
                        else:
                            op.desc._rename_input(var_name, target_name)
                            dist_op = self.dist_context.get_dist_op_for_program(
                                op)
                            op_dist_attr = dist_op.dist_attr
                            old_name = var_name
                            new_name = target_name
                            assert old_name != new_name
                            op_input_dist_attr = op_dist_attr.get_input_dist_attr(
                                old_name)
                            op_dist_attr.set_input_dist_attr(
                                new_name, op_input_dist_attr)
                            op_dist_attr.del_input_dist_attr(old_name)

                # the outputs also need to be renamed when the output name is the same with input name in inplace op
                for var_name in op.output_arg_names:
                    # if the tensor has been resharded multiply, it is not supported now.
                    if var_name in var_reshard_mapping:
                        if len(var_reshard_mapping[var_name]) > 1:
                            raise ValueError(
                                "The scene is not supported that the output is inplaced and the tensor has been resharded multiply when as input."
                            )
                        target_name = var_reshard_mapping[var_name][0][1]

                        op.desc._rename_output(var_name, target_name)
                        dist_op = self.dist_context.get_dist_op_for_program(op)
                        op_dist_attr = dist_op.dist_attr
                        old_name = var_name
                        new_name = target_name
                        assert old_name != new_name
                        op_output_dist_attr = op_dist_attr.get_output_dist_attr(
                            old_name)
                        op_dist_attr.set_output_dist_attr(
                            new_name, op_output_dist_attr)
                        op_dist_attr.del_output_dist_attr(old_name)

    def _reshard_input(self, block):
        idx = 0
        while idx < len(block.ops):
            pre_op_count = len(block.ops)
            op = block.ops[idx]

            if self.is_special_op(op):
                idx += 1
                continue

            dist_op = self.dist_context.get_dist_op_for_program(op)
            if dist_op is not None:
                op_input_dist_attrs = [
                ]  # [(op_process_mesh, op_input_dims_mapping), (op_process_mesh, op_input_dims_mapping)]
                if op.type == "while":
                    if not self.is_condition_replicative(op):
                        raise ValueError(
                            "Please check the condition due to the dims mapping is not replicative."
                        )
                    if op.attr(
                            "sub_block").id not in Resharder.while_block_info:
                        Resharder.while_block_info[op.attr("sub_block").id] = {}
                    Resharder.while_block_info[op.attr(
                        "sub_block").id]["op_id"] = op.desc.id()

                if op.type == "while":
                    # condition var process mesh is the same with op and dims_mapping is replicative, so it do not need reshard
                    input_var_names = op.input("X")
                else:
                    input_var_names = op.input_arg_names
                # to avoid while op X order different
                input_var_names.sort()

                idx_offset = 0
                for var_name in input_var_names:
                    # skip lod_tensor_blocking_queue_0
                    if var_name == "lod_tensor_blocking_queue_0":
                        continue
                    var = get_var_with_recursion(var_name, block,
                                                 self.auto_parallel_main_prog)
                    dist_tensor = self.dist_context.get_dist_tensor_for_program(
                        var)

                    # judge whether union tensor dims_mapping all -1
                    is_union_process_mesh_tensor = False
                    if dist_tensor.dist_attr.process_mesh not in self.dist_context.process_meshes and self.dist_context.process_meshes:
                        is_union_process_mesh_tensor = True
                        assert dist_tensor.dist_attr.dims_mapping.count(
                            -1) == len(dist_tensor.dist_attr.dims_mapping)

                    op_input_attrs = self.get_op_input_attrs(op, var_name)
                    for input_attr in op_input_attrs:
                        input_process_mesh = None

                        # deal with union tensor
                        if is_union_process_mesh_tensor:
                            # if op process mesh is subset of union tensor process mesh, need no reshard
                            if set(input_attr[0].processes) <= set(
                                    dist_tensor.dist_attr.process_mesh.processes
                            ):
                                continue

                        if dist_tensor is not None and self.need_reshard(
                                dist_tensor, input_attr):
                            reshard_op_desc = self.find_op_desc_seq(
                                dist_tensor, input_attr)
                            self.parse_op_desc(block, reshard_op_desc, var_name,
                                               op, input_attr)
                            cur_op_count = len(block.ops)
                            idx_offset = idx_offset + cur_op_count - pre_op_count
                            pre_op_count = cur_op_count
                idx = idx + idx_offset + 1
            else:
                idx += 1

    def _hadnle_recv(self, block, idx, var, op, send_rank, recv_rank):
        if self.rank_id == recv_rank:
            # if recv bool data, recv then cast
            if var.dtype == paddle.bool:
                recv_cast_out = block.create_var(
                    name=unique_name.generate(var.name + "@recv"),
                    shape=var.shape,
                    lod_level=var.lod_level,
                    dtype=paddle.int64,
                    type=var.type)
                Inserter.insert_recv_op(block, idx + 1,
                                        recv_cast_out, send_rank, recv_rank,
                                        op.attr('op_role'))
                reset_lod_out = None
                if var.lod_level != 0:
                    set_lod = False
                    for tmp_block in self.auto_parallel_main_prog.blocks:
                        for tmp_var_name in tmp_block.vars:
                            tmp_var = tmp_block.vars[tmp_var_name]
                            if tmp_var.is_data and tmp_var.lod_level == var.lod_level:
                                reset_lod_out = block.create_var(
                                    name=unique_name.generate(var.name +
                                                              "@RESETLOD"),
                                    shape=recv_cast_out.shape,
                                    type=recv_cast_out.type,
                                    dtype=recv_cast_out.dtype,
                                    lod_level=recv_cast_out.lod_level)
                                idx += 1
                                block._insert_op(
                                    idx,
                                    type="lod_reset",
                                    inputs={
                                        'X': recv_cast_out,
                                        'Y': tmp_var
                                    },
                                    outputs={'Out': reset_lod_out},
                                    attrs={'op_role': op.attr("op_role")})
                                set_lod = True
                                break
                        if set_lod:
                            break
                    assert set_lod is True

                # cast int64 to bool
                block._insert_op(idx + 2,
                                 type='cast',
                                 inputs={
                                     'X': [recv_cast_out] if
                                     reset_lod_out is None else [reset_lod_out]
                                 },
                                 outputs={'Out': [var]},
                                 attrs={
                                     'in_dtype': recv_cast_out.dtype,
                                     'out_dtype': var.dtype,
                                     'op_role': op.attr('op_role')
                                 })
            else:
                if var.lod_level != 0:
                    recv_out = block.create_var(
                        name=unique_name.generate(var.name + "@recv"),
                        shape=var.shape,
                        lod_level=var.lod_level,
                        dtype=var.int64,
                        type=var.type)
                    Inserter.insert_recv_op(block, idx + 1, recv_out, send_rank,
                                            recv_rank, op.attr('op_role'))
                    set_lod = False
                    for tmp_block in self.auto_parallel_main_prog.blocks:
                        for tmp_var_name in tmp_block.vars:
                            tmp_var = tmp_block.vars[tmp_var_name]
                            if tmp_var.is_data and tmp_var.lod_level == var.lod_level:
                                idx += 1
                                block._insert_op(
                                    idx,
                                    type="lod_reset",
                                    inputs={
                                        'X': recv_out,
                                        'Y': tmp_var
                                    },
                                    outputs={'Out': var},
                                    attrs={'op_role': op.attr("op_role")})
                                set_lod = True
                                break
                        if set_lod:
                            break
                    assert set_lod is True
                else:
                    Inserter.insert_recv_op(block, idx + 1, var, send_rank,
                                            recv_rank, op.attr('op_role'))

    def _handle_send(self, block, idx, var, op, send_rank, recv_rank):
        if var.dtype == paddle.bool:
            cast_out = Inserter.insert_cast_op(block, idx + 1, var,
                                               op.attr('op_role'), paddle.int64)
            Inserter.insert_send_op(block, idx + 2, cast_out, send_rank,
                                    recv_rank, op.attr('op_role'))
        else:
            Inserter.insert_send_op(block, idx + 1, var, send_rank, recv_rank,
                                    op.attr('op_role'))

    def _reshard_output(self, block):
        # insert send and recv op if output process mesh is different from tensor process mesh
        idx = 0
        # skip reader and ops whose process mesh is union
        skip_ops = [
            "create_py_reader", "create_double_buffer_reader", "read", "while",
            "write_to_array", "read_from_array"
        ]
        global _g_special_ops
        skip_ops += _g_special_ops
        while idx < len(block.ops):
            pre_op_count = len(block.ops)
            op = block.ops[idx]
            dist_op = self.dist_context.get_dist_op_for_program(op)
            if dist_op is not None and op.type not in skip_ops:
                idx_offset = 0
                for var_name in op.output_arg_names:
                    var = get_var_with_recursion(var_name, block,
                                                 self.auto_parallel_main_prog)
                    dist_tensor = self.dist_context.get_dist_tensor_for_program(
                        var)
                    tensor_process_mesh = dist_tensor.dist_attr.process_mesh
                    output_attr = [
                        dist_op.dist_attr.process_mesh,
                        dist_op.dist_attr.get_output_dims_mapping(var_name)
                    ]
                    if dist_tensor is not None and self.need_reshard(
                            dist_tensor, output_attr, False):
                        tensor_processes = set(
                            tensor_process_mesh.processes) - (
                                set(tensor_process_mesh.processes)
                                & set(output_attr[0].processes))
                        if tensor_processes:
                            if len(tensor_processes) != len(
                                    output_attr[0].processes):
                                if dist_tensor.dist_attr.dims_mapping.count(
                                        -1) != len(
                                            dist_tensor.dist_attr.dims_mapping
                                        ) or output_attr[1].count(-1) != len(
                                            output_attr[1]):
                                    raise ValueError(
                                        "The dims_mapping must be -1")
                                else:
                                    for index, tensor_process in enumerate(
                                            tensor_processes):
                                        recv_rank = tensor_process
                                        actual_index = index
                                        if index >= len(
                                                output_attr[0].processes):
                                            actual_index = (
                                                index -
                                                len(output_attr[0].processes)
                                            ) % len(output_attr[0].processes)
                                        item = output_attr[0].processes[
                                            actual_index]
                                        if recv_rank == item:
                                            continue
                                        if self.rank_id == item:
                                            # if send bool data, cast then send
                                            self._handle_send(
                                                block, idx, var, op, item,
                                                recv_rank)
                                        if self.rank_id == recv_rank:
                                            # if recv bool data, recv then cast
                                            self._hadnle_recv(
                                                block, idx, var, op, item,
                                                recv_rank)
                            else:
                                for index, tensor_process in enumerate(
                                        tensor_processes):
                                    recv_rank = tensor_process
                                    item = output_attr[0].processes[index]
                                    if recv_rank == item:
                                        continue
                                    if self.rank_id == item:
                                        # if send bool data, cast then send
                                        self._handle_send(
                                            block, idx, var, op, item,
                                            recv_rank)
                                    if self.rank_id == recv_rank:
                                        # if recv bool data, recv then cast
                                        self._hadnle_recv(
                                            block, idx, var, op, item,
                                            recv_rank)

                            cur_op_count = len(block.ops)
                            idx_offset = idx_offset + cur_op_count - pre_op_count
                            pre_op_count = cur_op_count

                idx = idx + idx_offset + 1
            else:
                idx += 1

    def reshard(self):
        self._remove_global_process_mesh()
        for block_idx, block in enumerate(self.auto_parallel_main_prog.blocks):
            # change the var_name before resharding sub block
            if block_idx in Resharder.while_block_info:
                self._change_subblock_op_input_and_output(block_idx, block)

            # reshard input
            self._reshard_input(block)

            # reshard output
            # NOTE: Only support that insert send and recv op if output process mesh is different from tensor process mesh
            self._reshard_output(block)

        # remove no need vars and ops in the main program
        Remover.remove_no_need_in_main(self.auto_parallel_main_prog,
                                       self.dist_context, self.rank_id,
                                       self.dist_params_grads)

        # remove no need vars and ops in the startip program
        Remover.remove_no_need_in_startup(self.auto_parallel_main_prog,
                                          self.auto_parallel_startup_prog)

        # reset some variable when remove operation ended
        Resharder.while_block_info = {}

    def get_cost(self, op, tensor, cluster):
        # NOTE: The program should be the serial_program which is not been parted
        global _g_special_ops
        not_supported_op_type = _g_special_ops + ["while"]
        reshard_op_cost = None
        if op.type in not_supported_op_type:
            return reshard_op_cost
        else:
            tensor_name = tensor.name
            if tensor_name == "lod_tensor_blocking_queue_0":
                return reshard_op_cost
            else:
                dist_tensor = self.dist_context.get_dist_tensor_for_program(
                    tensor)
                # simplified processing: ignore union process mesh and output reshard
                dist_op = self.dist_context.get_dist_op_for_program(op)
                dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                    tensor.name)
                process_mesh = dist_op.dist_attr.process_mesh
                dist_attr = [process_mesh, dims_mapping]
                if dist_tensor is not None and self.need_reshard(
                        dist_tensor, dist_attr):
                    if tensor_name not in self._has_resharded:
                        self._has_resharded[tensor_name] = [dist_op]
                    else:
                        for item in self._has_resharded[tensor_name]:
                            item_dist_attr = item.dist_attr
                            item_dims_mapping = item_dist_attr.get_input_dims_mapping(
                                tensor_name)
                            item_process_mesh = item_dist_attr.process_mesh
                            if dims_mapping == item_dims_mapping and item_process_mesh == process_mesh:
                                return reshard_op_cost
                        self._has_resharded[tensor_name].append(dist_op)

                    reshard_op_desc = self.find_op_desc_seq(dist_tensor,
                                                            dist_attr,
                                                            serial=True)
                    dtype = dist_tensor.serial_tensor.dtype
                    reshard_op_cost = self.parse_op_desc_for_cost(
                        reshard_op_desc, dtype, cluster)

        return reshard_op_cost

    def _concat_partitions_for_cost(self, partition_tensor_list,
                                    partition_index, dtype, rank_id,
                                    local_rank_comp_cost, cluster):
        if not partition_tensor_list:
            partition_tensor_list.append(partition_index)
        else:
            i = 0
            has_concat = False
            while i < len(partition_tensor_list):
                concat_axis, first_order, new_partition = Resharder.compute_concat_info(
                    partition_tensor_list[i], partition_index)
                if concat_axis != -1:
                    has_concat = True
                    concat_desc = {}
                    concat_desc["op"] = "concat"
                    concat_desc["attrs"] = {"axis": concat_axis}
                    if first_order == 0:
                        concat_desc["inputs"] = {
                            "X": [(dtype, partition_tensor_list[i]),
                                  (dtype, partition_index)]
                        }
                    else:
                        concat_desc["inputs"] = {
                            "X": [(dtype, partition_index),
                                  (dtype, partition_tensor_list[i])]
                        }
                    partition_tensor_list.pop(i)
                    if rank_id not in local_rank_comp_cost:
                        local_rank_comp_cost[rank_id] = []
                    local_rank_comp_cost[rank_id].append(
                        ConcatOpCost(op_desc=concat_desc, cluster=cluster))
                    self._concat_partitions_for_cost(partition_tensor_list,
                                                     new_partition, dtype,
                                                     rank_id,
                                                     local_rank_comp_cost,
                                                     cluster)
                    break
                i += 1
            if not has_concat:
                partition_tensor_list.append(partition_index)

    def parse_op_desc_for_cost(self, reshard_op_desc, dtype, cluster):

        def _get_idx(comm_ranks, group_ranks):
            res, is_the_same = None, False
            idx = 0
            while idx < len(comm_ranks):
                if comm_ranks[idx] == set(group_ranks):
                    is_the_same = True

                for rank in group_ranks:
                    if rank in comm_ranks[idx]:
                        res = idx
                        comm_ranks[idx].add(rank)
                if res is None:
                    idx += 1
                else:
                    break
            return res, is_the_same

        comm_context = CommContext(cluster)
        # run communication op before computation op
        # TODO: Communication cost is not calculated when the var has been transfered by the same group in the past
        comm_costs = []
        comm_ranks = []
        local_rank_comp_cost = {}
        for key in reshard_op_desc:
            partition_tensor_list = []
            op_desc_list = reshard_op_desc[key]
            for op_desc in op_desc_list:
                if isinstance(op_desc, SendOpDesc):
                    group_ranks = [key, op_desc.dst]
                    shape = op_desc.shape
                    send_desc = build_comm_desc("send_v2", group_ranks, dtype,
                                                shape)
                    idx, is_the_same = _get_idx(comm_ranks, group_ranks)
                    if idx is None:
                        comm_costs.append([
                            (group_ranks,
                             SendOpCost(op_desc=send_desc,
                                        comm_context=comm_context))
                        ])
                        comm_ranks.append(set(group_ranks))
                    else:
                        if not is_the_same:
                            comm_costs[idx].append(
                                (group_ranks,
                                 SendOpCost(op_desc=send_desc,
                                            comm_context=comm_context)))
                elif isinstance(op_desc, AllGatherOpDesc):
                    # NOTE: fill_const and other unnecessary op is not calculated because those cost is very small
                    group_ranks = op_desc.group
                    shape = op_desc.shape
                    allgather_desc = build_comm_desc("c_allgather", group_ranks,
                                                     dtype, shape)
                    split_inputs_shape = []
                    for idx, dim in enumerate(shape):
                        if idx == 0:
                            split_inputs_shape.append(dim * len(group_ranks))
                        else:
                            split_inputs_shape.append(dim)
                    idx, is_the_same = _get_idx(comm_ranks, group_ranks)
                    if idx is None:
                        comm_costs.append([
                            (group_ranks,
                             AllgatherOpCost(op_desc=allgather_desc,
                                             comm_context=comm_context))
                        ])
                        comm_ranks.append(set(group_ranks))
                    else:
                        if not is_the_same:
                            comm_costs[idx].append(
                                (group_ranks,
                                 AllgatherOpCost(op_desc=allgather_desc,
                                                 comm_context=comm_context)))
                    # calc the split op cost
                    if key not in local_rank_comp_cost:
                        local_rank_comp_cost[key] = []
                    split_desc = {}
                    split_desc["op"] = "split"
                    split_desc["inputs"] = {
                        "inputs": [(dtype, split_inputs_shape)]
                    }
                    split_desc["attrs"] = {"num": len(group_ranks), "axis": 0}
                    local_rank_comp_cost[key].append(
                        SplitOpCost(op_desc=split_desc, cluster=cluster))
                elif isinstance(op_desc, ConcatOpDesc):
                    partition_index_list = op_desc._partition_index_list
                    for idx, partion_idex in enumerate(partition_index_list):
                        self._concat_partitions_for_cost(
                            partition_tensor_list, partion_idex, dtype, key,
                            local_rank_comp_cost, cluster)

                elif isinstance(op_desc, SliceOpDesc):
                    if key not in local_rank_comp_cost:
                        local_rank_comp_cost[key] = []
                    assert len(
                        partition_tensor_list) == 1 or not partition_tensor_list
                    to_slice_tensor_shape = []
                    if len(partition_tensor_list) == 1:
                        for item in partition_tensor_list[0]:
                            to_slice_tensor_shape.append(item[1] - item[0])
                    else:
                        to_slice_tensor_shape = op_desc.shape
                    slice_desc = {}
                    slice_desc["op"] = "slice"
                    infer_flags = list(1 for i in range(len(op_desc.axes)))
                    slice_desc["attrs"] = {
                        "axes": op_desc.axes,
                        "starts": op_desc.starts,
                        "ends": op_desc.ends,
                        "infer_flags": infer_flags
                    }
                    slice_desc["inputs"] = {
                        "Input": [(dtype, to_slice_tensor_shape)]
                    }
                    local_rank_comp_cost[key].append(
                        SliceOpCost(op_desc=slice_desc, cluster=cluster))

        res = (comm_costs, local_rank_comp_cost)

        return res
