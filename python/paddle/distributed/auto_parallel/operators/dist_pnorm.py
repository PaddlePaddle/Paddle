# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import paddle
import paddle.fluid.layers.utils as utils

from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from .common import set_comm_op_dist_attr_for_program
from .dist_default import DistributedDefaultImpl0
from ..reshard import Resharder
from ..process_group import new_process_group
from ..utils import is_dim_shard, is_dim_replicate, _get_corresponding_rank
from ..utils import compute_compatible_dim_mapping, set_dist_op_desc_original_id, _get_comm_group
from ..dist_attribute import TensorDistributedAttribute, OperatorDistributedAttribute

from paddle.fluid import core, unique_name
from paddle.fluid.framework import Operator
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype


class DistributedPNorm(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedPNorm, self).__init__(op_type)


register_distributed_operator_impl_container(DistributedPNorm("p_norm"))


def _insert_fill_constant_op(block, op_role):
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
    fill_constant_op = block.append_op(
        type='fill_constant',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs)
    out.stop_gradient = True
    return out, fill_constant_op


# Row Parallel
class DistributedPNormImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedPNormImpl, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        if is_dim_replicate(x_dims_mapping[0]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        return True

    def is_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)) or \
            (not self.is_compatible(dist_op)):
            return False
        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        batch_dim_mappings = []
        for arg_name in op_desc.input_arg_names():
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if len(dims_mapping) >= 1:
                batch_dim_mappings.append(dims_mapping[0])
        for arg_name in op_desc.output_arg_names():
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if len(dims_mapping) >= 1:
                batch_dim_mappings.append(dims_mapping[0])

        compatible_dim_mapping = compute_compatible_dim_mapping(
            batch_dim_mappings)
        assert compatible_dim_mapping is not None, "There is no compatible dim mapping."

        for arg_name in op_desc.input_arg_names():
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if len(dims_mapping
                   ) >= 1 and compatible_dim_mapping != dims_mapping[0]:
                dims_mapping[0] = compatible_dim_mapping
                changed = True
        for arg_name in op_desc.output_arg_names():
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if len(dims_mapping
                   ) >= 1 and compatible_dim_mapping != dims_mapping[0]:
                dims_mapping[0] = compatible_dim_mapping
                changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert op_dist_attr is not None

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        X_var = main_block.var(kwargs['X'][0])
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(X_var.name)
        for axis in range(len(in_dims_mapping)):
            if in_dims_mapping[axis] != -1:
                break
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      axis, rank_id)
        group = new_process_group(group_ranks)

        check_variable_and_dtype(X_var, 'x', ['float16', 'float32', 'float64'],
                                 'norm')
        check_dtype(X_var.dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'norm')

        # 1. insert barrier op
        ref_process_mesh = op_dist_attr.process_mesh
        constant_out_dims_mapping = [-1]
        fill_constant_out, fill_constant_op = _insert_fill_constant_op(
            main_block, src_op.attr('op_role'))
        # set fill_constant_out tensor dist_attr
        constant_out_dist_attr = TensorDistributedAttribute()
        constant_out_dist_attr.process_mesh = ref_process_mesh
        constant_out_dist_attr.dims_mapping = constant_out_dims_mapping
        ctx.set_tensor_dist_attr_for_program(fill_constant_out,
                                             constant_out_dist_attr)
        # set fill_constant op dist_attr
        constant_op_dist_attr = OperatorDistributedAttribute()
        constant_op_dist_attr.process_mesh = ref_process_mesh
        constant_op_dist_attr.set_output_dims_mapping(fill_constant_out.name,
                                                      constant_out_dims_mapping)
        ctx.set_op_dist_attr_for_program(fill_constant_op,
                                         constant_op_dist_attr)
        barrier_op = main_block.append_op(
            type='barrier',
            inputs={'X': [fill_constant_out]},
            outputs={'Out': [fill_constant_out]},
            attrs={'ring_id': group.id})
        # set barrier op dist attr
        set_comm_op_dist_attr_for_program(barrier_op, ref_process_mesh,
                                          constant_out_dist_attr, ctx)

        # 2. insert c_allgather op
        # create c_allgather output var
        allgather_out = main_block.create_var(
            name=".".join(["c_allgather", X_var.name]),
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_var.stop_gradient)
        # set allgather_out tensor dist_attr
        allgather_out_dist_attr = TensorDistributedAttribute()
        allgather_out_dist_attr.process_mesh = op_dist_attr.process_mesh
        allgather_out_dist_attr.dims_mapping = [
            -1 for i in range(len(allgather_out.shape))
        ]
        ctx.set_tensor_dist_attr_for_program(allgather_out,
                                             allgather_out_dist_attr)
        c_allgather_op = main_block.append_op(
            type='c_allgather',
            inputs={'X': [X_var]},
            outputs={'Out': [allgather_out]},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'nranks': group.nranks,
                'op_role': src_op.attr('op_role')
            })
        # set c_allgather op dist_attr
        allgather_op_dist_attr = OperatorDistributedAttribute()
        allgather_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allgather_op_dist_attr.set_input_dims_mapping(X_var.name,
                                                      in_dims_mapping)
        allgather_op_dist_attr.set_output_dims_mapping(
            allgather_out.name, allgather_out_dist_attr.dims_mapping)
        ctx.set_op_dist_attr_for_program(c_allgather_op, allgather_op_dist_attr)

        # 3. copy p_norm op desc and reset input name
        # rename input
        kwargs['X'] = [allgather_out.name]
        # replicate op in dist program
        dist_op_desc = main_block.desc.append_op()
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])
        pnorm_op = Operator(main_block, dist_op_desc)
        op_dist_attr.set_input_dims_mapping(
            allgather_out.name, allgather_out_dist_attr.dims_mapping)
        ctx.set_op_dist_attr_for_program(pnorm_op, op_dist_attr)

        main_block._sync_with_cpp()

    @staticmethod
    def backward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert op_dist_attr is not None

        # check validation of inputs / outputs
        for input_name in backward_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                backward_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in backward_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                backward_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        X_grad_var = main_block.var(kwargs['X@GRAD'][0])

        # 1. copy p_norm_grad op and reset input name and output name
        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs['X'] = [".".join(["c_allgather", X_var.name])]
        new_X_var = main_block.var(new_kwargs['X'][0])
        new_X_grad = main_block.create_var(
            name=".".join(["c_allgather", X_grad_var.name]),
            dtype=X_grad_var.dtype,
            shape=new_X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_grad_var.stop_gradient)
        new_kwargs['X@GRAD'] = [new_X_grad.name]
        new_X_var_dist_attr = ctx.get_tensor_dist_attr_for_program(new_X_var)
        ctx.set_tensor_dist_attr_for_program(new_X_grad, new_X_var_dist_attr)
        # replicate op in dist program with new kwargs
        dist_op_desc = main_block.desc.append_op()
        dist_op_desc.copy_from(backward_op.desc)
        # Refer to the related dist op
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        for input_name in backward_op.desc.input_names():
            dist_op_desc.set_input(input_name, new_kwargs[input_name])
        for output_name in backward_op.desc.output_names():
            dist_op_desc.set_output(output_name, new_kwargs[output_name])
        p_norm_grad_op = Operator(main_block, dist_op_desc)
        op_dist_attr.set_input_dims_mapping(new_X_var.name,
                                            new_X_var_dist_attr.dims_mapping)
        op_dist_attr.set_output_dims_mapping(new_X_grad.name,
                                             new_X_var_dist_attr.dims_mapping)
        ctx.set_op_dist_attr_for_program(p_norm_grad_op, op_dist_attr)
        main_block._sync_with_cpp()

        # 2. insert slice op
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes
        dims_mapping = [0] + [-1 for _ in range(len(new_X_grad.shape) - 1)]
        partition_idx = Resharder.compute_partition_index(
            rank_id, new_X_grad.shape, dims_mapping, process_mesh_shape,
            process_mesh_group)
        slice_starts = []
        slice_ends = []
        slices_axes = []
        for idx, item in enumerate(partition_idx):
            slice_starts.append(item[0])
            slice_ends.append(item[1])
            slices_axes.append(idx)

        infer_flags = list(1 for i in range(len(slices_axes)))
        attrs = {
            "axes": slices_axes,
            "starts": slice_starts,
            "ends": slice_ends,
            "infer_flags": infer_flags,
            "op_role": backward_op.attr('op_role')
        }
        slice_op = main_block.append_op(
            type='slice',
            inputs={'Input': [new_X_grad]},
            outputs={'Out': [X_grad_var]},
            attrs=attrs)
        X_grad_var_dims_mapping = op_dist_attr.get_output_dims_mapping(
            X_grad_var.name)
        slice_op_dist_attr = OperatorDistributedAttribute()
        slice_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        slice_op_dist_attr.set_input_dims_mapping(
            new_X_grad.name, new_X_var_dist_attr.dims_mapping)
        slice_op_dist_attr.set_output_dims_mapping(X_grad_var.name,
                                                   X_grad_var_dims_mapping)
        ctx.set_op_dist_attr_for_program(slice_op, slice_op_dist_attr)
        main_block._sync_with_cpp()


register_distributed_operator_impl("p_norm",
                                   DistributedPNormImpl("row_parallel"))
