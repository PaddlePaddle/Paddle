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

from paddle.common_ops_import import check_dtype, check_variable_and_dtype
from paddle.framework import core
from paddle.static import Operator

from ..dist_attribute import OperatorDistAttr, TensorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    compute_compatible_dim_mapping,
    is_dim_replicate,
    is_dim_shard,
    set_dist_op_desc_original_id,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)


class DistributedPNorm(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedPNorm("p_norm"))


# Data Parallel
class DistributedPNormImpl0(DistributedOperatorImpl):
    """
    TODO: p_norm scene

    1. axis == None, isinstance(p, (int, float)), asvector = True
        1.1 x_dims_mapping == [0, -1, -1]
            allgather input if it is splitted by dp group
        1.2 x_dims_mapping == [-1, 0, -1]
            allgather, split and concat input if it is splitted by mp group
    2. isinstance(axis, int), asvector = False
        1.1 axis == 0 and x_dims_mapping == [0, -1, -1]
            allgather input if it's input[0] is splited by dp group.
        1.2 axis == 1 and x_dims_mapping == [-1, 0, -1]
            allgather, split and concat input if it's input[1] is splitted by mp group
    """

    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        axis = op_desc.attr('axis')
        asvector = op_desc.attr('asvector')
        x_name = op_desc.input('X')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        if is_dim_replicate(x_dims_mapping[0]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        if not (axis == -1 and asvector) and not (axis == 0 and not asvector):
            return False
        return True

    def is_output_compatible(self, dist_op):
        return True

    def is_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False
        return True

    def is_auto_compatible(self, dist_op):
        if (
            (not self.is_input_compatible(dist_op))
            or (not self.is_output_compatible(dist_op))
            or (not self.is_compatible(dist_op))
        ):
            return False
        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        axis = op_desc.attr('axis')
        keepdim = op_desc.attr('keepdim')

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
            batch_dim_mappings
        )
        if compatible_dim_mapping is None:
            return False

        for arg_name in op_desc.input_arg_names():
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if (
                len(dims_mapping) >= 1
                and compatible_dim_mapping != dims_mapping[0]
            ):
                dims_mapping[0] = compatible_dim_mapping
                op_dist_attr.set_input_dims_mapping(arg_name, dims_mapping)
                changed = True

        if axis == 0 and not keepdim:
            for arg_name in op_desc.output_arg_names():
                dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
                if len(dims_mapping) >= 1 and dims_mapping[0] != -1:
                    dims_mapping[0] = -1
                    op_dist_attr.set_output_dims_mapping(arg_name, dims_mapping)
                    changed = True
        else:
            for arg_name in op_desc.output_arg_names():
                dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
                if (
                    len(dims_mapping) >= 1
                    and compatible_dim_mapping != dims_mapping[0]
                ):
                    dims_mapping[0] = compatible_dim_mapping
                    op_dist_attr.set_output_dims_mapping(arg_name, dims_mapping)
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
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(X_var.name)
        for axis in range(len(in_dims_mapping)):
            if in_dims_mapping[axis] != -1:
                break
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, axis, rank_id
        )
        group = new_process_group(group_ranks)

        check_variable_and_dtype(
            X_var, 'x', ['float16', 'float32', 'float64'], 'norm'
        )
        check_dtype(
            X_var.dtype, 'dtype', ['float16', 'float32', 'float64'], 'norm'
        )

        # 2. insert c_allgather op
        # create c_allgather output var
        allgather_out = main_block.create_var(
            name=".".join(["c_allgather", X_var.name]),
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_var.stop_gradient,
        )
        # set allgather_out tensor dist_attr
        allgather_out_dist_attr = TensorDistAttr()
        allgather_out_dist_attr.process_mesh = op_dist_attr.process_mesh
        allgather_out_dist_attr.chunk_id = op_dist_attr.chunk_id
        allgather_out_dist_attr.dims_mapping = [
            -1 for i in range(len(allgather_out.shape))
        ]
        ctx.set_tensor_dist_attr_for_program(
            allgather_out, allgather_out_dist_attr
        )
        c_allgather_op = main_block.append_op(
            type='c_allgather',
            inputs={'X': [X_var]},
            outputs={'Out': [allgather_out]},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'nranks': group.nranks,
                'op_role': src_op.attr('op_role'),
            },
        )
        # set c_allgather op dist_attr
        allgather_op_dist_attr = OperatorDistAttr()
        allgather_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allgather_op_dist_attr.chunk_id = op_dist_attr.chunk_id
        allgather_op_dist_attr.set_input_dims_mapping(
            X_var.name, in_dims_mapping
        )
        allgather_op_dist_attr.set_output_dims_mapping(
            allgather_out.name, allgather_out_dist_attr.dims_mapping
        )
        ctx.set_op_dist_attr_for_program(c_allgather_op, allgather_op_dist_attr)

        # 3. copy p_norm op desc and reset input name
        # rename input
        kwargs['X'] = [allgather_out.name]
        # replicate op in dist program
        dist_op = main_block.append_op(type='nop')
        dist_op_desc = dist_op.desc
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])
        pnorm_op = Operator(main_block, dist_op_desc)
        op_dist_attr.set_input_dims_mapping(
            allgather_out.name, allgather_out_dist_attr.dims_mapping
        )
        # Remove the unrelated dist attr
        op_dist_attr.del_input_dist_attr(X_var.name)
        ctx.set_op_dist_attr_for_program(pnorm_op, op_dist_attr)
        # TODO: should we add a new dist attr for the new op here?

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
            assert input_name in kwargs, f"input [{input_name}] is not given"
            assert len(kwargs[input_name]) == len(
                backward_op.desc.input(input_name)
            ), f"number of tensor for input [{input_name}] is not match"
        for output_name in backward_op.desc.output_names():
            assert output_name in kwargs, f"input [{output_name}] is not given"
            assert len(kwargs[output_name]) == len(
                backward_op.desc.output(output_name)
            ), f"number of tensor for input [{output_name}] is not match"

        X_var = main_block._var_recursive(kwargs['X'][0])
        X_grad_var = main_block._var_recursive(kwargs['X@GRAD'][0])

        # 1. copy p_norm_grad op and reset input name and output name
        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs['X'] = [".".join(["c_allgather", X_var.name])]
        new_X_var = main_block._var_recursive(new_kwargs['X'][0])
        new_X_grad = main_block.create_var(
            name=".".join(["c_allgather", X_grad_var.name]),
            dtype=X_grad_var.dtype,
            shape=new_X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_grad_var.stop_gradient,
        )
        new_kwargs['X@GRAD'] = [new_X_grad.name]
        new_X_var_dist_attr = ctx.get_tensor_dist_attr_for_program(new_X_var)
        ctx.set_tensor_dist_attr_for_program(new_X_grad, new_X_var_dist_attr)
        # replicate op in dist program with new kwargs
        dist_op = main_block.append_op(type='nop')
        dist_op_desc = dist_op.desc
        dist_op_desc.copy_from(backward_op.desc)
        # Refer to the related dist op
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        for input_name in backward_op.desc.input_names():
            dist_op_desc.set_input(input_name, new_kwargs[input_name])
        for output_name in backward_op.desc.output_names():
            dist_op_desc.set_output(output_name, new_kwargs[output_name])
        p_norm_grad_op = Operator(main_block, dist_op_desc)
        op_dist_attr.set_input_dims_mapping(
            new_X_var.name, new_X_var_dist_attr.dims_mapping
        )
        # Store X_grad_var dims_mapping for later use
        X_grad_var_dims_mapping = op_dist_attr.get_output_dims_mapping(
            X_grad_var.name
        )
        # Remove the unrelated dist attr
        op_dist_attr.del_input_dist_attr(X_var.name)
        op_dist_attr.set_output_dims_mapping(
            new_X_grad.name, new_X_var_dist_attr.dims_mapping
        )
        # Remove the unrelated dist attr
        op_dist_attr.del_output_dist_attr(X_grad_var.name)
        ctx.set_op_dist_attr_for_program(p_norm_grad_op, op_dist_attr)
        # TODO: should we add a new dist attr for the new op here?

        # 2. insert slice op
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids
        dims_mapping = [0] + [-1 for _ in range(len(new_X_grad.shape) - 1)]
        from ..reshard import Resharder

        partition_idx = Resharder.compute_partition_index(
            rank_id,
            new_X_grad.shape,
            dims_mapping,
            process_mesh_shape,
            process_mesh_group,
        )
        slice_starts = []
        slice_ends = []
        slices_axes = []
        for idx, item in enumerate(partition_idx):
            slice_starts.append(item[0])
            slice_ends.append(item[1])
            slices_axes.append(idx)

        infer_flags = [1 for i in range(len(slices_axes))]
        attrs = {
            "axes": slices_axes,
            "starts": slice_starts,
            "ends": slice_ends,
            "infer_flags": infer_flags,
            "op_role": backward_op.attr('op_role'),
        }
        slice_op = main_block.append_op(
            type='slice',
            inputs={'Input': [new_X_grad]},
            outputs={'Out': [X_grad_var]},
            attrs=attrs,
        )
        slice_op_dist_attr = OperatorDistAttr()
        slice_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        slice_op_dist_attr.chunk_id = op_dist_attr.chunk_id
        slice_op_dist_attr.set_input_dims_mapping(
            new_X_grad.name, new_X_var_dist_attr.dims_mapping
        )
        slice_op_dist_attr.set_output_dims_mapping(
            X_grad_var.name, X_grad_var_dims_mapping
        )
        ctx.set_op_dist_attr_for_program(slice_op, slice_op_dist_attr)


register_distributed_operator_impl(
    "p_norm", DistributedPNormImpl0("data_parallel")
)
