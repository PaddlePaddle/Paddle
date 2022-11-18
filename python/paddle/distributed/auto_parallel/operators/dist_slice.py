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

from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from ..utils import is_dim_shard
from ..utils import compute_compatible_dim_mapping
from .dist_default import DistributedDefaultImpl0


class DistributedSlice(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedSlice("slice"))


class DistributedSliceImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        in_name = op_desc.input('Input')[0]
        out_name = op_desc.output('Out')[0]
        in_var = dist_op.serial_op.block._var_recursive(in_name)
        out_var = dist_op.serial_op.block._var_recursive(out_name)
        axes = op_desc.attr('axes')
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
        for axis in axes:
            if (
                is_dim_shard(in_dims_mapping[axis])
                and in_var.shape[axis] != out_var.shape[axis]
            ):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        in_name = op_desc.input('Input')[0]
        out_name = op_desc.output('Out')[0]
        in_var = dist_op.serial_op.block._var_recursive(in_name)
        out_var = dist_op.serial_op.block._var_recursive(out_name)
        axes = op_desc.attr('axes')
        decrease_axis = op_desc.attr('decrease_axis')
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        ref_indices = []
        for i in range(len(in_dims_mapping)):
            if i not in decrease_axis:
                ref_indices.append(i)
        if ref_indices == []:
            assert len(out_dims_mapping) == 1
            if is_dim_shard(out_dims_mapping[0]):
                return False
        else:
            for i in range(len(out_dims_mapping)):
                ref_index = ref_indices[i]
                if (
                    ref_index in axes
                    and is_dim_shard(out_dims_mapping[i])
                    and in_var.shape[ref_index] != out_var.shape[ref_index]
                ):
                    return False

        return True

    def is_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        in_name = op_desc.input('Input')[0]
        out_name = op_desc.output('Out')[0]
        decrease_axis = op_desc.attr('decrease_axis')
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if len(in_dims_mapping) - len(decrease_axis) != 0 and len(
            out_dims_mapping
        ) != len(in_dims_mapping) - len(decrease_axis):
            return False

        new_out_dims_mapping = []
        for i in range(len(in_dims_mapping)):
            if i not in decrease_axis:
                new_out_dims_mapping.append(in_dims_mapping[i])
        if new_out_dims_mapping == []:
            new_out_dims_mapping = [-1]
        if new_out_dims_mapping != out_dims_mapping:
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
        in_name = op_desc.input('Input')[0]
        out_name = op_desc.output('Out')[0]
        decrease_axis = op_desc.attr('decrease_axis')
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        ref_dims_mapping = []
        ref_indices = []
        for i in range(len(in_dims_mapping)):
            if i not in decrease_axis:
                ref_dims_mapping.append(in_dims_mapping[i])
                ref_indices.append(i)

        if ref_dims_mapping == []:
            ref_dims_mapping = [-1]
            assert len(ref_dims_mapping) == len(out_dims_mapping)
            assert ref_dims_mapping[0] == out_dims_mapping[0]
            changed = False
        else:
            assert len(ref_dims_mapping) == len(out_dims_mapping)
            for i in range(len(out_dims_mapping)):
                compatible_dim_mapping = compute_compatible_dim_mapping(
                    [out_dims_mapping[i], ref_dims_mapping[i]]
                )
                if compatible_dim_mapping is None:
                    continue
                if ref_dims_mapping[i] != compatible_dim_mapping:
                    in_dims_mapping[ref_indices[i]] = compatible_dim_mapping
                    changed = True
                if out_dims_mapping[i] != compatible_dim_mapping:
                    out_dims_mapping[i] = compatible_dim_mapping
                    changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "slice", DistributedSliceImpl("decrease_in_axis")
)
