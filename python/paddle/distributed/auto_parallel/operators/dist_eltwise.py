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

from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from .common import is_elementwise_op
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..dist_attribute import OperatorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank
from .dist_default import DistributedDefaultImpl0


class DistributedElementwise(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedElementwise, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedElementwise("elementwise"))


# Replicated Elementwise
class DistributedElementwiseImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedElementwiseImpl0, self).__init__(name)
        self._forward_implemented = False
        self._backward_implemented = False

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        if is_elementwise_op(op_desc.type()):
            return True
        else:
            return False

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_desc = dist_op.serial_op.desc
        if is_elementwise_op(op_desc.type()):
            return True
        else:
            return False

    def is_auto_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        dims_mapping_list = []
        input_arg_names = op_desc.input_arg_names()
        max_dims_mapping_len = -1
        for arg_name in input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if max_dims_mapping_len < len(dims_mapping):
                max_dims_mapping_len = len(dims_mapping)
            dims_mapping_list.append(dims_mapping)
        output_arg_names = op_desc.output_arg_names()
        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            assert len(dims_mapping) == max_dims_mapping_len
            dims_mapping_list.append(dims_mapping)

        for idx in range(max_dims_mapping_len):
            dim_mappings = []
            for dims_mapping in dims_mapping_list:
                if idx < len(dims_mapping):
                    dim_mappings.append(dims_mapping[-(idx + 1)])
            if not all(dim_mappings[0] == dim_mapping
                       for dim_mapping in dim_mappings):
                return False
        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        input_arg_names = op_desc.input_arg_names()
        input_dims_mapping_dict = {}
        input_dims_mapping_lens = {}
        max_dims_mapping_len = -1
        for arg_name in input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if max_dims_mapping_len < len(dims_mapping):
                max_dims_mapping_len = len(dims_mapping)
            input_dims_mapping_dict[arg_name] = dims_mapping
            input_dims_mapping_lens[arg_name] = len(dims_mapping)

        dims_mapping_list = []
        for arg_name in input_arg_names:
            if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
                new_dims_mapping = [-1 for _ in range(max_dims_mapping_len)]
                for i in range(input_dims_mapping_lens[arg_name]):
                    new_idx = (max_dims_mapping_len -
                               input_dims_mapping_lens[arg_name]) + i
                    new_dims_mapping[new_idx] = input_dims_mapping_dict[
                        arg_name][i]
                dims_mapping_list.append(new_dims_mapping)
            else:
                dims_mapping_list.append(input_dims_mapping_dict[arg_name])
        output_arg_names = op_desc.output_arg_names()
        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            assert len(dims_mapping) == max_dims_mapping_len
            dims_mapping_list.append(dims_mapping)

        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list)
        assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

        for arg_name in input_arg_names:
            if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
                new_dims_mapping = [
                    -1 for _ in range(input_dims_mapping_lens[arg_name])
                ]
                for i in range(input_dims_mapping_lens[arg_name]):
                    new_idx = (max_dims_mapping_len -
                               input_dims_mapping_lens[arg_name]) + i
                    new_dims_mapping[i] = compatible_dims_mapping[new_idx]
                if new_dims_mapping != input_dims_mapping_dict[arg_name]:
                    op_dist_attr.set_input_dims_mapping(arg_name,
                                                        new_dims_mapping)
                    changed = True
            else:
                if compatible_dims_mapping != input_dims_mapping_dict[arg_name]:
                    op_dist_attr.set_input_dims_mapping(arg_name,
                                                        compatible_dims_mapping)
                    changed = True

        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if compatible_dims_mapping != dims_mapping:
                op_dist_attr.set_output_dims_mapping(arg_name,
                                                     compatible_dims_mapping)
                changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "elementwise", DistributedElementwiseImpl0("replicate_parallel"))
