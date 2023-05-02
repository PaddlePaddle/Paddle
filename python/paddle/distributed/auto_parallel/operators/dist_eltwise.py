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

from paddle.distributed.fleet.meta_optimizers.common import OpRole

from ..cost import (
    _g_op_cost_factory,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..utils import (
    compute_compatible_dim_mapping,
    compute_compatible_dims_mapping,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    is_elementwise_op,
    is_parameter_related,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0


class DistributedElementwise(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(
    DistributedElementwise("elementwise")
)


# Replicated Elementwise
class DistributedElementwiseImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = False
        self._backward_implemented = False

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        """Calculate the cost by the op role."""
        cost = None
        if int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        else:
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        op_type = dist_op.serial_op.type
        cost_mapping = build_comp_costs_from_descs(
            _g_op_cost_factory[op_type], ctx, processes, desc_mapping, cluster
        )
        res_cost = [cost_mapping]

        return res_cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        backward_op = dist_op.serial_op
        op_type = backward_op.type
        cost_mapping = build_comp_costs_from_descs(
            _g_op_cost_factory[op_type], ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                    varname, main_block
                ):
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)
                    mesh_shape = process_mesh.shape
                    batch_size_axis = (
                        var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1
                    )
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        need_gradient_allreduce = True
                        break

        if need_gradient_allreduce:
            for input_name in backward_op.desc.input_names():
                for varname in backward_op.desc.input(input_name):
                    if "@GRAD" not in varname and is_parameter_related(
                        varname, main_block
                    ):
                        var_dim_mapping = dist_attr.get_input_dims_mapping(
                            varname
                        )
                        mesh_shape = process_mesh.shape
                        batch_size_axis = var_dim_mapping[0]
                        parallel_axis = batch_size_axis
                        attrs = {"use_calc_stream": True}
                        var_names = [varname + "@GRAD"]
                        build_dp_costs(
                            res,
                            dist_op,
                            ctx,
                            var_names,
                            attrs,
                            parallel_axis,
                            cluster,
                        )
        return res

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        if not is_elementwise_op(op_desc.type()):
            return False
        op_dist_attr = dist_op.dist_attr
        dims_mapping_list = []
        input_arg_names = op_desc.input_arg_names()
        max_dims_mapping_len = -1
        for arg_name in input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if max_dims_mapping_len < len(dims_mapping):
                max_dims_mapping_len = len(dims_mapping)
            dims_mapping_list.append(dims_mapping)

        for idx in range(max_dims_mapping_len):
            dim_mappings = []
            for dims_mapping in dims_mapping_list:
                if idx < len(dims_mapping):
                    dim_mappings.append(dims_mapping[-(idx + 1)])
            if compute_compatible_dim_mapping(dim_mappings) is None:
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        if not is_elementwise_op(op_desc.type()):
            return False
        op_dist_attr = dist_op.dist_attr
        dims_mapping_list = []
        output_arg_names = op_desc.output_arg_names()
        max_dims_mapping_len = -1
        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if max_dims_mapping_len < len(dims_mapping):
                max_dims_mapping_len = len(dims_mapping)
            dims_mapping_list.append(dims_mapping)

        for idx in range(max_dims_mapping_len):
            dim_mappings = []
            for dims_mapping in dims_mapping_list:
                if idx < len(dims_mapping):
                    dim_mappings.append(dims_mapping[-(idx + 1)])
            if compute_compatible_dim_mapping(dim_mappings) is None:
                return False
        return True

    def is_auto_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        if not is_elementwise_op(op_desc.type()):
            return False
        op_dist_attr = dist_op.dist_attr
        dims_mapping_list = []

        input_arg_names = op_desc.input_arg_names()
        input_max_dims_mapping_len = -1
        for arg_name in input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if input_max_dims_mapping_len < len(dims_mapping):
                input_max_dims_mapping_len = len(dims_mapping)
            dims_mapping_list.append(dims_mapping)

        output_arg_names = op_desc.output_arg_names()
        output_max_dims_mapping_len = -1
        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if output_max_dims_mapping_len < len(dims_mapping):
                output_max_dims_mapping_len = len(dims_mapping)
            dims_mapping_list.append(dims_mapping)

        assert input_max_dims_mapping_len == output_max_dims_mapping_len
        max_dims_mapping_len = input_max_dims_mapping_len

        for idx in range(max_dims_mapping_len):
            dim_mappings = []
            for dims_mapping in dims_mapping_list:
                if idx < len(dims_mapping):
                    dim_mappings.append(dims_mapping[-(idx + 1)])
            if not all(
                dim_mappings[0] == dim_mapping for dim_mapping in dim_mappings
            ):
                return False
        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        dims_mapping_list = []

        input_arg_names = op_desc.input_arg_names()
        input_dims_mapping_dict = {}
        input_dims_mapping_lens = {}
        input_max_dims_mapping_len = -1
        for arg_name in input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if input_max_dims_mapping_len < len(dims_mapping):
                input_max_dims_mapping_len = len(dims_mapping)
            input_dims_mapping_dict[arg_name] = dims_mapping
            input_dims_mapping_lens[arg_name] = len(dims_mapping)
        for arg_name in input_arg_names:
            if input_dims_mapping_lens[arg_name] < input_max_dims_mapping_len:
                new_dims_mapping = [
                    -1 for _ in range(input_max_dims_mapping_len)
                ]
                for i in range(input_dims_mapping_lens[arg_name]):
                    new_idx = (
                        input_max_dims_mapping_len
                        - input_dims_mapping_lens[arg_name]
                    ) + i
                    new_dims_mapping[new_idx] = input_dims_mapping_dict[
                        arg_name
                    ][i]
                dims_mapping_list.append(new_dims_mapping)
            else:
                dims_mapping_list.append(input_dims_mapping_dict[arg_name])

        output_arg_names = op_desc.output_arg_names()
        output_dims_mapping_dict = {}
        output_dims_mapping_lens = {}
        output_max_dims_mapping_len = -1
        for arg_name in output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if output_max_dims_mapping_len < len(dims_mapping):
                output_max_dims_mapping_len = len(dims_mapping)
            output_dims_mapping_dict[arg_name] = dims_mapping
            output_dims_mapping_lens[arg_name] = len(dims_mapping)
        for arg_name in output_arg_names:
            if output_dims_mapping_lens[arg_name] < output_max_dims_mapping_len:
                new_dims_mapping = [
                    -1 for _ in range(output_max_dims_mapping_len)
                ]
                for i in range(output_dims_mapping_lens[arg_name]):
                    new_idx = (
                        output_max_dims_mapping_len
                        - output_dims_mapping_lens[arg_name]
                    ) + i
                    new_dims_mapping[new_idx] = output_dims_mapping_dict[
                        arg_name
                    ][i]
                dims_mapping_list.append(new_dims_mapping)
            else:
                dims_mapping_list.append(output_dims_mapping_dict[arg_name])

        assert input_max_dims_mapping_len == output_max_dims_mapping_len
        max_dims_mapping_len = input_max_dims_mapping_len
        compatible_dims_mapping = compute_compatible_dims_mapping(
            dims_mapping_list
        )
        if compatible_dims_mapping is None:
            return False

        for arg_name in input_arg_names:
            if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
                new_dims_mapping = [
                    -1 for _ in range(input_dims_mapping_lens[arg_name])
                ]
                for i in range(input_dims_mapping_lens[arg_name]):
                    new_idx = (
                        max_dims_mapping_len - input_dims_mapping_lens[arg_name]
                    ) + i
                    new_dims_mapping[i] = compatible_dims_mapping[new_idx]
                if new_dims_mapping != input_dims_mapping_dict[arg_name]:
                    op_dist_attr.set_input_dims_mapping(
                        arg_name, new_dims_mapping
                    )
                    changed = True
            else:
                if compatible_dims_mapping != input_dims_mapping_dict[arg_name]:
                    op_dist_attr.set_input_dims_mapping(
                        arg_name, compatible_dims_mapping
                    )
                    changed = True

        for arg_name in output_arg_names:
            if output_dims_mapping_lens[arg_name] < max_dims_mapping_len:
                new_dims_mapping = [
                    -1 for _ in range(output_dims_mapping_lens[arg_name])
                ]
                for i in range(output_dims_mapping_lens[arg_name]):
                    new_idx = (
                        max_dims_mapping_len
                        - output_dims_mapping_lens[arg_name]
                    ) + i
                    new_dims_mapping[i] = compatible_dims_mapping[new_idx]
                if new_dims_mapping != output_dims_mapping_dict[arg_name]:
                    op_dist_attr.set_output_dims_mapping(
                        arg_name, new_dims_mapping
                    )
                    changed = True
            else:
                if (
                    compatible_dims_mapping
                    != output_dims_mapping_dict[arg_name]
                ):
                    op_dist_attr.set_output_dims_mapping(
                        arg_name, compatible_dims_mapping
                    )
                    changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "elementwise", DistributedElementwiseImpl0("replicate_parallel")
)
