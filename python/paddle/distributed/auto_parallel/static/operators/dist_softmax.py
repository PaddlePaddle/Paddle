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
    SoftmaxGradOpCost,
    SoftmaxOpCost,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..utils import compute_compatible_and_update_dim_mapping, is_dim_shard
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    is_parameter_related,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0


class DistributedSoftmax(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedSoftmax("softmax"))


class DistributedSoftmaxImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = False
        self._backward_implemented = False

    def calc_cost(self, op_role, dist_op, ctx, cluster):
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
        cost_mapping = build_comp_costs_from_descs(
            SoftmaxOpCost, ctx, processes, desc_mapping, cluster
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
        cost_mapping = build_comp_costs_from_descs(
            SoftmaxGradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and is_parameter_related(
                    varname, main_block
                ):
                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    mesh_shape = process_mesh.shape
                    batch_size_axis = (
                        var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1
                    )
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
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
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)

        # if axis != -1 and axis != len(x_dims_mapping) - 1:
        #     return False

        if is_dim_shard(x_dims_mapping[axis]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        axis = op_desc.attr('axis')
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        # if axis != -1 and axis != len(out_dims_mapping) - 1:
        #     return False

        if is_dim_shard(out_dims_mapping[axis]):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        # if axis != -1 and axis != len(x_dims_mapping) - 1:
        #     return False

        if x_dims_mapping != out_dims_mapping:
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for i in range(len(x_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "softmax", DistributedSoftmaxImpl("replicate_last_axis")
)
