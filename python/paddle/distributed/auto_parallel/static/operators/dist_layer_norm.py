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
import os

from paddle.distributed.fleet.meta_optimizers.common import OpRole

from ..cost import (
    _g_op_cost_factory,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..utils import compute_compatible_dim_mapping, infer_with_spmd
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    is_parameter_related,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0


class DistributedLayerNorm(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedLayerNorm("layer_norm"))


class DistributedLayerNormImpl0(DistributedOperatorImpl):
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
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        scale_name = op_desc.input('Scale')[0]
        bias_name = op_desc.input('Bias')[0]
        begin_norm_axis = op_desc.attr('begin_norm_axis')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        scale_dims_mapping = op_dist_attr.get_input_dims_mapping(scale_name)
        bias_dims_mapping = op_dist_attr.get_input_dims_mapping(bias_name)

        if scale_dims_mapping[0] > -1 or bias_dims_mapping[0] > -1:
            return False
        if any(val > -1 for val in x_dims_mapping[0:]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Y')[0]
        mean_name = op_desc.output('Mean')[0]
        var_name = op_desc.output('Variance')[0]
        begin_norm_axis = op_desc.attr('begin_norm_axis')
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        mean_dims_mapping = op_dist_attr.get_output_dims_mapping(mean_name)
        var_dims_mapping = op_dist_attr.get_output_dims_mapping(var_name)

        # mean_dims_mapping[0] and var_dims_mapping[0] should be compatible
        # with out_dims_mapping[0]
        if (
            compute_compatible_dim_mapping(
                [out_dims_mapping[0], mean_dims_mapping[0]]
            )
            is None
        ):
            return False
        if (
            compute_compatible_dim_mapping(
                [out_dims_mapping[0], var_dims_mapping[0]]
            )
            is None
        ):
            return False
        if (
            compute_compatible_dim_mapping(
                [mean_dims_mapping[0], var_dims_mapping[0]]
            )
            is None
        ):
            return False
        if any(val > -1 for val in out_dims_mapping[0:]):
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
        out_name = op_desc.output('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        begin_norm_axis = op_desc.attr('begin_norm_axis')

        for x_val, out_val in zip(
            x_dims_mapping[:begin_norm_axis], out_dims_mapping[:begin_norm_axis]
        ):
            if x_val != out_val:
                return False
        return True

    def update_dims_mapping(self, dist_op):
        if os.getenv("ENABLE_SPMD_RULE") == 'true':
            print("################ layer_norm spmd ####################")
            op_desc = dist_op.serial_op.desc
            op_dist_attr = dist_op.dist_attr
            x_name = op_desc.input('X')[0]
            scale_name = op_desc.input('Scale')[0]
            bias_name = op_desc.input('Bias')[0]
            out_name = op_desc.output('Y')[0]
            mean_name = op_desc.output('Mean')[0]
            var_name = op_desc.output('Variance')[0]
            input_names = [x_name, scale_name, bias_name]
            output_names = [out_name, mean_name, var_name]
            attr_names = ['begin_norm_axis']

            return infer_with_spmd(
                dist_op, input_names, output_names, attr_names, "layer_norm"
            )
        else:
            default_impl = DistributedDefaultImpl0("default")
            return default_impl.update_dims_mapping(dist_op)

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "layer_norm", DistributedLayerNormImpl0("layer_norm")
)
