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
from .common import register_distributed_operator_impl, is_parameter_related
from ..utils import is_dim_shard
from ..utils import compute_compatible_and_update_dim_mapping
from ..utils import set_dist_op_desc_original_id
from .dist_default import DistributedDefaultImpl0
from ..cost import build_comp_desc_from_dist_op, build_comp_costs_from_descs
from ..cost import Reshape2OpCost
from ..cost import Reshape2GradOpCost
from ..cost import build_dp_costs
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class DistributedReshape2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedReshape2("reshape2"))


class DistributedReshapeImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
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
        res = []
        op = dist_op.serial_op
        vars = op.block.vars
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.processes
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        op_type = dist_op.serial_op.type

        cost_mapping = build_comp_costs_from_descs(
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        vars = main_block.vars
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and is_parameter_related(
                    varname, main_block
                ):
                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
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
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) - 1:
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) - 1:
            return False

        if is_dim_shard(out_dims_mapping[-1]):
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
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for idx, dim_mapping in enumerate(out_dims_mapping[:-1]):
            if x_dims_mapping[idx] != dim_mapping:
                return False

        if x_shape_dims_mapping[0] != -1:
            return False

        if x_shape_dims_mapping[1:] != x_dims_mapping[:]:
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )

        for i in range(len(x_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        for i in range(len(x_dims_mapping)):
            x_shape_dims_mapping[i + 1] = x_dims_mapping[i]

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name
            )
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block.var(kwargs['X'][0])
        Out_var = main_block.var(kwargs['Out'][0])
        XShape_var = main_block.var(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op_desc = main_block.append_op(type='nop').desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


class DistributedReshapeImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
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
        res = []
        op = dist_op.serial_op
        vars = op.block.vars
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.processes
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        op_type = dist_op.serial_op.type

        cost_mapping = build_comp_costs_from_descs(
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        vars = main_block.vars
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                    varname, main_block
                ):
                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
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
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) + 1:
            return False

        if is_dim_shard(x_dims_mapping[-1]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping) + 1:
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
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )

        if is_dim_shard(x_dims_mapping[-1]):
            return False

        for idx, item in enumerate(x_dims_mapping[:-1]):
            if out_dims_mapping[idx] != item:
                return False

        if x_shape_dims_mapping[0] != -1:
            return False

        if x_shape_dims_mapping[1:] != x_dims_mapping[:]:
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )

        for i in range(len(out_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        for i in range(len(x_dims_mapping)):
            x_shape_dims_mapping[i + 1] = x_dims_mapping[i]

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name
            )
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block.var(kwargs['X'][0])
        Out_var = main_block.var(kwargs['Out'][0])
        XShape_var = main_block.var(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op_desc = main_block.append_op(type='nop').desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


class DistributedReshapeImpl2(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
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
        res = []
        op = dist_op.serial_op
        vars = op.block.vars
        dist_attr = dist_op.dist_attr

        shape_list = op.desc.attr("shape")
        # got dist attribute info
        dim_mapping = dist_attr.get_output_dims_mapping(op.output("Out")[0])
        process_mesh_shape = dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_attr.process_mesh.processes
        for key in desc_mapping:
            desc_mapping[key]["shape"] = shape_list

        cost_mapping = build_comp_costs_from_descs(
            Reshape2OpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        return res

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        op_type = dist_op.serial_op.type

        cost_mapping = build_comp_costs_from_descs(
            Reshape2GradOpCost, ctx, processes, desc_mapping, cluster
        )
        res.append(cost_mapping)

        backward_op = dist_op.serial_op
        main_block = backward_op.block
        need_gradient_allreduce = False
        vars = main_block.vars
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                    varname, main_block
                ):
                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
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
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        x_name = op_desc.input('X')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if len(x_dims_mapping) != len(out_dims_mapping):
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
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )

        for idx, item in enumerate(x_dims_mapping[:-1]):
            if out_dims_mapping[idx] != item:
                return False

        if x_shape_dims_mapping[0] != -1:
            return False

        if x_shape_dims_mapping[1:] != out_dims_mapping[:]:
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name
        )

        for i in range(len(out_dims_mapping) - 1):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [x_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        for i in range(len(out_dims_mapping)):
            x_shape_dims_mapping[i + 1] = out_dims_mapping[i]

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        src_op = dist_op_context.cur_src_op
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name
            )
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block.var(kwargs['X'][0])
        Out_var = main_block.var(kwargs['Out'][0])
        XShape_var = main_block.var(kwargs['XShape'][0])
        shape_list = src_op.desc.attr("shape")
        ShapeTensor_var_list = []
        for name in kwargs['ShapeTensor']:
            ShapeTensor_var_list.append(name)
        Shape_var_list = []
        for name in kwargs['Shape']:
            Shape_var_list.append(name)

        # got dist attribute info
        out_dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
        process_mesh_shape = op_dist_attr.process_mesh.topology

        # modify target shape
        for idx, axis in enumerate(out_dim_mapping):
            if axis >= 0:
                if len(shape_list) > idx:
                    shape_list[idx] = (
                        shape_list[idx] // process_mesh_shape[axis]
                    )

        # create op
        new_op_desc = main_block.append_op(type='nop').desc
        new_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(new_op_desc, src_op.desc, ctx)
        new_op_desc.set_input('ShapeTensor', ShapeTensor_var_list)
        new_op_desc.set_input('Shape', Shape_var_list)
        new_op_desc.set_input('X', [X_var.name])
        new_op_desc.set_output('XShape', [XShape_var.name])
        new_op_desc.set_output('Out', [Out_var.name])
        new_op_desc._set_attr('shape', shape_list)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl0("add_one_dim_back")
)
register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl1("remove_one_dim_back")
)
register_distributed_operator_impl(
    "reshape2", DistributedReshapeImpl2("same_dim_shape")
)
