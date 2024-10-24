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


from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..completion import contains_spmd_rule, get_phi_spmd_rule
from ..cost import (
    _g_op_cost_factory,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..dist_attribute import DistTensorSpec, OperatorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    compute_compatible_dim_mapping,
    get_dist_tensor_spec,
    is_prim_op,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    copy_op_without_infer_shape,
    get_default_distributed_operator_impl,
    gradient_synchronization,
    is_parameter_related,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
    update_op_dims_mapping,
)

__op_not_need_param_init__ = ["while", "cond"]
__op_has_shape_attr__ = [
    "fill_constant_batch_size_like",
    "fill_constant",
    "expand_v2",
    "expand_as_v2",
]


def prim_operator_data_parallel_functor(ctx, src_op):
    dist_op_context = ctx.dist_op_context
    main_block = dist_op_context.work_block
    startup_block = dist_op_context.startup_block

    var_name = src_op.output_arg_names[0]
    if var_name in ctx.grads_params:
        assert (
            var_name not in ctx.synced_gradient
        ), f"in primitive mode, grad is already {var_name} synced"
        ctx.synced_gradient.add(var_name)
        sync_group = new_process_group(ctx.data_parallel_group)

        allreduce_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [var_name]},
            outputs={'Out': [var_name]},
            attrs={
                'ring_id': sync_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )

        param = ctx.grads_params[var_name]
        startup_block = dist_op_context.startup_block
        new_op = startup_block.append_op(
            type='broadcast',
            inputs={'x': [param]},
            outputs={'out': [param]},
            attrs={
                'ring_id': sync_group.id,
                'root': 0,
                OP_ROLE_KEY: OpRole.Forward,
            },
        )

        grad_var = main_block._var_recursive(var_name)
        dims_mapping = ctx.get_tensor_dist_attr_for_program(
            grad_var
        ).dims_mapping
        dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        process_mesh = dist_attr.process_mesh
        op_attr = OperatorDistAttr()
        op_attr.process_mesh = process_mesh
        op_attr.set_output_dims_mapping(grad_var.name, dims_mapping)
        op_attr.set_input_dims_mapping(grad_var.name, dims_mapping)
        ctx.set_op_dist_attr_for_program(allreduce_op, op_attr)


class DistributedDefault(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)

        op_desc = dist_op.serial_op.desc
        input_arg_names = op_desc.input_arg_names()
        output_arg_names = op_desc.output_arg_names()
        main_block = dist_op.serial_op.block

        num_inputs = len(input_arg_names)
        input_specs = []
        for i in range(num_inputs):
            assert not is_parameter_related(
                input_arg_names[i], main_block
            ), f"input {input_arg_names[i]} of op {dist_op.serial_op} is parameter, op should not use default rule."
            input_specs.append(
                get_dist_tensor_spec(dist_op, input_arg_names[i])
            )
        num_outputs = len(output_arg_names)
        output_specs = []
        for i in range(num_outputs):
            assert not is_parameter_related(
                output_arg_names[i], main_block
            ), f"output {output_arg_names[i]} of op {dist_op.serial_op} is parameter, op should not use default rule."
            output_specs.append(
                get_dist_tensor_spec(dist_op, output_arg_names[i], False)
            )

        # step2: infer spmd
        if contains_spmd_rule(dist_op.serial_op.type):
            # when some inputs are optional, the input_arg_names will be less than input_names
            # and we can pass empty DistTensorSpec() as argument
            if len(op_desc.input_names()) > len(op_desc.input_arg_names()):
                for i in range(
                    len(op_desc.input_names()) - len(op_desc.input_arg_names())
                ):
                    input_specs.append(DistTensorSpec())
            rule = get_phi_spmd_rule(dist_op.serial_op.type)
            fw_results = rule.infer_forward(*input_specs)
            bw_results = rule.infer_backward(*input_specs, output_specs)
        else:
            rule = get_phi_spmd_rule('default_')
            # tensor order following order in PHI definition
            fw_results = rule.infer_forward(input_specs, output_specs)
            bw_results = rule.infer_backward(input_specs, output_specs)

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, input_arg_names, output_arg_names, fw_results, bw_results
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        # all op use default dist operator impl.
        op_dist_attr = dist_op.dist_attr
        default_impl = get_default_distributed_operator_impl()
        op_dist_attr.impl_type = default_impl.type
        op_dist_attr.impl_idx = default_impl.idx

        return False


register_distributed_operator_impl_container(DistributedDefault("default"))


# Replicated Default
class DistributedDefaultImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

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
        batch_dim_mappings = []
        input_names = op_desc.input_names()
        xshape_arg_names = []
        if "XShape" in input_names:
            xshape_arg_names = op_desc.input("XShape")
        for arg_name in op_desc.input_arg_names():
            serial_tensor = dist_op.get_serial_input(arg_name)
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if serial_tensor.is_parameter:
                for mapping in dims_mapping:
                    if mapping != -1:
                        return False
                continue
            if arg_name not in xshape_arg_names:
                if len(dims_mapping) > 1:
                    for mapping in dims_mapping[1:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                if dims_mapping[0] != -1:
                    return False
                if len(dims_mapping) > 2:
                    for mapping in dims_mapping[2:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 2:
                    batch_dim_mappings.append(dims_mapping[1])

        if compute_compatible_dim_mapping(batch_dim_mappings) is None:
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        output_names = op_desc.output_names()
        batch_dim_mappings = []
        xshape_arg_names = []
        if "XShape" in output_names:
            xshape_arg_names = op_desc.output("XShape")
        for arg_name in op_desc.output_arg_names():
            serial_tensor = dist_op.get_serial_output(arg_name)
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if serial_tensor.is_parameter:
                for mapping in dims_mapping:
                    if mapping != -1:
                        return False
                continue
            if arg_name not in xshape_arg_names:
                if len(dims_mapping) > 1:
                    for mapping in dims_mapping[1:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                if dims_mapping[0] != -1:
                    return False
                if len(dims_mapping) > 2:
                    for mapping in dims_mapping[2:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 2:
                    batch_dim_mappings.append(dims_mapping[1])

        if compute_compatible_dim_mapping(batch_dim_mappings) is None:
            return False

        return True

    def is_auto_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        batch_dim_mappings = []
        # Check input compatibility
        input_names = op_desc.input_names()
        xshape_arg_names = []
        if "XShape" in input_names:
            xshape_arg_names = op_desc.input("XShape")
        for arg_name in op_desc.input_arg_names():
            serial_tensor = dist_op.get_serial_input(arg_name)
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if serial_tensor is not None and serial_tensor.is_parameter:
                for mapping in dims_mapping:
                    if mapping != -1:
                        return False
                continue
            if arg_name not in xshape_arg_names:
                if len(dims_mapping) > 1:
                    for mapping in dims_mapping[1:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                if dims_mapping[0] != -1:
                    return False
                if len(dims_mapping) > 2:
                    for mapping in dims_mapping[2:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 2:
                    batch_dim_mappings.append(dims_mapping[1])

        # Check output compatibility
        output_names = op_desc.output_names()
        xshape_arg_names = []
        if "XShape" in output_names:
            xshape_arg_names = op_desc.output("XShape")
        for arg_name in op_desc.output_arg_names():
            serial_tensor = dist_op.get_serial_output(arg_name)
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if serial_tensor is not None and serial_tensor.is_parameter:
                for mapping in dims_mapping:
                    if mapping != -1:
                        return False
                continue
            if arg_name not in xshape_arg_names:
                if len(dims_mapping) > 1:
                    for mapping in dims_mapping[1:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                if dims_mapping[0] != -1:
                    return False
                if len(dims_mapping) > 2:
                    for mapping in dims_mapping[2:]:
                        if mapping != -1:
                            return False
                if len(dims_mapping) >= 2:
                    batch_dim_mappings.append(dims_mapping[1])

        # Check batch dim mapping compatibility
        if not all(
            batch_dim_mappings[0] == dim_mapping
            for dim_mapping in batch_dim_mappings
        ):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        if op_desc.type() == "while":
            return False

        input_names = op_desc.input_names()
        input_xshape_arg_names = []
        if "XShape" in input_names:
            input_xshape_arg_names = op_desc.input("XShape")

        output_names = op_desc.output_names()
        output_xshape_arg_names = []
        if "XShape" in output_names:
            output_xshape_arg_names = op_desc.output("XShape")

        batch_dim_mappings = []
        for arg_name in op_desc.input_arg_names():
            serial_tensor = dist_op.get_serial_input(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if arg_name not in input_xshape_arg_names:
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                batch_dim_mappings.append(dims_mapping[1])
        for arg_name in op_desc.output_arg_names():
            if op_desc.type() == 'fill_any_like':
                input_tensor = dist_op.get_serial_input(
                    op_desc.input_arg_names()[0]
                )
                if input_tensor.is_parameter:
                    continue
            serial_tensor = dist_op.get_serial_output(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if arg_name not in output_xshape_arg_names:
                if len(dims_mapping) >= 1:
                    batch_dim_mappings.append(dims_mapping[0])
            else:
                batch_dim_mappings.append(dims_mapping[1])

        if not batch_dim_mappings:
            return changed

        compatible_dim_mapping = compute_compatible_dim_mapping(
            batch_dim_mappings
        )
        if compatible_dim_mapping is None:
            return False

        for arg_name in op_desc.input_arg_names():
            serial_tensor = dist_op.get_serial_input(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if arg_name not in input_xshape_arg_names:
                if (
                    len(dims_mapping) >= 1
                    and compatible_dim_mapping != dims_mapping[0]
                ):
                    dims_mapping[0] = compatible_dim_mapping
                    op_dist_attr.set_input_dims_mapping(arg_name, dims_mapping)
                    changed = True
            else:
                if (
                    len(dims_mapping) >= 2
                    and compatible_dim_mapping != dims_mapping[1]
                ):
                    dims_mapping[1] = compatible_dim_mapping
                    op_dist_attr.set_input_dims_mapping(arg_name, dims_mapping)
                    changed = True
        for arg_name in op_desc.output_arg_names():
            if op_desc.type() == 'fill_any_like':
                input_tensor = dist_op.get_serial_input(
                    op_desc.input_arg_names()[0]
                )
                if input_tensor.is_parameter:
                    continue
            if op_desc.type() in ["shape", "slice"]:
                continue
            serial_tensor = dist_op.get_serial_output(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if arg_name not in output_xshape_arg_names:
                if (
                    len(dims_mapping) >= 1
                    and compatible_dim_mapping != dims_mapping[0]
                ):
                    dims_mapping[0] = compatible_dim_mapping
                    op_dist_attr.set_output_dims_mapping(arg_name, dims_mapping)
                    changed = True
            else:
                if (
                    len(dims_mapping) >= 2
                    and compatible_dim_mapping != dims_mapping[1]
                ):
                    dims_mapping[1] = compatible_dim_mapping
                    op_dist_attr.set_output_dims_mapping(arg_name, dims_mapping)
                    changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id

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

        # replicate op in dist program
        dst_op = copy_op_without_infer_shape(src_op, main_block, ctx, kwargs)

        def get_shape_attr_name():
            for name in ["shape", "target_shape"]:
                if src_op.has_attr(name) and src_op.attr(name):
                    return name
            return None

        shape_attr_name = get_shape_attr_name()
        if shape_attr_name and src_op.type in __op_has_shape_attr__:
            shape_list = src_op.attr(shape_attr_name)
            Out_var = main_block._var_recursive(kwargs['Out'][0])
            op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
            dim_mapping = op_dist_attr.get_output_dims_mapping(Out_var.name)
            process_mesh_shape = op_dist_attr.process_mesh.shape
            assert len(shape_list) == len(dim_mapping)
            # modify target shape
            for idx, axis in enumerate(dim_mapping):
                if axis >= 0:
                    if len(shape_list) > idx:
                        shape_list[idx] = (
                            shape_list[idx] // process_mesh_shape[axis]
                        )
            dst_op.desc._set_attr(shape_attr_name, shape_list)

        # data parallel synchronization for primitive operators
        from paddle.incubate.autograd import prim_enabled

        if prim_enabled():
            assert is_prim_op(src_op)
            prim_operator_data_parallel_functor(ctx, src_op)
            return

        # param initialization sync
        if src_op.type in __op_not_need_param_init__:
            return

        for varname in dst_op.desc.input_arg_names():
            if (
                startup_block.has_var(varname)
                and startup_block.var(varname).is_parameter
                and varname not in dist_op_context.already_init_sync_vars
            ):
                dist_op_context.already_init_sync_vars.add(varname)
                param = startup_block.var(varname)
                param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
                process_mesh = param_dist_attr.process_mesh
                dims_mapping = param_dist_attr.dims_mapping

                # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
                if rank_id not in process_mesh.process_ids:
                    rank_id = _get_corresponding_rank(
                        ctx, process_mesh, rank_id
                    )

                # NOTE all not splited axis should be presented in mesh
                for axis, size in enumerate(process_mesh.shape):
                    if size <= 1 or axis in dims_mapping:
                        pass
                    else:
                        group_ranks = _get_comm_group(
                            process_mesh.process_ids,
                            process_mesh.shape,
                            axis,
                            rank_id,
                        )
                        sync_group = new_process_group(group_ranks)

                        new_op = startup_block.append_op(
                            type='broadcast',
                            inputs={'x': param},
                            outputs={'out': param},
                            attrs={
                                'ring_id': sync_group.id,
                                'root': 0,
                                OP_ROLE_KEY: OpRole.Forward,
                            },
                        )
                        set_comm_op_dist_attr_for_program(
                            new_op,
                            process_mesh,
                            param_dist_attr,
                            ctx,
                        )

    @staticmethod
    def backward(ctx, *args, **kwargs):
        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert (
            dist_attr is not None
        ), f"backward op [{backward_op}] don't have dist attribute !"
        rank_id = dist_op_context.rank_id

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

        # replicate op in dist program
        copy_op_without_infer_shape(backward_op, main_block, ctx, kwargs)

        # data parallel gradient synchronization
        act_grad_names = []
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                    varname, main_block
                ):
                    act_grad_names.append(varname)

        out_grad_names = []
        for output_name in backward_op.desc.output_names():
            for varname in backward_op.desc.output(output_name):
                if varname in kwargs["grad_var_to_var"]:
                    fwd_name = kwargs["grad_var_to_var"][varname]
                    if not main_block._find_var_recursive(fwd_name):
                        continue
                    if is_parameter_related(fwd_name, main_block):
                        out_grad_names.append(varname)

        gradient_synchronization(
            ctx, backward_op, act_grad_names, out_grad_names, rank_id
        )


register_distributed_operator_impl(
    "default", DistributedDefaultImpl0("replicate_parallel")
)
