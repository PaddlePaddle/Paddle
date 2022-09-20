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
from .common import gradient_synchronization
from .common import register_distributed_operator_impl, is_parameter_related
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index, is_prim_op
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..utils import set_dist_op_desc_original_id
from ..dist_attribute import OperatorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank
from ..cost import _g_op_cost_factory
from ..cost import build_comp_desc_from_dist_op, build_dp_costs
from ..cost import build_comp_costs_from_descs

__op_not_need_param_init__ = ["while", "cond"]


def prim_operator_data_parallel_functor(ctx, src_op):
    dist_op_context = ctx.dist_op_context
    main_block = dist_op_context.work_block
    startup_block = dist_op_context.startup_block

    var_name = src_op.output_arg_names[0]
    if var_name in ctx.grads_params:
        assert var_name not in ctx.synced_gradient, "in primtive mode, grad is already {} synced".format(
            var_name)
        ctx.synced_gradient.add(var_name)
        sync_group = new_process_group(ctx.data_parallel_group)

        allreduce_op = main_block.append_op(type='c_allreduce_sum',
                                            inputs={'X': [var_name]},
                                            outputs={'Out': [var_name]},
                                            attrs={
                                                'ring_id': sync_group.id,
                                                'use_calc_stream': True,
                                                OP_ROLE_KEY: OpRole.Backward
                                            })

        param = ctx.grads_params[var_name]
        startup_block = dist_op_context.startup_block
        new_op = startup_block.append_op(type='c_broadcast',
                                         inputs={'X': [param]},
                                         outputs={'Out': [param]},
                                         attrs={
                                             'ring_id': sync_group.id,
                                             'root': 0,
                                             'use_calc_stream': True,
                                             OP_ROLE_KEY: OpRole.Forward
                                         })

        grad_var = main_block.var(var_name)
        dims_mapping = ctx.get_tensor_dist_attr_for_program(
            grad_var).dims_mapping
        dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        process_mesh = dist_attr.process_mesh
        op_attr = OperatorDistributedAttribute()
        op_attr.process_mesh = process_mesh
        op_attr.set_output_dims_mapping(grad_var.name, dims_mapping)
        op_attr.set_input_dims_mapping(grad_var.name, dims_mapping)
        ctx.set_op_dist_attr_for_program(allreduce_op, op_attr)

    return


class DistributedDefault(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedDefault, self).__init__(op_type)


register_distributed_operator_impl_container(DistributedDefault("default"))


# Replicated Default
class DistributedDefaultImpl0(DistributedOperatorImpl):

    def __init__(self, name):
        super(DistributedDefaultImpl0, self).__init__(name)
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
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        op_type = dist_op.serial_op.type
        cost_mapping = build_comp_costs_from_descs(_g_op_cost_factory[op_type],
                                                   ctx, processes, desc_mapping,
                                                   cluster)
        res_cost = [cost_mapping]

        return res_cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        res = []
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        backward_op = dist_op.serial_op
        op_type = backward_op.type
        cost_mapping = build_comp_costs_from_descs(_g_op_cost_factory[op_type],
                                                   ctx, processes, desc_mapping,
                                                   cluster)
        res.append(cost_mapping)

        main_block = backward_op.block
        vars = main_block.vars
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                        varname, main_block):
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)
                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        need_gradient_allreduce = True
                        break

        if need_gradient_allreduce:
            for input_name in backward_op.desc.input_names():
                for varname in backward_op.desc.input(input_name):
                    if "@GRAD" not in varname and is_parameter_related(
                            varname, main_block):
                        var_dim_mapping = dist_attr.get_input_dims_mapping(
                            varname)
                        mesh_shape = process_mesh.topology
                        batch_size_axis = var_dim_mapping[0]
                        parallel_axis = batch_size_axis
                        attrs = {"use_calc_stream": True}
                        var_names = [varname + "@GRAD"]
                        build_dp_costs(res, dist_op, ctx, var_names, attrs,
                                       parallel_axis, cluster)
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
        if not all(batch_dim_mappings[0] == dim_mapping
                   for dim_mapping in batch_dim_mappings):
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
                    op_desc.input_arg_names()[0])
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
            batch_dim_mappings)
        if compatible_dim_mapping is None:
            return False

        for arg_name in op_desc.input_arg_names():
            serial_tensor = dist_op.get_serial_input(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
            if arg_name not in input_xshape_arg_names:
                if len(dims_mapping) >= 1 and \
                    compatible_dim_mapping != dims_mapping[0]:
                    dims_mapping[0] = compatible_dim_mapping
                    changed = True
            else:
                if len(dims_mapping) >= 2 and \
                    compatible_dim_mapping != dims_mapping[1]:
                    dims_mapping[1] = compatible_dim_mapping
                    changed = True
        for arg_name in op_desc.output_arg_names():
            if op_desc.type() == 'fill_any_like':
                input_tensor = dist_op.get_serial_input(
                    op_desc.input_arg_names()[0])
                if input_tensor.is_parameter:
                    continue
            if op_desc.type() in ["shape", "slice"]:
                continue
            serial_tensor = dist_op.get_serial_output(arg_name)
            if serial_tensor.is_parameter:
                continue
            dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
            if arg_name not in output_xshape_arg_names:
                if len(dims_mapping
                       ) >= 1 and compatible_dim_mapping != dims_mapping[0]:
                    dims_mapping[0] = compatible_dim_mapping
                    changed = True
            else:
                if len(dims_mapping
                       ) >= 2 and compatible_dim_mapping != dims_mapping[1]:
                    dims_mapping[1] = compatible_dim_mapping
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

        # replicate op in dist program
        dist_op_desc = main_block.append_op(type='nop').desc
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        # data parallel synchronization for primtive operators
        from paddle.incubate.autograd import prim_enabled
        if prim_enabled():
            assert is_prim_op(src_op)
            prim_operator_data_parallel_functor(ctx, src_op)
            return

        # param initialization sync
        if src_op.type in __op_not_need_param_init__:
            return

        for varname in dist_op_desc.input_arg_names():
            if startup_block.has_var(varname) and startup_block.var(
                    varname
            ).is_parameter and varname not in dist_op_context.already_init_sync_vars:
                dist_op_context.already_init_sync_vars.add(varname)
                param = startup_block.var(varname)
                param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
                process_mesh = param_dist_attr.process_mesh
                dims_mapping = param_dist_attr.dims_mapping

                # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
                if rank_id not in process_mesh.processes:
                    rank_id = _get_corresponding_rank(ctx, process_mesh,
                                                      rank_id)

                # NOTE all not splited axis should be presented in mesh
                for axis, size in enumerate(process_mesh.topology):
                    if size <= 1 or axis in dims_mapping:
                        pass
                    else:
                        group_ranks = _get_comm_group(process_mesh.processes,
                                                      process_mesh.topology,
                                                      axis, rank_id)
                        sync_group = new_process_group(group_ranks)

                        new_op = startup_block.append_op(type='c_broadcast',
                                                         inputs={'X': param},
                                                         outputs={'Out': param},
                                                         attrs={
                                                             'ring_id':
                                                             sync_group.id,
                                                             'root':
                                                             0,
                                                             'use_calc_stream':
                                                             True,
                                                             OP_ROLE_KEY:
                                                             OpRole.Forward
                                                         })

                        # set distributed attribute
                        op_attr = OperatorDistributedAttribute()
                        op_attr.process_mesh = process_mesh
                        op_attr.set_output_dims_mapping(param.name,
                                                        dims_mapping)
                        op_attr.set_input_dims_mapping(param.name, dims_mapping)
                        ctx.set_op_dist_attr_for_program(new_op, op_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))
        rank_id = dist_op_context.rank_id

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

        # replicate op in dist program
        dist_op_desc = main_block.append_op(type='nop').desc
        dist_op_desc.copy_from(backward_op.desc)
        # Refer to the related dist op
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        for input_name in backward_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in backward_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        # data parallel gradient synchronization
        act_grad_names = []
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not is_parameter_related(
                        varname, main_block):
                    act_grad_names.append(varname)

        out_grad_names = []
        for output_name in backward_op.desc.output_names():
            for varname in backward_op.desc.output(output_name):
                if varname in kwargs["grad_var_to_var"]:
                    fwd_name = kwargs["grad_var_to_var"][varname]
                    if fwd_name not in main_block.vars:
                        continue
                    if is_parameter_related(fwd_name, main_block):
                        out_grad_names.append(varname)

        gradient_synchronization(ctx, backward_op, act_grad_names,
                                 out_grad_names, rank_id)


register_distributed_operator_impl(
    "default", DistributedDefaultImpl0("replicate_parallel"))
