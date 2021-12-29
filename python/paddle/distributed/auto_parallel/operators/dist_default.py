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
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..utils import set_dist_op_desc_original_id
from ..dist_attribute import OperatorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank


class DistributedDefault(DistributedOperatorImplContainer):
    def __init__(self, name):
        super(DistributedDefault, self).__init__()
        self._name = name


register_distributed_operator_impl_container("default",
                                             DistributedDefault("default"))


# Replicated Default
class DistributedDefaultImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedDefaultImpl0, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method.")

    def is_output_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method.")

    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method.")

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        startup_block = dist_op_context.get_dst_startup_program().global_block()
        src_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()

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
        dist_op_desc = main_block.desc.append_op()
        dist_op_desc.copy_from(src_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        main_block._sync_with_cpp()

        # param initialization sync
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

                        new_op = startup_block.append_op(
                            type='c_broadcast',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={
                                'ring_id': sync_group.id,
                                'root': 0,
                                'use_calc_stream': True,
                                OP_ROLE_KEY: OpRole.Forward
                            })

                        # set distributed attribute
                        op_attr = OperatorDistributedAttribute()
                        op_attr.process_mesh = process_mesh
                        op_attr.set_output_dims_mapping(param.name,
                                                        dims_mapping)
                        op_attr.set_input_dims_mapping(param.name, dims_mapping)
                        ctx.set_op_dist_attr_for_program(new_op, op_attr)

                startup_block._sync_with_cpp()

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        backward_op = dist_op_context.get_cur_src_op()
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))
        rank_id = dist_op_context.get_rank_id()

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
        dist_op_desc = main_block.desc.append_op()
        dist_op_desc.copy_from(backward_op.desc)
        # Refer to the related dist op
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        for input_name in backward_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in backward_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        main_block._sync_with_cpp()

        # check if need gradient allreduce
        # if there is a non-gradient & non-parameter input and its batch dimension is splited,
        # we need insert gradient allreduce for the gradient of parameter in its output
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not main_block.var(
                        varname).is_parameter:

                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    process_mesh = dist_attr.process_mesh
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)

                    # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
                    if rank_id not in process_mesh.processes:
                        rank_id = _get_corresponding_rank(ctx, process_mesh,
                                                          rank_id)

                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        need_gradient_allreduce = True
                        group_ranks = _get_comm_group(process_mesh.processes,
                                                      process_mesh.topology,
                                                      batch_size_axis, rank_id)
                        dp_degree = len(group_ranks)
                        dp_group = new_process_group(group_ranks)
                        break

        if need_gradient_allreduce:
            allreduce_vars = []
            for input_name in backward_op.desc.input_names():
                for varname in backward_op.desc.input(input_name):
                    if "@GRAD" not in varname and main_block.var(
                            varname).is_parameter:
                        assert len(
                            backward_op.desc.input(input_name)
                        ) == 1, "parameter input to grad op should be length 1, but got [{}]".format(
                            backward_op.desc.input(input_name))

                        assert varname + "@GRAD" in backward_op.desc.output_arg_names(
                        ), "parameter's grad [{}] not found in the grad op's output".format(
                            varname + "@GRAD")
                        assert len(
                            backward_op.desc.output(input_name + "@GRAD")
                        ) == 1, "parameter grad of grad op should be length 1, but got [{}]".format(
                            backward_op.desc.output(input_name + "@GRAD"))
                        allreduce_vars.append(
                            backward_op.desc.output(input_name + "@GRAD")[0])

            if len(allreduce_vars) > 0:

                for varname in allreduce_vars:

                    grad_var = main_block.var(varname)
                    allreduce_op = main_block.append_op(
                        type='c_allreduce_sum',
                        inputs={'X': [grad_var]},
                        outputs={'Out': [grad_var]},
                        attrs={
                            'ring_id': dp_group.id,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Backward
                        })

                    scale_op = main_block.append_op(
                        type='scale',
                        inputs={'X': grad_var},
                        outputs={'Out': grad_var},
                        attrs={
                            'scale': 1.0 / dp_degree,
                            OP_ROLE_KEY: OpRole.Backward
                        })

                    dims_mapping = ctx.get_tensor_dist_attr_for_program(
                        grad_var).dims_mapping
                    process_mesh = dist_attr.process_mesh
                    for op in [allreduce_op, scale_op]:
                        op_attr = OperatorDistributedAttribute()
                        op_attr.process_mesh = process_mesh
                        op_attr.set_output_dims_mapping(grad_var.name,
                                                        dims_mapping)
                        op_attr.set_input_dims_mapping(grad_var.name,
                                                       dims_mapping)
                        ctx.set_op_dist_attr_for_program(op, op_attr)

                main_block._sync_with_cpp()


register_distributed_operator_impl(
    "default", DistributedDefaultImpl0("replicate_parallel"))
