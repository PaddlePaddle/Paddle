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

from .common import DistributedOperator
from .common import DistributedOperatorImpl
from .common import register_distributed_operator
from .common import register_distributed_operator_impl
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from ..process import new_process_group
from ..utils import _get_comm_group


class DistributedDefault(DistributedOperator):
    def __init__(self, name):
        super(DistributedDefault, self).__init__()
        self._name = name


register_distributed_operator("default", DistributedDefault("default"))


# Replicated Default 
class DistributedDefaultImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedDefaultImpl0, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = True

    def is_process_mesh_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def is_input_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def is_output_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def update_dims_mapping(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dst_block = ctx.get_dst_main_program().global_block()
        src_op = ctx.get_cur_src_op()
        varname_mapping = ctx.get_varname_mapping()

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
        dist_op_desc = dst_block.desc.append_op()
        dist_op_desc.copy_from(src_op.desc)
        for input_name in src_op.desc.input_names():
            dist_op_desc.set_input(input_name, kwargs[input_name])
        for output_name in src_op.desc.output_names():
            dist_op_desc.set_output(output_name, kwargs[output_name])

        dst_block._sync_with_cpp()

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself

        dst_block = ctx.get_dst_main_program().global_block()
        backward_op = ctx.get_cur_src_op()
        dist_attr = ctx.get_cur_dist_attr()
        rank_id = ctx.get_rank_id()
        # check if need gradient allreduce
        # if there is a non-gradient & non-parameter input and its batch dimension is splited, 
        # we need insert gradient allreduce for the gradient of parameter in its output
        need_gradient_allreduce = False
        for input_name in backward_op.desc.input_names():
            for varname in backward_op.desc.input(input_name):
                if "@GRAD" not in varname and not dst_block.var(
                        varname).is_parameter():

                    # NOTE input var's dim_mapping of backward op should be the same with input var instead of corresponding varname of forward op
                    process_mesh = dist_attr.get_process_mesh()
                    var_dim_mapping = dist_attr.get_input_dims_mapping(varname)
                    mesh_shape = process_mesh.topology
                    batch_size_axis = var_dim_mapping[0]
                    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                        need_gradient_allreduce = True
                        group_ranks = _get_comm_group(
                            process_mesh.process_group, process_mesh.topology,
                            batch_size_axis, rank_id)
                        dp_degree = len(group_ranks)
                        dp_group = new_process_group(group_ranks)
                        break

        if need_gradient_allreduce:
            allreduce_vars = []
            for input_name in backward_op.desc.input_names():
                for varname in backward_op.desc.input(input_name):
                    if "@GRAD" not in varname and dst_block.vars(
                            varname).is_parameter():
                        assert len(
                            backward_op.desc.input(input_name)
                        ) == 1, "parameter input to grad op should be length 1, but got [{}]".format(
                            backward_op.desc.input(input_name))
                        assert varname + "@GRAD" in backward_op.desc.output_names(
                        ), "parameter's grad [{}] not found in the grad op's output".format(
                            varname + "@GRAD")
                        assert len(
                            backward_op.desc.output_names(varname + "@GRAD")
                        ) == 1, "parameter grad of grad op should be length 1, but got [{}]".format(
                            backward_op.desc.output_names(varname + "@GRAD"))
                        allreduce_vars.append(
                            backward_op.desc.output_names(varname + "@GRAD")[0])

            if len(allreduce_vars) > 0:
                for varname in allreduce_vars:

                    grad_var = dst_block.vars(varname)
                    dst_block.append_op(
                        type='c_allreduce_sum',
                        inputs={'X': [grad_var]},
                        outputs={'Out': [grad_var]},
                        attrs={
                            'ring_id': dp_group.id,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                    dst_block.append_op(
                        type='scale',
                        inputs={'X': grad_var},
                        outputs={'Out': grad_var},
                        attrs={
                            'scale': 1.0 / dp_degree,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                dst_block._sync_with_cpp()


register_distributed_operator_impl(
    "default", DistributedDefaultImpl0("replicate_parallel"))
