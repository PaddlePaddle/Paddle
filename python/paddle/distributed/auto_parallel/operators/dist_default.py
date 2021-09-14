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
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def is_input_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def is_output_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def update_dims_mapping(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method.")

    def forward(ctx, *args, **kwargs):

        dst_block = ctx.get_dist_block()
        src_op = ctx.get_src_op()
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
                src_op.desc.input(output_name)
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


register_distributed_operator_impl(
    "default", DistributedDefaultImpl0("replicate_parallel"))
