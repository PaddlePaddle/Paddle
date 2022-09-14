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
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
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


class DistributedReduceSumPrimtive(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedReduceSumPrimtive, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedReduceSumPrimtive("reduce_sum_p"))


# Batch Dimension ReduceSum Primitive
class DistributedReduceSumPrimtiveImpl0(DistributedOperatorImpl):

    def __init__(self, name):
        super(DistributedReduceSumPrimtiveImpl0, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        return len(op_desc.input_arg_names()) == 1

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        outputs = op_desc.output_arg_names()

        if len(outputs) != 1:
            return False

        output_name = outputs[0]
        output_var = dist_op.serial_op.block.var(output_name)
        if output_var.shape != (1, ):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        return self.is_input_compatible(dist_op) and self.is_output_compatible(
            dist_op)

    def update_dims_mapping(self, dist_op):
        changed = False

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

        # batch dimension synchronization
        var_name = src_op.output_arg_names[0]
        sync_group = new_process_group(ctx.data_parallel_group)
        allreduce_op = main_block.append_op(type='c_allreduce_sum',
                                            inputs={'X': [var_name]},
                                            outputs={'Out': [var_name]},
                                            attrs={
                                                'ring_id': sync_group.id,
                                                'use_calc_stream': True,
                                                OP_ROLE_KEY: OpRole.Forward
                                            })

        # dist attr
        var = main_block.var(var_name)
        tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(var)
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        new_op_attr = OperatorDistributedAttribute()
        new_op_attr.process_mesh = op_dist_attr.process_mesh
        new_op_attr.set_output_dims_mapping(var.name,
                                            tensor_dist_attr.dims_mapping)
        new_op_attr.set_input_dims_mapping(var.name,
                                           tensor_dist_attr.dims_mapping)
        ctx.set_op_dist_attr_for_program(allreduce_op, new_op_attr)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise RuntimeError(
            "primitive operator does NOT have backward function, op type: {}".
            format(str(op.type)))


register_distributed_operator_impl(
    "reduce_sum_p",
    DistributedReduceSumPrimtiveImpl0("batch_dimension_reduce_sum_p"))
