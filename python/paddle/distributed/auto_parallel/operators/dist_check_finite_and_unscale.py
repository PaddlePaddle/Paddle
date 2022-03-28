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
from paddle.fluid import core, unique_name
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..utils import set_var_dist_attr
from ..utils import set_dist_op_desc_original_id
from ..process_group import new_process_group
from ..dist_attribute import OperatorDistributedAttribute
from paddle.distributed.auto_parallel.process_group import get_world_process_group

world_process_group = get_world_process_group()


class DistributedCheckFiniteAndUnscale(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedCheckFiniteAndUnscale, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedCheckFiniteAndUnscale("check_finite_and_unscale"))


class DistributedCheckFiniteAndUnscaleImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedCheckFiniteAndUnscaleImpl, self).__init__(name)
        self._name = name
        self._forward_implemented = False
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        raise RuntimeError(
            "DistributedCheckFiniteAndUnscaleImpl's is_input_compatible should not be called !"
        )

    def is_output_compatible(self, dist_op):
        raise RuntimeError(
            "DistributedCheckFiniteAndUnscaleImpl's is_output_compatible should not be called !"
        )

    def is_auto_compatible(self, dist_op):
        raise RuntimeError(
            "DistributedCheckFiniteAndUnscaleImpl's is_auto_compatible should not be called !"
        )

    def update_dims_mapping(self, dist_op):
        raise RuntimeError(
            "DistributedCheckFiniteAndUnscaleImpl's update_dims_mapping should not be called !"
        )

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise RuntimeError(
            "DistributedCheckFiniteAndUnscaleImpl's forward should not be called !"
        )

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.main_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))

        assert rank_id in dist_attr.process_mesh.processes

        assert 'X' in kwargs, "input [{}] is not given".format('X')
        assert 'Scale' in kwargs, "input [{}] is not given".format('Scale')
        assert 'Out' in kwargs, "input [{}] is not given".format('Out')
        assert 'FoundInfinite' in kwargs, "output [{}] is not given".format(
            'FoundInfinite')

        assert len(
            kwargs['Scale']
        ) == 1, "check_finite_and_unscale input Scale take 1 variable but got {}".format(
            kwargs['Scale'])
        assert len(
            kwargs['FoundInfinite']
        ) == 1, "check_finite_and_unscale input FoundInfinite take 1 variable but got {}".format(
            kwargs['FoundInfinite'])
        assert len(kwargs['X']) == len(
            kwargs['Out']
        ), "check_finite_and_unscale got [{}] X and [{}] Out, which are supposed to be equal".format(
            len(kwargs['X']), len(kwargs['Out']))

        filter_vars = []
        for varname in kwargs['X']:
            if rank_id in ctx.get_tensor_dist_attr_for_program(
                    main_block.var(varname)).process_mesh.processes:
                filter_vars.append(varname)

        # replicate op in dist program
        dist_op_desc = main_block.desc.append_op()
        dist_op_desc.copy_from(backward_op.desc)
        set_dist_op_desc_original_id(dist_op_desc, backward_op.desc, ctx)
        dist_op_desc.set_input('X', filter_vars)
        dist_op_desc.set_output('Out', filter_vars)
        main_block._sync_with_cpp()

        # sync result
        group = new_process_group(world_process_group.ranks)

        inf_var = main_block.var(kwargs['FoundInfinite'][0])
        inf_var_int32 = main_block.create_var(
            name=inf_var.name + "@cast_int32",
            shape=inf_var.shape,
            dtype=core.VarDesc.VarType.INT32)
        set_var_dist_attr(
            ctx, inf_var_int32,
            ctx.get_tensor_dist_attr_for_program(inf_var).dims_mapping,
            ctx.get_tensor_dist_attr_for_program(inf_var).process_mesh)
        cast_op1 = main_block.append_op(
            type='cast',
            inputs={'X': inf_var},
            outputs={'Out': inf_var_int32},
            attrs={
                "in_dtype": inf_var.dtype,
                "out_dtype": inf_var_int32.dtype,
                OP_ROLE_KEY: OpRole.Backward
            })
        allreduce_op = main_block.append_op(
            type='c_allreduce_max',
            inputs={'X': inf_var_int32},
            outputs={'Out': inf_var_int32},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward
            })
        cast_op2 = main_block.append_op(
            type='cast',
            inputs={'X': inf_var_int32},
            outputs={'Out': inf_var},
            attrs={
                "in_dtype": inf_var_int32.dtype,
                "out_dtype": inf_var.dtype,
                OP_ROLE_KEY: OpRole.Backward
            })
        main_block._sync_with_cpp()

        for op in [cast_op1, allreduce_op, cast_op2]:
            new_op_dist_attr = OperatorDistributedAttribute()
            for varname in op.input_arg_names:
                var_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    main_block.var(varname))
                assert var_dist_attr is not None
                new_op_dist_attr.set_input_dims_mapping(
                    varname, var_dist_attr.dims_mapping)
            for varname in op.output_arg_names:
                var_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    main_block.var(varname))
                new_op_dist_attr.set_output_dims_mapping(
                    varname, var_dist_attr.dims_mapping)
            new_op_dist_attr.process_mesh = var_dist_attr.process_mesh
            ctx.set_op_dist_attr_for_program(op, new_op_dist_attr)


register_distributed_operator_impl(
    "check_finite_and_unscale",
    DistributedCheckFiniteAndUnscaleImpl("check_finite_and_unscale"))
