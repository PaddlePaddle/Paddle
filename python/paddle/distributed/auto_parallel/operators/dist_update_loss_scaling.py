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
from ..utils import set_dist_op_desc_original_id


class DistributedUpdateLossScaling(DistributedOperatorImplContainer):
    def __init__(self, name):
        super(DistributedUpdateLossScaling, self).__init__()
        self._name = name


register_distributed_operator_impl_container(
    "update_loss_scaling", DistributedUpdateLossScaling("update_loss_scaling"))


class DistributedUpdateLossScalingImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedUpdateLossScalingImpl, self).__init__()
        self._name = name
        self._forward_implemented = False
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        raise RuntimeError(
            "DistributedUpdateLossScalingImpl's is_input_compatible should not be called !"
        )

    def is_output_compatible(self, dist_op):
        raise RuntimeError(
            "DistributedUpdateLossScalingImpl's is_output_compatible should not be called !"
        )

    def update_dims_mapping(self, dist_op):
        raise RuntimeError(
            "DistributedUpdateLossScalingImpl's update_dims_mapping should not be called !"
        )

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise RuntimeError(
            "DistributedUpdateLossScalingImpl's forward should not be called !")

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # the backward function only filte the gradient with current rank id
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        backward_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))

        assert rank_id in dist_attr.process_mesh.processes

        assert 'X' in kwargs, "input [{}] is not given".format('X')
        assert 'FoundInfinite' in kwargs, "input [{}] is not given".format(
            'FoundInfinite')
        assert 'PrevLossScaling' in kwargs, "input [{}] is not given".format(
            'PrevLossScaling')
        assert 'InGoodSteps' in kwargs, "input [{}] is not given".format(
            'InGoodSteps')
        assert 'InBadSteps' in kwargs, "input [{}] is not given".format(
            'InBadSteps')

        assert 'Out' in kwargs, "output [{}] is not given".format('Out')
        assert 'LossScaling' in kwargs, "output [{}] is not given".format(
            'LossScaling')
        assert 'OutGoodSteps' in kwargs, "output [{}] is not given".format(
            'OutGoodSteps')
        assert 'OutBadSteps' in kwargs, "output [{}] is not given".format(
            'OutBadSteps')

        assert len(kwargs['FoundInfinite']) == 1, \
            "update_loss_scaling input FoundInfinite take 1 variable but got {}".format(
            kwargs['FoundInfinite'])
        assert len(kwargs['PrevLossScaling']) == 1, \
            "update_loss_scaling input PrevLossScaling take 1 variable but got {}".format(
            kwargs['PrevLossScaling'])
        assert len(kwargs['InGoodSteps']) == 1, \
            "update_loss_scaling input InGoodSteps take 1 variable but got {}".format(
            kwargs['InGoodSteps'])
        assert len(kwargs['InBadSteps']) == 1, \
            "update_loss_scaling input InBadSteps take 1 variable but got {}".format(
            kwargs['InBadSteps'])
        assert len(kwargs['LossScaling']) == 1, \
            "update_loss_scaling output LossScaling take 1 variable but got {}".format(
            kwargs['LossScaling'])
        assert len(kwargs['OutGoodSteps']) == 1, \
            "update_loss_scaling output OutGoodSteps take 1 variable but got {}".format(
            kwargs['OutGoodSteps'])
        assert len(kwargs['OutBadSteps']) == 1, \
            "update_loss_scaling output OutBadSteps take 1 variable but got {}".format(
            kwargs['OutBadSteps'])

        assert len(kwargs['X']) == len(kwargs['Out']), \
            "update_loss_scaling got [{}] X and [{}] Out, which are supposed to be equal".format(
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


register_distributed_operator_impl(
    "update_loss_scaling",
    DistributedUpdateLossScalingImpl("update_loss_scaling"))
