# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

from paddle.distributed.fleet.meta_optimizers.common import OpRole
from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from ..utils import is_dim_shard
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from .dist_default import DistributedDefaultImpl0
from ..cost import SplitOpCost
from ..cost import build_comp_desc_from_dist_op
from ..cost import build_comp_costs_from_desc_mapping


class DistributedSplit(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedSplit, self).__init__(op_type)


register_distributed_operator_impl_container(DistributedSplit("split"))


class DistributedSplitImpl(DistributedOperatorImpl):

    def __init__(self, name):
        super(DistributedSplitImpl, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Backward):
            raise NotImplementedError(
                "The backward cost of dist split has not implemented.")
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
        cost_mapping = build_comp_costs_from_desc_mapping(
            SplitOpCost, ctx, processes, desc_mapping, cluster)

        res_cost = [cost_mapping]
        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)

        if is_dim_shard(x_dims_mapping[axis]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_names = op_desc.output('Out')
        axis = op_desc.attr('axis')
        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            if is_dim_shard(out_dims_mapping[axis]):
                return False

        return True

    def is_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        axis = op_desc.attr('axis')
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            if x_dims_mapping != out_dims_mapping:
                return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)

        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            for i in range(len(x_dims_mapping)):
                dim_changed = compute_compatible_and_update_dim_mapping(
                    [x_dims_mapping, out_dims_mapping], [i, i])
                if dim_changed:
                    changed = True

        return changed

    def is_auto_compatible(self, dist_op):
        raise NotImplementedError(
            "Auto Search is not supported by dist split yet.")

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl("split",
                                   DistributedSplitImpl("replicate_in_axis"))
