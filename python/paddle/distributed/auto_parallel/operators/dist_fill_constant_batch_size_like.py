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
from paddle.fluid import core, unique_name
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from .dist_default import DistributedDefaultImpl0


class DistributedFillConstantBatchSizeLike(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedFillConstantBatchSizeLike, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedFillConstantBatchSizeLike("fill_constant_batch_size_like"))


class DistributedFillConstantBatchSizeLikeImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedFillConstantBatchSizeLikeImpl0, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        shape_list = op_desc.attr("shape")

        if len(shape_list) != len(out_dims_mapping):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        in_name = op_desc.input('Input')[0]
        in_dims_mapping = op_dist_attr.get_input_dims_mapping(in_name)

        # the dim_mapping of batch dimension should be the same
        return out_dims_mapping[0] == in_dims_mapping[0]

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        # only the batch size dimemsion of input and output are relative.
        dim_changed = compute_compatible_and_update_dim_mapping(
            [x_dims_mapping, out_dims_mapping], [0, 0])
        if dim_changed:
            changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)
        dist_op_context = ctx.dist_op_context
        src_op = dist_op_context.cur_src_op
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        main_block = dist_op_context.work_block
        op = main_block.ops[-1]
        assert op.type == "fill_constant_batch_size_like"

        # modify shape attr according to how output are partitioned
        out_name = op.output('Out')[0]
        dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        shape_list = op.attr("shape")
        # modify target shape
        for idx, axis in enumerate(dims_mapping):
            if axis >= 0:
                shape_list[idx] = shape_list[idx] // process_mesh_shape[axis]

        op._set_attr("shape", shape_list)
        main_block._sync_with_cpp()

    @staticmethod
    def backward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "fill_constant_batch_size_like",
    DistributedFillConstantBatchSizeLikeImpl0("fill_by_shape"))
