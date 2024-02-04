# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from ..completion import get_phi_spmd_rule
from ..utils import get_dist_tensor_spec
from .common import (
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)


class DistributedUnSqueeze2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc

        axes_tensor = op_desc.input('AxesTensor')
        axes_tensor_list = op_desc.input('AxesTensorList')
        assert len(axes_tensor) == 0 and len(axes_tensor_list) == 0

        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        xshape_name = op_desc.output('XShape')[0]
        axes = op_desc.attr('axes')

        input_spec = get_dist_tensor_spec(dist_op, x_name)
        output_spec = get_dist_tensor_spec(dist_op, out_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("unsqueeze2")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(input_spec, axes)
        bw_results = rule.infer_backward(input_spec, output_spec, axes)

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op,
            [x_name],
            [out_name, xshape_name],
            fw_results,
            bw_results,
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        op_dist_attr = dist_op.dist_attr
        default_impl = get_default_distributed_operator_impl()
        op_dist_attr.impl_type = default_impl.type
        op_dist_attr.impl_idx = default_impl.idx

        return False


register_distributed_operator_impl_container(
    DistributedUnSqueeze2("unsqueeze2")
)
