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

import copy
import logging

from paddle.base.log_helper import get_logger

from ..completion import get_phi_spmd_rule
from ..utils import get_dist_tensor_spec, is_dim_shard
from .common import (
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class DistributedLayerNorm(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc

        x_name = op_desc.input('X')[0]
        scale_name = op_desc.input('Scale')[0]
        bias_name = op_desc.input('Bias')[0]
        y_name = op_desc.output('Y')[0]
        var_name = op_desc.output('Variance')[0]
        mean_name = op_desc.output('Mean')[0]
        begin_norm_axis = op_desc.attr('begin_norm_axis')

        x_spec = get_dist_tensor_spec(dist_op, x_name)
        scale_spec = get_dist_tensor_spec(dist_op, scale_name)
        bias_spec = get_dist_tensor_spec(dist_op, bias_name)
        y_spec = get_dist_tensor_spec(dist_op, y_name, False)
        var_spec = get_dist_tensor_spec(dist_op, var_name, False)
        mean_spec = get_dist_tensor_spec(dist_op, mean_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("layer_norm")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(
            x_spec, scale_spec, bias_spec, 1.0, begin_norm_axis
        )
        bw_results = rule.infer_backward(
            x_spec,
            scale_spec,
            bias_spec,
            y_spec,
            var_spec,
            mean_spec,
            1.0,
            begin_norm_axis,
        )

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op,
            [x_name, scale_name, bias_name],
            [y_name, var_name, mean_name],
            fw_results,
            bw_results,
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        begin_norm_axis = op_desc.attr('begin_norm_axis')

        # sharded on begin_norm_axis
        x_name = op_desc.input('X')[0]
        x_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(x_name)
        )
        if (begin_norm_axis > 0) and is_dim_shard(
            x_dims_mapping[begin_norm_axis]
        ):
            # TODO (ljz) support sharding on `begin_norm_axis`
            _logger.info(
                "sharding on `begin_norm_axis` is not supported yet, we resharded it as replicated"
            )
            x_dims_mapping[begin_norm_axis] = -1
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)

            param_names = [op_desc.input('Scale')[0], op_desc.input('Bias')[0]]
            for p_name in param_names:
                p_dims_mapping = copy.deepcopy(
                    op_dist_attr.get_input_dims_mapping(p_name)
                )
                p_dims_mapping[begin_norm_axis] = -1
                op_dist_attr.set_input_dims_mapping(p_name, p_dims_mapping)

            y_name = op_desc.output('Y')[0]
            y_dims_mapping = copy.deepcopy(
                op_dist_attr.get_output_dims_mapping(y_name)
            )
            y_dims_mapping[begin_norm_axis] = -1
            op_dist_attr.set_input_dims_mapping(y_name, y_dims_mapping)

        # default impl
        default_impl = get_default_distributed_operator_impl()
        op_dist_attr.impl_type = default_impl.type
        op_dist_attr.impl_idx = default_impl.idx

        return False


register_distributed_operator_impl_container(DistributedLayerNorm("layer_norm"))
