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

import logging

from paddle.base.log_helper import get_logger

from ..completion import get_phi_spmd_rule
from ..utils import get_dist_tensor_spec
from .common import (
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class DistributedFusedRope(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc
        q = op_desc.input('q')[0]
        k = op_desc.input('k')[0]
        v = op_desc.input('v')[0]
        sin = op_desc.input('sin')[0]
        cos = op_desc.input('cos')[0]
        position_ids = op_desc.input('position_ids')[0]
        out_q = op_desc.output('out_q')[0]
        out_k = op_desc.output('out_k')[0]
        out_v = op_desc.output('out_v')[0]

        q_spec = get_dist_tensor_spec(dist_op, q)
        k_spec = get_dist_tensor_spec(dist_op, k)
        v_spec = get_dist_tensor_spec(dist_op, v)
        sin_spec = get_dist_tensor_spec(dist_op, sin)
        cos_spec = get_dist_tensor_spec(dist_op, cos)
        position_ids_spec = get_dist_tensor_spec(dist_op, position_ids)
        out_q_spec = get_dist_tensor_spec(dist_op, out_q, is_input=False)
        out_k_spec = get_dist_tensor_spec(dist_op, out_k, is_input=False)
        out_v_spec = get_dist_tensor_spec(dist_op, out_v, is_input=False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("fused_rotary_position_embedding")
        # tensor order following order in PHI defition
        fw_results = rule.infer_forward(
            q_spec, k_spec, v_spec, sin_spec, cos_spec, position_ids_spec
        )
        bw_results = rule.infer_backward(
            q_spec,
            k_spec,
            v_spec,
            sin_spec,
            cos_spec,
            position_ids_spec,
            out_q_spec,
            out_k_spec,
            out_v_spec,
        )

        # step3: update dist_attr
        # tensor order following order in PHI defition
        changed = update_op_dims_mapping(
            dist_op,
            [q, k, v, sin, cos, position_ids],
            [out_q, out_k, out_v],
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
    DistributedFusedRope("fused_rotary_position_embedding")
)
