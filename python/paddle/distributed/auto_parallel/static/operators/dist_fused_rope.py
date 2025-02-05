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
from ..dist_attribute import DistTensorSpec
from ..utils import get_dist_tensor_spec
from .common import (
    DistributedOperatorImplContainer,
    get_default_distributed_operator_impl,
    register_distributed_operator_impl_container,
    update_op_dims_mapping,
)


class DistributedFusedRope(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args), build fake spec for optional args

        op_desc = dist_op.serial_op.desc
        input_parameters = op_desc.input_names()
        output_parameters = op_desc.output_names()
        is_input_arg_exist = (
            lambda parameter: parameter in input_parameters
            and op_desc.input(parameter)
        )
        is_output_arg_exist = (
            lambda parameter: parameter in output_parameters
            and op_desc.output(parameter)
        )

        q = op_desc.input('q')[0]
        k = op_desc.input('k')[0] if is_input_arg_exist('k') else None
        v = op_desc.input('v')[0] if is_input_arg_exist('v') else None
        sin = op_desc.input('sin')[0] if is_input_arg_exist('sin') else None
        cos = op_desc.input('cos')[0] if is_input_arg_exist('cos') else None
        position_ids = (
            op_desc.input('position_ids')[0]
            if is_input_arg_exist('position_ids')
            else None
        )
        out_q = op_desc.output('out_q')[0]
        out_k = (
            op_desc.output('out_k')[0] if is_output_arg_exist('out_k') else None
        )
        out_v = (
            op_desc.output('out_v')[0] if is_output_arg_exist('out_v') else None
        )

        q_spec = get_dist_tensor_spec(dist_op, q)
        k_spec = (
            get_dist_tensor_spec(dist_op, k)
            if k is not None
            else DistTensorSpec()
        )
        v_spec = (
            get_dist_tensor_spec(dist_op, v)
            if v is not None
            else DistTensorSpec()
        )
        sin_spec = (
            get_dist_tensor_spec(dist_op, sin)
            if sin is not None
            else DistTensorSpec()
        )
        cos_spec = (
            get_dist_tensor_spec(dist_op, cos)
            if cos is not None
            else DistTensorSpec()
        )
        position_ids_spec = (
            get_dist_tensor_spec(dist_op, position_ids)
            if position_ids is not None
            else DistTensorSpec()
        )
        out_q_spec = get_dist_tensor_spec(dist_op, out_q, is_input=False)
        out_k_spec = (
            get_dist_tensor_spec(dist_op, out_k, is_input=False)
            if out_k is not None
            else DistTensorSpec()
        )
        out_v_spec = (
            get_dist_tensor_spec(dist_op, out_v, is_input=False)
            if out_v is not None
            else DistTensorSpec()
        )

        use_neox_rotary_style = op_desc.attr("use_neox_rotary_style")
        time_major = op_desc.attr("time_major")
        rotary_emb_base = op_desc.attr("rotary_emb_base")

        # step2: infer spmd
        rule = get_phi_spmd_rule("fused_rotary_position_embedding")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(
            q_spec,
            k_spec,
            v_spec,
            sin_spec,
            cos_spec,
            position_ids_spec,
            use_neox_rotary_style,
            time_major,
            rotary_emb_base,
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
            use_neox_rotary_style,
            time_major,
            rotary_emb_base,
        )

        # remove optional args in spmd results
        input_args = [q, k, v, sin, cos, position_ids]
        output_args = [out_q, out_k, out_v]
        fw_and_bw_results_without_optional_arg = []
        for results in [fw_results, bw_results]:
            input_results = results[0]
            output_results = results[1]
            input_results_without_optional_arg = []
            output_results_without_optional_arg = []
            for idx, input_arg in enumerate(input_args):
                if input_arg is not None:
                    input_results_without_optional_arg.append(
                        input_results[idx]
                    )
            for idx, output_arg in enumerate(output_args):
                if output_arg is not None:
                    output_results_without_optional_arg.append(
                        output_results[idx]
                    )
            fw_and_bw_results_without_optional_arg.append(
                [
                    input_results_without_optional_arg,
                    output_results_without_optional_arg,
                ]
            )

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op,
            input_arg_names=[
                input_arg for input_arg in input_args if input_arg is not None
            ],
            output_arg_names=[
                output_arg
                for output_arg in output_args
                if output_arg is not None
            ],
            fw_results=fw_and_bw_results_without_optional_arg[0],
            bw_results=fw_and_bw_results_without_optional_arg[1],
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
