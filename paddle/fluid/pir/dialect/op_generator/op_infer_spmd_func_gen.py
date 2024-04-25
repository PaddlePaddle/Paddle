# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

OP_INFER_SPMD_TEMPLATE = """
  static phi::distributed::SpmdInfo InferSpmd({infer_spmd_args}) {{
    return phi::distributed::{func}({args});
  }}
"""


def gen_op_infer_spmd_func(args, op_info, op_info_items):
    if not args.with_distributed or op_info.spmd_rule_func is None:
        return [], None, None
    input_types_map = {
        'paddle::dialect::DenseTensorType': 'const phi::distributed::DistMetaTensor&',
        'pir::VectorType<paddle::dialect::DenseTensorType>': 'const std::vector<phi::distributed::DistMetaTensor>&',
    }
    input_name_list = op_info.input_name_list
    input_type_list = op_info.input_type_list
    input_name_type_dict = {}
    for attr_idx in range(len(input_name_list)):
        input_name_type_dict[input_name_list[attr_idx]] = input_types_map[
            input_type_list[attr_idx]
        ]

    attr_name_list = op_info.attribute_name_list
    attr_type_list = op_info.attribute_gen_arg_type_list

    attr_name_type_dict = {}
    for attr_idx in range(len(attr_type_list)):
        attr_name_type_dict[attr_name_list[attr_idx]] = attr_type_list[attr_idx]
        scalar_list = [
            "Scalar(int64_t)",
            "Scalar(int)",
            "Scalar(float)",
            "Scalar(double)",
        ]
        if op_info.op_yaml_item['attrs'][attr_idx]['typename'] in scalar_list:
            attr_name_type_dict[attr_name_list[attr_idx]] = "const phi::Scalar&"

    spmd_params = input_name_list + attr_name_list
    if op_info.kernel_map is not None:
        spmd_params = op_info.kernel_map['param']
    args_list_with_type = []
    args_list = []
    for param in spmd_params:
        # is input
        if param in op_info.input_name_list:
            args_list_with_type.append(
                input_name_type_dict[param] + " " + param
            )
            args_list.append(param)
        # is attribute
        else:
            param_type = attr_name_type_dict[param]
            if param_type == "phi::IntArray":
                param_type = "const std::vector<int64_t>&"
            args_list_with_type.append(param_type + " " + param)
            args_list.append(param)

    spmd_rule_func = op_info.spmd_rule_func
    if spmd_rule_func is None:
        spmd_rule_func = "VariadicReplicatedInferSpmdDynamic"
    declare_str = OP_INFER_SPMD_TEMPLATE.format(
        infer_spmd_args=', '.join(args_list_with_type),
        func=spmd_rule_func,
        args=', '.join(args_list),
    )
    return [], declare_str, None
