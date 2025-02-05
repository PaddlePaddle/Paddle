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

import argparse
import logging
import os

import yaml
from op_gen import (
    OpCompatParser,
    OpInfoParser,
    cache_grad_op_shape_black_list,
    to_pascal_case,
)

CPP_FILE_TEMPLATE = """
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/shape_analysis_utils.h"

namespace paddle {{
namespace dialect {{

{body}

}}  // namespace dialect
}}  // namespace paddle
"""

GET_ORIGINAL_ATTR_MAP_FUNC_CODE_TEMPLATE = """
std::unordered_map<std::string, std::set<std::string>> GetAllOpOriginalAttributes() {{
  return {{{original_attr_map_items}
  }};
}}
"""

ORIGINAL_ATTR_MAP_ITEM_CODE_TEMPLATE = """
    {{"{op_name}", {{{original_attr_name_list}}}}},"""

CACHE_GRAD_OP_SYMBOL_SHAPE_FUNC_CODE_TEMPLATE = """
void {op_name}Op::CacheGradOpSymbolicShape(pir::InferSymbolicShapeContext* infer_context) {{
{create_grad_op_shape_info_code}
  pir::InferSymbolicShapeCacheKey op_shape_info(
      "{grad_op_name}",
      {{{input_shape_list}}},
      pir::GetOrderedOriginalAttributes("{grad_op_name}", this->operation()->attributes()));
{create_grad_op_output_shape_code}
  std::vector<symbol::ShapeOrDataDimExprs> output_shape_or_data{{{output_shape_list}}};

  infer_context->SetOpInferSymbolicShapeCache(op_shape_info,
                                              output_shape_or_data);
}}
"""

UNIMPLEMENTED_CODE_TEMPLATE = """
void {op_name}Op::CacheGradOpSymbolicShape(pir::InferSymbolicShapeContext* infer_context) {{
  PADDLE_THROW(common::errors::Unimplemented("{op_name} CacheGradOpSymbolicShape is not implemented!"));
}}
"""

SHAPE_VAR_NAME_SUFFIX = "_shape"

GET_INPUT_SHAPE_CODE_TEMPLATE = """
  const auto& {input_name}{name_suffix} = GetInputShape(infer_context, this->operation(), {index});"""

GET_OUTPUT_SHAPE_CODE_TEMPLATE = """
  const auto& {output_name}{name_suffix} = GetOutputShape(infer_context, this->operation(), {index});"""

GET_OUT_GRAD_SHAPE_CODE_TEMPLATE = """
  const auto& {output_grad_name}{name_suffix} = GetGradVarShapeFromOutput(infer_context, this->operation(), {index});"""

GET_INPUT_GRAD_SHAPE_CODE_TEMPLATE = """
  const auto& {input_grad_name}{name_suffix} = GetGradVarShapeFromInput(infer_context, this->operation(), {index});"""


class CacheGradOpSymbolShapeCodeGen:
    def __init__(
        self, op_yaml_files, op_compat_yaml_file, dialect_name="pd_op"
    ):
        self.op_info_maps = self.parse_yaml(
            op_yaml_files,
            op_compat_yaml_file,
        )
        self.dialect_name = dialect_name

    def parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        op_compat_parser = OpCompatParser(op_compat_yaml_file)

        op_info_maps = {}
        for yaml_file in op_yaml_files:
            with open(yaml_file, "r") as f:
                ops = yaml.safe_load(f)
                for op in ops:
                    op_compat_item = op_compat_parser.get_compat(op['name'])
                    if (
                        op_compat_item is not None
                        and op_compat_item['op'] == "pow"
                        and 'scalar' in op_compat_item
                    ):
                        op_compat_item = op_compat_item.pop('scalar')
                    op_info_maps[op["name"]] = OpInfoParser(
                        op, op_compat_item, yaml_file
                    )
        return op_info_maps

    def gen_cpp_file_code(self, cpp_file_path):
        cache_func_code = ""
        original_attr_map_items_code = ""
        get_op_name_with_dialect = (
            lambda op_name: self.dialect_name + "." + op_name
        )
        for op_info_item in self.op_info_maps.values():
            if op_info_item.backward_name is None:
                continue
            if op_info_item.backward_name not in self.op_info_maps:
                continue
            grad_op_item = self.op_info_maps[op_info_item.backward_name]

            if (
                op_info_item.kernel_map is None
                or grad_op_item.kernel_map is None
            ):
                continue

            original_attr_name = set(op_info_item.attribute_name_list) & set(
                grad_op_item.attribute_name_list
            )
            convert_to_cpp_str = lambda str: '"' + str + '"'
            original_attr_name_list_str = f"{', '.join(convert_to_cpp_str(item) for item in original_attr_name)}"
            original_attr_map_items_code += (
                ORIGINAL_ATTR_MAP_ITEM_CODE_TEMPLATE.format(
                    op_name=get_op_name_with_dialect(
                        op_info_item.backward_name
                    ),
                    original_attr_name_list=original_attr_name_list_str,
                )
            )
            for op_phi_name in op_info_item.op_phi_name:
                if op_phi_name in cache_grad_op_shape_black_list:
                    continue
                original_attr_map_items_code += (
                    ORIGINAL_ATTR_MAP_ITEM_CODE_TEMPLATE.format(
                        op_name=get_op_name_with_dialect(op_phi_name),
                        original_attr_name_list=original_attr_name_list_str,
                    )
                )
                create_grad_op_shape_info_code = ""
                for input_name in grad_op_item.input_name_list:
                    if input_name in grad_op_item.forward_input_name_list:
                        # forward input
                        index = grad_op_item.forward_input_name_list.index(
                            input_name
                        )
                        create_grad_op_shape_info_code += (
                            GET_INPUT_SHAPE_CODE_TEMPLATE.format(
                                input_name=input_name,
                                name_suffix=SHAPE_VAR_NAME_SUFFIX,
                                index=index,
                            )
                        )
                    elif input_name in grad_op_item.forward_output_name_list:
                        # forward output
                        index = grad_op_item.forward_output_name_list.index(
                            input_name
                        )
                        create_grad_op_shape_info_code += (
                            GET_OUTPUT_SHAPE_CODE_TEMPLATE.format(
                                output_name=input_name,
                                name_suffix=SHAPE_VAR_NAME_SUFFIX,
                                index=index,
                            )
                        )
                    elif input_name.endswith("_grad"):
                        # output grad
                        origin_out_name = input_name[:-5]
                        index = grad_op_item.forward_output_name_list.index(
                            origin_out_name
                        )
                        create_grad_op_shape_info_code += (
                            GET_OUT_GRAD_SHAPE_CODE_TEMPLATE.format(
                                output_grad_name=input_name,
                                name_suffix=SHAPE_VAR_NAME_SUFFIX,
                                index=index,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Not found input name {input_name} for backward op {op_info_item.backward_name}."
                        )
                # mutable attribute
                for (
                    mutable_attribute_name
                ) in grad_op_item.mutable_attribute_name_list:
                    assert (
                        mutable_attribute_name
                        in op_info_item.mutable_attribute_name_list
                    ), f"{mutable_attribute_name} is not found in {op_info_item.backward_name}'s mutable_attribute name list."
                    index = len(
                        op_info_item.input_name_list
                    ) + op_info_item.mutable_attribute_name_list.index(
                        mutable_attribute_name
                    )
                    create_grad_op_shape_info_code += (
                        GET_INPUT_SHAPE_CODE_TEMPLATE.format(
                            input_name=mutable_attribute_name,
                            name_suffix=SHAPE_VAR_NAME_SUFFIX,
                            index=index,
                        )
                    )

                create_grad_op_output_shape_code = ""
                for output_name in grad_op_item.output_name_list:
                    if not output_name.endswith("_grad"):
                        create_grad_op_output_shape_code = ""
                        break
                    origin_input_name = output_name[:-5]
                    if (
                        origin_input_name
                        not in grad_op_item.forward_input_name_list
                    ):
                        continue
                    index = grad_op_item.forward_input_name_list.index(
                        origin_input_name
                    )
                    create_grad_op_output_shape_code += (
                        GET_INPUT_GRAD_SHAPE_CODE_TEMPLATE.format(
                            input_grad_name=output_name,
                            name_suffix=SHAPE_VAR_NAME_SUFFIX,
                            index=index,
                        )
                    )
                if len(create_grad_op_output_shape_code) == 0:
                    logging.warning(
                        f"{op_phi_name}'s grad op has some exception, please check it in yaml file."
                    )
                    cache_func_code += UNIMPLEMENTED_CODE_TEMPLATE.format(
                        op_name=to_pascal_case(op_phi_name),
                    )
                    continue

                cache_func_code += CACHE_GRAD_OP_SYMBOL_SHAPE_FUNC_CODE_TEMPLATE.format(
                    op_name=to_pascal_case(op_phi_name),
                    create_grad_op_shape_info_code=create_grad_op_shape_info_code,
                    grad_op_name=get_op_name_with_dialect(
                        grad_op_item.op_phi_name[0]
                    ),
                    input_shape_list=", ".join(
                        [
                            input_name + SHAPE_VAR_NAME_SUFFIX
                            for input_name in (
                                grad_op_item.input_name_list
                                + grad_op_item.mutable_attribute_name_list
                            )
                        ]
                    ),
                    create_grad_op_output_shape_code=create_grad_op_output_shape_code,
                    output_shape_list=", ".join(
                        [
                            output_name + SHAPE_VAR_NAME_SUFFIX
                            for output_name in grad_op_item.output_name_list
                        ]
                    ),
                )

                if len(op_info_item.kernel_map['func']) == 1:
                    continue
                for kernel_func_name in op_info_item.kernel_map['func']:
                    is_inplace_version = op_phi_name.endswith('_')
                    op_origin_name = (
                        op_phi_name[:-1] if is_inplace_version else op_phi_name
                    )
                    if kernel_func_name == op_origin_name:
                        continue
                    inplace_suffix = '_' if is_inplace_version else ''
                    cache_func_code += UNIMPLEMENTED_CODE_TEMPLATE.format(
                        op_name=to_pascal_case(kernel_func_name)
                        + inplace_suffix
                    )
        get_original_attr_map_func_code = (
            GET_ORIGINAL_ATTR_MAP_FUNC_CODE_TEMPLATE.format(
                original_attr_map_items=original_attr_map_items_code
            )
        )

        directory_path = os.path.dirname(cpp_file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        with open(cpp_file_path, 'w') as f:
            f.write(
                CPP_FILE_TEMPLATE.format(
                    body=get_original_attr_map_func_code + cache_func_code,
                )
            )


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Cache GradOp Symbol Shape Interface Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--cache_grad_op_symbol_shape_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    cache_grad_op_symbol_shape_file = args.cache_grad_op_symbol_shape_file

    code_gen = CacheGradOpSymbolShapeCodeGen(
        op_yaml_files,
        op_compat_yaml_file,
    )
    code_gen.gen_cpp_file_code(cache_grad_op_symbol_shape_file)
