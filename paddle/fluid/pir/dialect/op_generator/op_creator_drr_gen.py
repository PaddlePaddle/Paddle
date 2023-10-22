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
# limitations under the License.

import argparse

import yaml
from op_gen import OpCompatParser, OpInfoParser, to_pascal_case

CPP_FILE_TEMPLATE = """
#include "paddle/fluid/pir/drr/ir_operation_factory.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"

namespace pir {{
namespace drr {{

void OperationFactory::RegisterGeneratedOpCreator() {{
{body}
}}

}}  // namespace drr
}}  // namespace pir

"""

NORMAL_FUNCTION_TEMPLATE = """
  RegisterOperationCreator(
      "{op_name}",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {{
        return rewriter.Build<paddle::dialect::{op_class_name}>(
         {params_code});
      }});
"""

MUTABLE_ATTR_FUNCTION_TEMPLATE = """
  RegisterOperationCreator(
      "{op_name}",
      [](const std::vector<Value>& inputs,
         const pir::AttributeMap& attrs,
         pir::PatternRewriter& rewriter) {{
        // mutable_attr is tensor
        if (inputs.size() > {inputs_num}) {{
          return rewriter.Build<paddle::dialect::{op_class_name}>(
          {params_code_with_mutable_attr});
        }} else {{
          return rewriter.Build<paddle::dialect::{op_class_name}>(
          {params_code_no_mutable_attr});
        }}
      }});
"""


class OpCreatorCodeGen:
    def __init__(self, op_yaml_files, op_compat_yaml_file, dialect_name):
        self.op_info_items = self.parse_yaml(op_yaml_files, op_compat_yaml_file)
        self.dialect_name = dialect_name

    def parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        op_compat_parser = OpCompatParser(op_compat_yaml_file)

        op_yaml_items = []
        for yaml_file in op_yaml_files:
            with open(yaml_file, "r") as f:
                ops = yaml.safe_load(f)
                op_yaml_items = op_yaml_items + ops
        op_info_items = []
        for op in op_yaml_items:
            op_compat_item = op_compat_parser.get_compat(op['name'])
            if (
                op_compat_item is not None
                and op_compat_item['op'] == "pow"
                and 'scalar' in op_compat_item
            ):
                op_compat_item = op_compat_item.pop('scalar')
            op_info_items.append(OpInfoParser(op, op_compat_item))
        return op_info_items

    def gen_cpp_file_code(self, cpp_file_path):
        body_code = ""
        for op_info_item in self.op_info_items:
            if op_info_item.infer_meta_map is None:
                continue
            for phi_op_name in op_info_item.op_phi_name:
                ir_op_name = self.dialect_name + "." + phi_op_name
                params_no_mutable_attr = []
                for i in range(len(op_info_item.input_name_list)):
                    params_no_mutable_attr.append(
                        f"inputs[{i}].dyn_cast<pir::OpResult>()"
                    )
                if len(op_info_item.attribute_name_list) > 0:
                    params_no_mutable_attr.append("attrs")

                if len(op_info_item.mutable_attribute_name_list) == 0:
                    body_code += NORMAL_FUNCTION_TEMPLATE.format(
                        op_name=ir_op_name,
                        op_class_name=(to_pascal_case(phi_op_name) + "Op"),
                        params_code=", ".join(params_no_mutable_attr),
                    )
                else:
                    params_with_mutable_attr = []
                    for i in range(
                        len(op_info_item.input_name_list)
                        + len(op_info_item.mutable_attribute_name_list)
                    ):
                        params_with_mutable_attr.append(
                            f"inputs[{i}].dyn_cast<pir::OpResult>()"
                        )
                    if len(op_info_item.attribute_name_list) > len(
                        op_info_item.mutable_attribute_name_list
                    ):
                        # TODO(zyfncg): Currently Op::Build Interface doesn't support this case.
                        continue
                        # params_with_mutable_attr.append("attrs")

                    body_code += MUTABLE_ATTR_FUNCTION_TEMPLATE.format(
                        op_name=ir_op_name,
                        inputs_num=len(op_info_item.input_name_list),
                        op_class_name=(to_pascal_case(phi_op_name) + "Op"),
                        params_code_with_mutable_attr=",".join(
                            params_with_mutable_attr
                        ),
                        params_code_no_mutable_attr=", ".join(
                            params_no_mutable_attr
                        ),
                    )

        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body_code))


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Op Creator Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--dialect_name', type=str)
    parser.add_argument('--op_creator_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    op_creator_file = args.op_creator_file
    dialect_name = args.dialect_name

    code_gen = OpCreatorCodeGen(
        op_yaml_files, op_compat_yaml_file, dialect_name
    )
    code_gen.gen_cpp_file_code(op_creator_file)
