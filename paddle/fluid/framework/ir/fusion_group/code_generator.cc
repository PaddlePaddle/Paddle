/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include <set>
#include <sstream>
#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

CodeGenerator::CodeGenerator() {
  // Only support elementwise operations now.
  code_templates_.resize(1);

  CodeTemplate elementwise_t(elementwise_cuda_template);
  code_templates_[0] = elementwise_t;
}

// In order to get the right result of expression, we need to calculate and
// store the expression as suffix Expressions using vector.
std::string CodeGenerator::GenerateCode(
    std::string func_name, std::vector<OperationExpression> expressions) {
  // Check whether all expressions are elementwise operations.
  TemplateVariable template_var;
  template_var.Add("func_name", func_name);
  template_var.Add("parameters", EmitParameters(expressions, "float"));
  template_var.Add("compute_body", EmitComputeBody(expressions));
  return predefined_cuda_functions + code_templates_[0].Format(template_var);
}

// we get the parameter list code for the expression information
std::string CodeGenerator::EmitParameters(
    std::vector<OperationExpression> expressions, std::string dtype) {
  std::stringstream ret;

  std::set<int> input_ids;
  std::set<int> output_ids;

  for (size_t i = 0; i < expressions.size(); i++) {
    std::vector<int> tmp_input = expressions[i].GetInputIds();
    for (size_t j = 0; j < tmp_input.size(); j++) {
      int id = tmp_input[j];
      input_ids.insert(id);
    }
    int tmp_output = expressions[i].GetOutputIds()[0];
    output_ids.insert(tmp_output);
  }

  std::set<int>::iterator it = input_ids.begin();
  while (it != input_ids.end()) {
    int var_index = *it;
    if (output_ids.find(var_index) != output_ids.end()) {
      input_ids.erase(it++);
    } else {
      it++;
    }
  }

  ret << "int N, ";
  for (it = input_ids.begin(); it != input_ids.end(); it++) {
    int var_index = *it;
    ret << dtype << R"(* var)" << var_index;
    ret << ", ";
  }

  size_t count_index = 0;
  for (it = output_ids.begin(); it != output_ids.end(); it++) {
    int var_index = *it;
    ret << dtype << R"(* var)" << var_index;
    if (count_index != output_ids.size() - 1) {
      ret << ", ";
    }
    count_index++;
  }

  return ret.str();
}

std::string CodeGenerator::EmitComputeBody(
    std::vector<OperationExpression> expressions) {
  // get the right experssion code using suffix expression
  std::stringstream ret;
  for (size_t i = 0; i < expressions.size(); i++) {
    ret << expressions[i].GetExpression();
  }
  return ret.str();
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
