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

#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/operation.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

OperationExpression::OperationExpression(std::vector<int> input_ids,
                                         int output_id, std::string op) {
  input_ids_ = input_ids;
  output_id_ = output_id;
  op_ = op;
}

std::string OperationExpression::GetRHSTemplate() {
  std::stringstream ret;
  std::string rhs_end = ";";
  auto rhs = OperationMap::Instance().Get(op_).expr;
  for (size_t i = 0; i < input_ids_.size(); i++) {
    auto replaced_str = replaced_element_in_order[i];
    auto pos = rhs.find(replaced_str);
    auto index = input_ids_[i];
    rhs.replace(pos, replaced_str.length(), std::to_string(index) + R"([idx])");
  }
  ret << rhs << rhs_end;
  return ret.str();
}

std::string OperationExpression::GetLHSTemplate() {
  std::stringstream ret;
  ret << "var" << output_id_ << R"([idx] = )";
  return ret.str();
}

bool OperationExpression::IsSupport() {
  return OperationMap::Instance().Has(op_);
}

// we Traverse the graph and get the group , all input id and output id is
// unique for the node which belong the group
std::string OperationExpression::GetExpression() {
  std::stringstream ret;
  if (!IsSupport()) {
    ret << GetLHSTemplate() << GetRHSTemplate();
  }

  return ret.str();
}

std::string EmitUniqueName(std::vector<OperationExpression> expression) {
  std::stringstream ret;
  ret << "fused_kernel";
  for (size_t i = 0; i < expression.size(); i++) {
    ret << expression[i].GetOutputId();
  }
  return ret.str();
}

// we get the parameter list code for the expression information
std::string EmitDeclarationCode(std::vector<OperationExpression> expression,
                                std::string type) {
  std::stringstream ret;

  std::set<int> input_ids;
  std::set<int> output_ids;

  for (size_t i = 0; i < expression.size(); i++) {
    std::vector<int> tmp_input = expression[i].GetInputIds();
    for (size_t j = 0; j < tmp_input.size(); j++) {
      int id = tmp_input[j];
      input_ids.insert(id);
    }
    int tmp_output = expression[i].GetOutputId();
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
    ret << type << R"(* var)" << var_index;
    ret << ", ";
  }

  size_t count_index = 0;
  for (it = output_ids.begin(); it != output_ids.end(); it++) {
    int var_index = *it;
    ret << type << R"(* var)" << var_index;
    if (count_index != output_ids.size() - 1) {
      ret << ", ";
    }
    count_index++;
  }

  return ret.str();
}

std::string EmitComputeCode(std::vector<OperationExpression> expression) {
  // get the right experssion code using suffix expression
  std::stringstream ret;
  for (size_t i = 0; i < expression.size(); i++) {
    ret << expression[i].GetExpression();
  }
  return ret.str();
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
