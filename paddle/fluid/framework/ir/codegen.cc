/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/ir/codegen.h"
#include <set>
#include <sstream>
#include "paddle/fluid/framework/ir/codegen_helper.h"
namespace paddle {
namespace framework {
namespace ir {

// we get the parameter list code for the expression information
std::string CodeGen::GetDeclarationCode(
    std::vector<OperationExpression> expression) {
  std::stringstream ret;
  ret << "fuse_kernel";
  ret << R"((int N )";
  std::set<int> input_ids;
  std::set<int> output_ids;
  std::vector<int> last_output_idis;

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

  for (it = input_ids.begin(); it != input_ids.end(); it++) {
    int var_index = *it;
    ret << R"(, const T* var)" << var_index;
  }

  for (it = output_ids.begin(); it != output_ids.end(); it++) {
    int var_index = *it;
    ret << R"(, T* var)" << var_index;
  }

  ret << R"())";

  return ret.str();
}

std::string CodeGen::GetOffsetCode() {
  std::stringstream ret;
  ret << indentation << "int offset = idx;" << std::endl;
  return ret.str();
}

std::string CodeGen::GetComputeCode(
    std::vector<OperationExpression> expression) {
  // get the right experssion code using suffix expression
  std::stringstream ret;
  for (size_t i = 0; i < expression.size(); i++) {
    ret << expression[i].GetExpression();
  }
  return ret.str();
}
// in order to get the right result of expression, we need to calculate, we
// store the expression as
// suffix Expressions using vector
std::string CodeGen::GetKernelCode(
    std::vector<OperationExpression> expression) {
  auto declaration_code = GetDeclarationCode(expression);
  auto offset_code = GetOffsetCode();
  auto compute_code = GetComputeCode(expression);
  auto cuda_kernel = const_kernel_start + declaration_code + const_kernel_mid +
                     offset_code + compute_code + const_kernel_end;
  return cuda_kernel;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
