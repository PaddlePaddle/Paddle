/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
#include "paddle/fluid/framework/ir/codegen_helper.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
namespace paddle {
namespace framework {
namespace ir {

OperationExpression::OperationExpression(std::vector<int> input_ids,
                                         int output_id,
                                         std::string search_operation) {
  input_ids_ = input_ids;
  output_id_ = output_id;
  search_operation_ = search_operation;
}

// we Traverse the graph and get the group , all input id and output id is
// unique for the node which belong the group
std::string OperationExpression::GetExpression() {
  std::stringstream ret;
  if (operator_cuda_table.find(search_operation_) ==
      operator_cuda_table.end()) {
    std::cerr << "Not supportted operation, " << search_operation_ << std::endl;
  } else {
    auto rhs = operator_cuda_table[search_operation_];
    std::string replaced_str = "$";
    int count = 0;
    auto pos = rhs.find(replaced_str);
    while (pos != -1) {
      auto index = input_ids_[count];
      rhs.replace(pos, replaced_str.length(),
                  std::to_string(index) + R"([offset])");
      pos = rhs.find(replaced_str);
      count++;
    }
    auto lhs = std::string(indentation) + "var" + std::to_string(output_id_) +
               R"([offset])";
    auto equal_split = R"( = )";
    auto semicolon = R"(;)";
    ret << lhs << equal_split << rhs << semicolon << std::endl;
  }

  return ret.str();
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
