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
                                         int output_id, std::string op) {
  input_ids_ = input_ids;
  output_id_ = output_id;
  op_ = op;
}

std::string OperationExpression::GetRHSTemplate() {
  std::stringstream ret;
  std::string rhs_end = ";";
  auto rhs = support_table[op_];
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

bool OperationExpression::SupportState() {
  return (support_table.find(op_) == support_table.end());
}
// we Traverse the graph and get the group , all input id and output id is
// unique for the node which belong the group
std::string OperationExpression::GetExpression() {
  std::stringstream ret;
  if (!SupportState()) {
    ret << GetLHSTemplate() << GetRHSTemplate();
  }

  return ret.str();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
