/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/operators/net_op.h"
#include <set>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

void NetOp::CompleteAddOp(bool calc) {
  add_op_done_ = true;
  if (!calc) return;
  std::set<std::string> input_set;
  std::set<std::string> output_set;
  std::set<std::string> temp_output;
  for (auto& op : ops_) {
    for (auto& ipt : op->inputs_) {
      for (auto& var_name : ipt.second) {
        if (!Contains(output_set, var_name)) {  // Not other op's output
          input_set.insert(var_name);
        } else {
          temp_output.insert(var_name);
        }
      }
    }

    for (auto& opt : op->outputs_) {
      for (auto& var_name : opt.second) {
        output_set.insert(var_name);
      }
    }
  }
  auto& inputs = inputs_["all"];
  inputs.reserve(input_set.size());
  std::copy(input_set.begin(), input_set.end(), std::back_inserter(inputs));
  auto& outputs = outputs_["all"];
  outputs.reserve(output_set.size());
  std::copy(output_set.begin(), output_set.end(), std::back_inserter(outputs));

  //! TODO figure out how to generate temporary_index in Network.
  std::vector<int> tmp_index;
  tmp_index.reserve(temp_output.size());
  int output_len = static_cast<int>(outputs.size());
  for (int i = 0; i < output_len; ++i) {
    if (Contains(temp_output, outputs[i])) {
      tmp_index.push_back(i);
    }
  }

  attrs_["temporary_index"] = tmp_index;
}

std::string NetOp::DebugString() const {
  std::ostringstream os;
  os << OperatorBase::DebugString() << std::endl;
  for (auto& op : ops_) {
    std::istringstream is(op->DebugString());
    for (std::string line; std::getline(is, line);) {
      os << "    " << line << std::endl;
    }
  }
  return os.str();
}

bool NetOp::IsNetOp() const { return true; }

}  // namespace operators
}  // namespace paddle
