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

#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

void PlainNet::CompleteAddOp(bool calc) {
  add_op_done_ = true;
  if (!calc) return;

  std::unordered_set<std::string> input_set;
  std::unordered_set<std::string> output_set;
  std::unordered_set<std::string> temp_output;
  for (auto& op : ops_) {
    for (auto& ipt : op->inputs_) {
      if (!Contains(output_set, ipt)) {  // Not other op's output
        input_set.insert(ipt);
      } else {
        temp_output.insert(ipt);
      }
    }

    for (auto& opt : op->outputs_) {
      output_set.insert(opt);
    }
  }
  inputs_.reserve(input_set.size());
  std::copy(input_set.begin(), input_set.end(), std::back_inserter(inputs_));

  outputs_.reserve(output_set.size());
  std::vector<int> tmp_index;
  tmp_index.reserve(temp_output.size());
  int idx = 0;
  for (auto& opt : output_set) {
    if (Contains(temp_output, opt)) {
      tmp_index.push_back(idx);
    }
    outputs_.push_back(opt);
    ++idx;
  }

  attrs_["temporary_index"] = tmp_index;
}

std::string PlainNet::DebugString() const {
  std::ostringstream os;
  os << this->type_ << ":" << std::endl;
  for (auto& op : ops_) {
    os << "\t" << op->DebugString() << std::endl;
  }
  return os.str();
}

}  // namespace framework
}  // namespace paddle
