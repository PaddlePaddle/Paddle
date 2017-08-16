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

const char NetOp::kAll[] = "all";

void NetOp::CompleteAddOp(bool calc) {
  add_op_done_ = true;
  if (!calc) return;
  std::set<std::string> input_set;
  std::set<std::string> output_set;
  for (auto& op : ops_) {
    for (auto& ipt : op->Inputs()) {
      for (auto& var_name : ipt.second) {
        if (!Contains(output_set, var_name)) {  // Not other op's output
          input_set.insert(var_name);
        } else {
          intermediate_outputs_.insert(var_name);
        }
      }
    }

    for (auto& opt : op->Outputs()) {
      for (auto& var_name : opt.second) {
        output_set.insert(var_name);
      }
    }
  }
  auto& inputs = inputs_[kAll];
  inputs.reserve(input_set.size());
  std::copy(input_set.begin(), input_set.end(), std::back_inserter(inputs));
  auto& outputs = outputs_[kAll];
  outputs.reserve(output_set.size());
  std::copy(output_set.begin(), output_set.end(), std::back_inserter(outputs));
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

std::vector<std::string> NetOp::OutputVars(bool has_intermediate) const {
  if (has_intermediate) {
    return this->outputs_.at(kAll);
  }
  auto& all = this->outputs_.at(kAll);
  std::vector<std::string> ret_val;
  for (auto& each : all) {
    if (!Contains(intermediate_outputs_, each)) {
      ret_val.push_back(each);
    }
  }
  return ret_val;
}

NetOp::NetOp(const std::string& type,
             const framework::OperatorBase::VarNameMap& inputs,
             const framework::OperatorBase::VarNameMap& outputs,
             const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

}  // namespace operators
}  // namespace paddle
