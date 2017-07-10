/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

void OperatorBase::InferShape(std::shared_ptr<Scope> scope) const {}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "=================\n";
  ss << "type = " << desc_.type() << "\n";
  ss << "inputs = [";
  for (auto& ipt : inputs_) {
    ss << ipt << ", ";
  }
  ss << "]\n";
  ss << "outputs = [";
  for (auto& opt : outputs_) {
    ss << opt << ", ";
  }
  ss << "]\n";
  ss << "attr_keys = [";
  for (auto& attr : attrs_) {
    ss << attr.first << ", ";
  }
  ss << "]\n";
  return ss.str();
}

const Variable* OpContext::Input(int index) const {
  return scope->GetVariable(op->inputs_[index]);
}

Variable* OpContext::Output(int index) const {
  return scope->GetVariable(op->outputs_[index]);
}

}  // namespace framework
}  // namespace paddle