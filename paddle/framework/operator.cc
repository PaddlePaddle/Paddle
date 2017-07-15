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

inline const std::string& OperatorBase::Input(const std::string& name) const {
  if (arg_idxs_.empty()) {
    int idx = 0;
    for (const auto& arg : desc_.inputs()) {
      arg_idxs_[arg] = idx;
    }
    for (const auto& arg : desc_.outputs()) {
      arg_idxs_[arg] = idx;
    }
  }
  if (attrs_.count("input_format") == 0) {
    return inputs_[arg_idxs_.at(name)];
  } else {
    const auto& input_format = GetAttr<std::vector<int>>("input_format");
    int idx = input_format[arg_idxs_.at(name)];
    return inputs_.at(idx);
  }
}

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

}  // namespace framework
}  // namespace paddle
