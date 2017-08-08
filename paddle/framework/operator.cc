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

#include <algorithm>

#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

template <>
Eigen::DefaultDevice& ExecutionContext::GetEigenDevice<
    platform::CPUPlace, Eigen::DefaultDevice>() const {
  return *device_context_->get_eigen_device<Eigen::DefaultDevice>();
}

#ifndef PADDLE_ONLY_CPU
template <>
Eigen::GpuDevice&
ExecutionContext::GetEigenDevice<platform::GPUPlace, Eigen::GpuDevice>() const {
  return *device_context_->get_eigen_device<Eigen::GpuDevice>();
}
#endif

const std::string& OperatorBase::Input(const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Op %s does not have output %s", type_,
                 name);
  PADDLE_ENFORCE_EQ(it->second.size(), 1UL,
                    "Op %s input %s should contain only one variable", type_,
                    name);
  return it->second[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  return inputs_.at(name);
}

const std::string& OperatorBase::Output(const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(), "Op %s does not have output %s", type_,
                 name);
  PADDLE_ENFORCE_EQ(it->second.size(), 1UL,
                    "Op %s input %s should contain only one variable", type_,
                    name);
  return it->second[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  return outputs_.at(name);
}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto& input : inputs_) {
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }
  ss << "}, outputs:{";
  for (auto& output : outputs_) {
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }
  ss << "}.";
  return ss.str();
}

void OperatorBase::Rename(const std::string& old_name,
                          const std::string& new_name) {
  for (auto& input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }
  for (auto& output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }
}

}  // namespace framework
}  // namespace paddle
