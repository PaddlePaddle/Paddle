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
Eigen::DefaultDevice* ExecutionContext::GetEigenDevice<
    platform::CPUPlace, Eigen::DefaultDevice>() const {
  return device_context_.get_eigen_device<Eigen::DefaultDevice>();
}

#ifndef PADDLE_ONLY_CPU
template <>
Eigen::GpuDevice*
ExecutionContext::GetEigenDevice<platform::GPUPlace, Eigen::GpuDevice>() const {
  return device_context_.get_eigen_device<Eigen::GpuDevice>();
}
#endif

const std::string& OperatorBase::Input(const std::string& name) const {
  PADDLE_ENFORCE(in_out_idxs_ != nullptr,
                 "Input Output Indices could not be nullptr");
  auto it = in_out_idxs_->find(name);
  PADDLE_ENFORCE(it != in_out_idxs_->end(), "no key [%s] in in_out_idxs_",
                 name);
  if (attrs_.count("input_format") == 0) {
    return inputs_.at((size_t)it->second);
  } else {
    const auto& input_format = GetAttr<std::vector<int>>("input_format");
    int idx = input_format[it->second];
    return inputs_.at((size_t)idx);
  }
}

std::vector<std::string> OperatorBase::Inputs(const std::string& name) const {
  PADDLE_ENFORCE(in_out_idxs_ != nullptr, "IO Idx could not be nullptr");
  auto input_format = GetAttr<std::vector<int>>("input_format");
  auto offset = in_out_idxs_->at(name);
  PADDLE_ENFORCE(input_format.at((size_t)offset + 1) <= (int)inputs_.size(),
                 "Input Out Of Range");

  return std::vector<std::string>{
      inputs_.begin() + input_format.at(offset),
      inputs_.begin() + input_format.at(offset + 1)};
}

const std::string& OperatorBase::Output(const std::string& name) const {
  PADDLE_ENFORCE(in_out_idxs_ != nullptr, "InOut Indice could not be nullptr");
  auto it = in_out_idxs_->find(name);
  PADDLE_ENFORCE(it != in_out_idxs_->end(), "no key [%s] in in_out_idxs_",
                 name);
  if (attrs_.count("output_format") == 0) {
    return outputs_.at((size_t)it->second);
  } else {
    const auto& output_format = GetAttr<std::vector<int>>("output_format");
    int idx = output_format[it->second];
    return outputs_.at((size_t)idx);
  }
}

std::vector<std::string> OperatorBase::Outputs(const std::string& name) const {
  PADDLE_ENFORCE(in_out_idxs_ != nullptr, "InOut Indice could not be nullptr");
  auto output_format = GetAttr<std::vector<int>>("output_format");
  auto offset = in_out_idxs_->at(name);
  PADDLE_ENFORCE(output_format.at((size_t)offset + 1) <= (int)outputs_.size(),
                 "Output Out of Range");
  return std::vector<std::string>{
      outputs_.begin() + output_format.at(offset),
      outputs_.begin() + output_format.at(offset + 1)};
}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:(";
  for (size_t i = 0; i < inputs_.size(); ++i) {
    ss << inputs_[i];
    if (i != inputs_.size() - 1) {
      ss << ", ";
    }
  }
  ss << "), outputs:(";
  for (size_t i = 0; i < outputs_.size(); ++i) {
    ss << outputs_[i];
    if (i != outputs_.size() - 1) {
      ss << ", ";
    }
  }
  ss << ").";
  return ss.str();
}

void OperatorBase::Rename(const std::string& old_name,
                          const std::string& new_name) {
  std::replace(inputs_.begin(), inputs_.end(), old_name, new_name);
  std::replace(outputs_.begin(), outputs_.end(), old_name, new_name);
}

}  // namespace framework
}  // namespace paddle
