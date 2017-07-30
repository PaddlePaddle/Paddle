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
  auto it = in_out_idxs_->find(name);
  PADDLE_ENFORCE(it != in_out_idxs_->end(), "no key [%s] in in_out_idxs_",
                 name);

  if (attrs_.count("input_format") == 0) {
    return inputs_[it->second];
  } else {
    const auto& input_format = GetAttr<std::vector<int>>("input_format");
    int idx = input_format[it->second];
    return inputs_.at(idx);
  }
}

std::vector<std::string> OperatorBase::Inputs(const std::string& name) const {
  auto input_format = GetAttr<std::vector<int>>("input_format");
  auto offset = in_out_idxs_->at(name);

  return std::vector<std::string>{
      inputs_.begin() + input_format.at(offset),
      inputs_.begin() + input_format.at(offset + 1)};
}

const std::string& OperatorBase::Output(const std::string& name) const {
  auto it = in_out_idxs_->find(name);
  PADDLE_ENFORCE(it != in_out_idxs_->end(), "no key [%s] in in_out_idxs_",
                 name);

  if (attrs_.count("output_format") == 0) {
    return outputs_[it->second];
  } else {
    const auto& output_format = GetAttr<std::vector<int>>("output_format");
    int idx = output_format[it->second];
    return outputs_.at(idx);
  }
}

std::vector<std::string> OperatorBase::Outputs(const std::string& name) const {
  auto output_format = GetAttr<std::vector<int>>("output_format");
  auto offset = in_out_idxs_->at(name);

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

template <>
const Variable* OperatorContext::Input<Variable>(size_t index) const {
  return scope_->GetVariable(op_.inputs_[index]);
}

template <>
Variable* OperatorContext::Output<Variable>(size_t index) const {
  return scope_->GetVariable(op_.outputs_[index]);
}

template <>
const Variable* OperatorContext::Input<Variable>(
    const std::string& name) const {
  return scope_->GetVariable(op_.Input(name));
}

template <>
Variable* OperatorContext::Output(const std::string& name) const {
  return scope_->GetVariable(op_.Output(name));
}

template <>
const std::vector<const Variable*> OperatorContext::MultiInput(
    const std::string& name) const {
  auto names = op_.Inputs(name);
  std::vector<const Variable*> res;
  res.reserve(names.size());
  std::transform(
      names.begin(), names.end(), std::back_inserter(res),
      [this](const std::string& name) { return scope_->GetVariable(name); });
  return res;
}

template <>
std::vector<const Variable*> OperatorContext::MultiOutput(
    const std::string& name) const {
  auto names = op_.Outputs(name);
  std::vector<const Variable*> res;
  res.reserve(names.size());
  std::transform(
      names.begin(), names.end(), std::back_inserter(res),
      [this](const std::string& name) { return scope_->GetVariable(name); });
  return res;
}

}  // namespace framework
}  // namespace paddle
