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
#include <algorithm>
#include "paddle/framework/op_registry.h"

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

static std::unordered_map<std::string, OpProto>* g_op_protos = nullptr;
std::unordered_map<std::string, OpProto>& OpProtos() {
  if (g_op_protos == nullptr) {
    g_op_protos = new std::unordered_map<std::string, OpProto>();
  }
  return *g_op_protos;
}

const std::string& OperatorBase::Input(const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Op %s does not have input %s", type_,
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
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
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

OperatorBase::OperatorBase(const std::string& type,
                           const OperatorBase::VarNameMap& inputs,
                           const OperatorBase::VarNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}
}  // namespace framework
}  // namespace paddle
