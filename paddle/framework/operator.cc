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
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_EQ(ins.size(), 1UL,
                    "Op %s input %s should contain only one variable", type_,
                    name);
  return ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Op %s do not have input %s", type_,
                 name);
  return it->second;
}

const std::string& OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_EQ(outs.size(), 1UL,
                    "Op %s output %s should contain only one variable", type_,
                    name);
  return outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(), "Op %s does not have output %s", type_,
                 name);
  return it->second;
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

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto it = OpProtos().find(type_);
  PADDLE_ENFORCE(
      it != OpProtos().end(),
      "Operator %s not registered, cannot figure out intermediate outputs",
      type_);

  // get all OpProto::Var for outputs
  for (auto& o : it->second.outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

}  // namespace framework
}  // namespace paddle
