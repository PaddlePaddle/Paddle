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

template <>
Eigen::DefaultDevice* OpKernel::KernelContext::get_eigen_device<
    platform::CPUPlace, Eigen::DefaultDevice>() const {
  return device_context_.get_eigen_device<Eigen::DefaultDevice>();
}

#ifndef PADDLE_ONLY_CPU
template <>
Eigen::GpuDevice* OpKernel::KernelContext::get_eigen_device<
    platform::GPUPlace, Eigen::GpuDevice>() const {
  return device_context_.get_eigen_device<Eigen::GpuDevice>();
}
#endif

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "=================\n";
  ss << "type = " << type_ << "\n";
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
