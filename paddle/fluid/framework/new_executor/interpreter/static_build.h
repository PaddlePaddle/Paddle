// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

extern std::set<std::string> OperatorBasesMustRunInStaticBuild;

namespace paddle {
namespace framework {
namespace interpreter {

struct VarMetaInfo {
  std::string name_;
  phi::DataType dtype_;
  phi::Place place_;

  explicit VarMetaInfo(const std::string& name) : name_(name) {
    dtype_ = phi::DataType::UNDEFINED;
    place_ = phi::Place();
  }

  VarMetaInfo(const std::string& name,
              const phi::DataType& dtype,
              const phi::Place& place)
      : name_(name), dtype_(dtype), place_(place) {}

  bool operator==(const VarMetaInfo& other) const {
    return name_ == other.name_ && dtype_ == other.dtype_ &&
           place_ == other.place_;
  }

  bool operator!=(const VarMetaInfo& other) const {
    return name_ != other.name_ || dtype_ != other.dtype_ ||
           place_ != other.place_;
  }
};

bool BlockCanBeStaticBuilt(const framework::BlockDesc& block);

void FakeInitializeOutputsForOperatorBase(
    const OperatorBase& op,
    const phi::Place& place,
    Scope* scope,
    std::vector<std::shared_ptr<OperatorBase>> following_ops);

void FakeInitializeOutputsForFunctionKernel(
    const framework::OperatorBase& op,
    const phi::Kernel& phi_kernel,
    const phi::KernelSignature& kernel_sig,
    const RuntimeContext& ctx,
    const phi::DeviceContext& dev_ctx);

void FakeInitializeOutputsForStructureKernel(
    const framework::OpKernelType& op_kernel_type,
    ExecutionContext* execution_context);

std::vector<VarMetaInfo> GetVarsInfo(const Scope* scope,
                                     VariableNameMap var_map,
                                     const OperatorBase& op);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
