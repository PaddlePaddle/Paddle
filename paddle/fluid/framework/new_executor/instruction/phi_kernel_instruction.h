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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"

namespace paddle {
namespace framework {

// struct OpFuncNode {

//   // fit for phi kernel
//   phi::Kernel* phi_kernel_{nullptr};  // not owned
//   platform::DeviceContext* dev_ctx_;  // not owned

//   // TODO(zhiqiu): Better make it unique_ptr
//   std::shared_ptr<OperatorBase> operator_base_{nullptr};

//   OpKernelComputeFunc kernel_func_;

//   // the next only for new IR
//   phi::KernelContext kernel_context_;
//   phi::InferMetaContext infer_meta_context_;
//   std::string phi_op_name_;
//   paddle::dialect::InferMetaInterface::Concept*
//   infer_meta_interface_{nullptr};
// };

class PhiKernelInstruction : public InstructionBase {
 public:
  //   OpKernelComputeFunc KernelFunc() const;

  //   phi::Kernel* PhiKernel() const;

  //   const std::map<int, int>& InplaceBackMap() const;

  //   OperatorBase* OpBase() const;

  //   bool OpBaseValid() const;

  //   void ResetContext(const VariableValueMap& in_vars,
  //                     const VariableValueMap& out_vars);

  //   void ResetContextWithScope(const VariableValueMap& in_vars,
  //                              const VariableValueMap& out_vars,
  //                              const framework::Scope& scope);

  //   std::shared_ptr<RuntimeContext> InnerRuntimeContext() const;

  //   std::shared_ptr<RuntimeInferShapeContext> InnerInferShapeContext() const;

  //   std::shared_ptr<ExecutionContext> InnerExecutionContext() const;

  //   std::shared_ptr<RuntimeContext> runtime_ctx_;
  //   std::shared_ptr<RuntimeInferShapeContext> infershape_ctx_;
  //   std::shared_ptr<ExecutionContext> execution_ctx_;
};

}  // namespace framework
}  // namespace paddle
