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

namespace pir {
class Operation;
class Value;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;

class LegacyKernelInstruction : public InstructionBase {
 public:
  LegacyKernelInstruction(
      size_t id,
      const platform::Place& place,
      ::pir::Operation* op,
      Scope* scope,
      Scope* local_scope,
      const std::unordered_map<::pir::Value, std::string>& value_2_var_name,
      const std::map<std::string, int>& var_name_2_id,
      const std::unordered_map<const paddle::framework::Variable*, std::string>&
          variable_2_var_name);

  ~LegacyKernelInstruction();
  phi::Kernel* PhiKernel() const { return phi_kernel_; }

  const phi::InferMetaContext& InferMetaContext() const {
    return infer_meta_context_;
  }

  paddle::dialect::InferMetaInterface::Concept* InferMetaInterface() const {
    return infer_meta_interface_;
  }

  void Run() override;

  const std::string& Name() const override { return legacy_op_name_; }

  OperatorBase* OpBase() const override;

 private:
  std::string legacy_op_name_;

  paddle::dialect::InferMetaInterface::Concept* infer_meta_interface_{
      nullptr};  // not owned

  phi::InferMetaContext infer_meta_context_;

  paddle::framework::ExecutionContext* kernel_context_{nullptr};
  std::shared_ptr<framework::RuntimeContext> runtime_context_;
  std::shared_ptr<paddle::framework::OperatorBase> operator_base_;

  phi::Kernel* phi_kernel_{nullptr};  // not owned
};

}  // namespace framework
}  // namespace paddle
