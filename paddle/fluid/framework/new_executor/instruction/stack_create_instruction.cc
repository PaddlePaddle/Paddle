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

#include "paddle/fluid/framework/new_executor/instruction/stack_create_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle {
namespace framework {
StackCreateInstruction::StackCreateInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place) {
  PADDLE_ENFORCE(
      op->isa<pir::StackCreateOp>(),
      phi::errors::PreconditionNotMet("Cond instruction only support if op"));
  auto stack_create_op = op->dyn_cast<pir::StackCreateOp>();
  op_ = op;

  auto& value_2_var_name = value_exe_info->GetValue2VarName();
  auto stack_value = stack_create_op.stack();
  std::stringstream ss;
  ss << this
     << std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::string var_name_prefix = ss.str();
  var_ = CreateVar(stack_value, var_name_prefix, false, value_exe_info);
  var_->GetMutable<VariableRefArray>();
}

void StackCreateInstruction::Run() {
  std::cout << "StackCreateInstruction.run";
}
}  // namespace framework
}  // namespace paddle
