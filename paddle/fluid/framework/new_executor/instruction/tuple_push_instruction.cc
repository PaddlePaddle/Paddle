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

#include "paddle/fluid/framework/new_executor/instruction/tuple_push_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle {
namespace framework {
TuplePushInstruction::TuplePushInstruction(size_t id,
                                           const platform::Place& place,
                                           ::pir::Operation* op,
                                           ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place) {
  tuple_push_op_ = op->dyn_cast<pir::TuplePushOp>();
  value_exe_info_ = value_exe_info;
  auto stack_value = tuple_push_op_.container();
  auto& value_2_var_name = value_exe_info_->GetValue2VarName();
  PADDLE_ENFORCE_EQ(
      value_2_var_name.find(stack_value) != value_2_var_name.end(),
      true,
      phi::errors::NotFound(
          "stack input of PushBackOp not in value2varname map"));
  auto var_array =
      value_exe_info_->GetScope()->FindVar(value_2_var_name.at(stack_value));
  variable_ref_array_ = var_array->GetMutable<VariableRefArray>();
}

void TuplePushInstruction::Run() {
  if (tuple_push_op_.tuple_size() == 0) {
    Variable* var = nullptr;
    variable_ref_array_->emplace_back(var);
  } else {
    auto& value_2_var_name = value_exe_info_->GetValue2VarName();
    for (size_t i = tuple_push_op_.tuple_size() - 1; i >= 0; ++i) {
      auto inlet_element = tuple_push_op_.inlet_element(i);
      Variable* var = value_exe_info_->GetScope()->FindVar(
          value_2_var_name.at(inlet_element));

      uint32_t stack_size = tuple_push_op_.tuple_size();
      std::string new_name =
          "copy_" + stack_size + '_' + value_exe_info_->GetVarName(var);
      auto copy_var = value_exe_info_->GetScope()->Var(new_name);
      DeepCopyVariable(var, copy_var, value_exe_info_, stack_size);
      variable_ref_array_->emplace_back(copy_var);
    }
  }
}
}  // namespace framework
}  // namespace paddle
