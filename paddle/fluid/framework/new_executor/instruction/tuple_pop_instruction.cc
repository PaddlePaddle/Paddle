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

#include "paddle/fluid/framework/new_executor/instruction/tuple_pop_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle {
namespace framework {
TuplePopInstruction::TuplePopInstruction(size_t id,
                                         const platform::Place& place,
                                         ::pir::Operation* op,
                                         ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  tuple_pop_op_ = op->dyn_cast<pir::TuplePopOp>();
  VLOG(6) << "construct tuple_pop instruction for: " << tuple_pop_op_->name();
  auto stack_value = tuple_pop_op_.container();
  auto& value_2_var_name = value_exe_info_->GetValue2VarName();
  PADDLE_ENFORCE_EQ(
      value_2_var_name.find(stack_value) != value_2_var_name.end(),
      true,
      phi::errors::NotFound(
          "stack input of PopBackOp not in value2varname map"));
  auto var_array =
      value_exe_info_->GetScope()->FindVar(value_2_var_name.at(stack_value));
  stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();
}

void TuplePopInstruction::Run() {
  auto& value_2_var_name = value_exe_info_->GetValue2VarName();
  if (tuple_pop_op_.tuple_size() == 0) {
    stack_element_var_array_->pop_back();
    auto outlet_element = tuple_pop_op_.outlet_element(0);
    auto grad_var = value_exe_info_->GetScope()->FindVar(
        value_2_var_name.at(outlet_element));
    grad_var = nullptr;
  } else {
    for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
      auto front_var = stack_element_var_array_->back();
      stack_element_var_array_->pop_back();

      auto outlet_element = tuple_pop_op_.outlet_element(i);
      auto grad_var = value_exe_info_->GetScope()->FindVar(
          value_2_var_name.at(outlet_element));

      grad_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
          front_var->Get<phi::DenseTensor>());

      tuple_pop_gc_var_ids_.insert(value_exe_info_->GetVarId(grad_var));
    }
  }
}
}  // namespace framework
}  // namespace paddle
