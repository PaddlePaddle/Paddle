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

#include "paddle/fluid/framework/new_executor/instruction/has_elements_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/tensor_ref_array.h"

namespace paddle {
namespace framework {
HasElementsInstruction::HasElementsInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  auto has_elements_op = op->dyn_cast<pir::TuplePopOp>();
  VLOG(6) << "construct has_elements instruction for: "
          << has_elements_op->name();
}

void HasElementsInstruction::Run() {
  auto stack_value = op_->dyn_cast<pir::HasElementsOp>().operand_source(0);
  auto& value_2_var_name = value_exe_info_->GetValue2VarName();
  auto var_array =
      value_exe_info_->GetScope()->FindVar(value_2_var_name.at(stack_value));
  auto stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();
  bool is_empty = stack_element_var_array_->empty();
  auto bool_var =
      value_exe_info_->GetScope()->FindVar(value_2_var_name.at(op_->result(0)));
  bool* bool_ptr = bool_var->GetMutable<phi::DenseTensor>()->data<bool>();
  *bool_ptr = is_empty;
}
}  // namespace framework
}  // namespace paddle
