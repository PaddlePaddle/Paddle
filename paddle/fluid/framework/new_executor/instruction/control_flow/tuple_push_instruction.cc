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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/tuple_push_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/phi/core/compat/convert_utils.h"

namespace paddle::framework {
bool ParsePlace(const pir::Type& type, OpFuncType* type_) {
  if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
    auto place =
        type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>().place();
    if (place == phi::GPUPlace()) {
      *type_ = OpFuncType::kGpuAsync;
      return true;
    }
  } else if (type.isa<paddle::dialect::AllocatedDenseTensorArrayType>()) {
    auto place =
        type.dyn_cast<paddle::dialect::AllocatedDenseTensorArrayType>().place();
    if (place == phi::GPUPlace()) {
      *type_ = OpFuncType::kGpuAsync;
      return true;
    }
  } else if (type.isa<pir::VectorType>()) {
    pir::VectorType inlet_element_type = type.dyn_cast<pir::VectorType>();
    for (size_t i = 0; i < static_cast<size_t>(inlet_element_type.size());
         i++) {
      if (ParsePlace(inlet_element_type[i], type_)) {
        return true;
      }
    }
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Only support AllocatedDenseTensorType and "
        "AllocatedDenseTensorArrayType in vectortype now, but get: %s",
        type));
  }
  return false;
}

TuplePushInstruction::TuplePushInstruction(size_t id,
                                           const phi::Place& place,
                                           ::pir::Operation* op,
                                           ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  tuple_push_op_ = op->dyn_cast<pir::TuplePushOp>();
  VLOG(6) << "construct tuple_push instruction for: " << tuple_push_op_->name();
  auto stack_value = tuple_push_op_.container();
  auto var_array = value_exe_info_->GetVarByValue(stack_value);
  stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < tuple_push_op_.tuple_size(); ++i) {
    auto inlet_element_value = tuple_push_op_.inlet_element(i);
    inputs.emplace(inlet_element_value,
                   GetValueIds(inlet_element_value, *value_exe_info_));
  }
  SetInputs(inputs);

  // NOTE(zhangbo): TuplePush will change the variables corresponding to the
  // inlet, so it needs to be marked as output.
  std::unordered_map<pir::Value, std::vector<int>> outputs;
  outputs.emplace(tuple_push_op_.inlet(),
                  std::initializer_list<int>{
                      value_exe_info_->GetVarId(tuple_push_op_.inlet())});
  SetOutputs(outputs);

  type_ = OpFuncType::kCpuSync;
  for (size_t i = 0; i < tuple_push_op_.tuple_size(); ++i) {
    auto inlet_element_value = tuple_push_op_.inlet_element(i);
    if (ParsePlace(inlet_element_value.type(), &type_)) {
      break;
    }
  }
}

void TuplePushInstruction::Run() {
  VLOG(4) << "run tuple_push instruction";
  if (tuple_push_op_.tuple_size() == 0) {
    stack_element_var_array_->emplace_back(nullptr);
  } else {
    auto& value_2_var_name = value_exe_info_->GetValue2VarName();
    // TODO(zhangbo): Performance optimization: static acquisition of TuplePush
    // input variables and name.
    std::map<const Variable*, Variable*> src_to_dst;
    for (size_t i = 0; i < tuple_push_op_.tuple_size(); i++) {
      auto inlet_element_value = tuple_push_op_.inlet_element(i);
      Variable* var = value_exe_info_->GetVarByValue(inlet_element_value);

      auto var_name = value_2_var_name.at(inlet_element_value);
      auto num_str = std::to_string(stack_element_var_array_->size());
      std::string new_name = var_name + "_copied_" + num_str + "_in_tuple_" +
                             std::to_string(op_->id());
      auto* copy_var = value_exe_info_->GetScope()->Var(new_name);
      bool is_optional = (inlet_element_value.impl() == nullptr ||
                          !inlet_element_value.type());
      DeepCopyVariable(var,
                       &copy_var,
                       value_exe_info_,
                       stack_element_var_array_->size(),
                       is_optional,
                       &src_to_dst);
      VLOG(10) << "done DeepCopyVariable " << new_name;
      stack_element_var_array_->emplace_back(copy_var);
      VLOG(6) << "push back var: " << new_name << "[" << copy_var << "]"
              << "to: " << stack_element_var_array_;
    }
  }
}
}  // namespace paddle::framework
