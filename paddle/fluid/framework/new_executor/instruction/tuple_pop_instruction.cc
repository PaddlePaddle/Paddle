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
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
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
  auto var_array = value_exe_info_->GetVarByValue(stack_value);
  stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
    auto outlet_element_value = tuple_pop_op_.outlet_element(i);
    outputs.emplace(outlet_element_value,
                    GetValueIds(outlet_element_value, *value_exe_info_));
  }
  SetOutputs(outputs);

  type_ = OpFuncType::kCpuSync;
}

void TuplePopInstruction::Run() {
  if (tuple_pop_op_.tuple_size() == 0) {
    stack_element_var_array_->pop_back();
  } else {
    for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
      auto front_var =
          stack_element_var_array_->at(tuple_pop_op_.tuple_size() - i - 1);
      auto outlet_element_value = tuple_pop_op_.outlet_element(i);

      auto grad_var = value_exe_info_->GetVarByValue(outlet_element_value);
      grad_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
          front_var->Get<phi::DenseTensor>());

      stack_element_var_array_->pop_back();
      Variable* gc_front_var = const_cast<Variable*>(front_var);
      AddEagerGCVar(gc_front_var);
    }
  }
}
}  // namespace framework
}  // namespace paddle
