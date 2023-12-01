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

#include <stack>

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

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  inputs.emplace(tuple_pop_op_.outlet(),
                 std::initializer_list<int>{
                     value_exe_info_->GetVarId(tuple_pop_op_.outlet())});
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
    auto outlet_element_value = tuple_pop_op_.outlet_element(i);
    outputs.emplace(outlet_element_value,
                    GetValueIds(outlet_element_value, *value_exe_info_));
  }

  // NOTE(zhangbo): TuplePop will change the variables corresponding to the
  // outlet, so it needs to be marked as output.
  outputs.emplace(tuple_pop_op_.outlet(),
                  std::initializer_list<int>{
                      value_exe_info_->GetVarId(tuple_pop_op_.outlet())});
  SetOutputs(outputs);

  type_ = OpFuncType::kCpuSync;
}

// NOTE(zhangbo): TuplePop is an instruction used in conjunction with TuplePush.
// Each TuplePush pushes 0-n variables into a
// VariableRefArray(vector<Variable*>), which may be executed any number of
// times; TuplePop needs to retrieve variables from
// VariableRefArray(vector<Variable*>) before and after, corresponding to 0-n
// variables at a time. Therefore, a Stack data structure is designed here to
// implement the function of extracting variables from vector<Variable*>
// mentioned above.
static std::stack<const Variable*> PopElements(VariableRefArray* var_array,
                                               uint64_t num) {
  std::stack<const Variable*> rtn;
  for (uint64_t i = 0; i < num; i++) {
    rtn.push(var_array->back());
    var_array->pop_back();
  }
  return rtn;
}

void TuplePopInstruction::Run() {
  VLOG(6) << "run tuple_pop instruction";
  if (tuple_pop_op_.tuple_size() == 0) {
    stack_element_var_array_->pop_back();
  } else {
    std::stack<const Variable*> var_elements =
        PopElements(stack_element_var_array_, tuple_pop_op_.tuple_size());
    // TODO(zhangbo): Performance optimization: static acquisition of TuplePoop
    // output variables.
    for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
      auto front_var = var_elements.top();
      var_elements.pop();
      VLOG(6) << "pop back var: " << front_var;
      auto outlet_element_value = tuple_pop_op_.outlet_element(i);
      auto grad_var = value_exe_info_->GetVarByValue(outlet_element_value);
      grad_var->GetMutable<phi::DenseTensor>()->ShareDataWith(
          front_var->Get<phi::DenseTensor>());

      Variable* gc_front_var = const_cast<Variable*>(front_var);
      AddEagerGCVar(gc_front_var);
    }
  }
}

}  // namespace framework
}  // namespace paddle
