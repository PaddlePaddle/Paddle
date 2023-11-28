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
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle {
namespace framework {
HasElementsInstruction::HasElementsInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  auto has_elements_op = op->dyn_cast<pir::HasElementsOp>();
  VLOG(6) << "construct has_elements instruction for: "
          << has_elements_op->name();

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  outputs.emplace(has_elements_op.out(),
                  GetValueIds(has_elements_op.out(), *value_exe_info_));
  SetOutputs(outputs);

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  inputs.emplace(has_elements_op.input(),
                 GetValueIds(has_elements_op.input(), *value_exe_info_));
  SetInputs(inputs);

  type_ = OpFuncType::kCpuSync;

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  dev_ctx_ = pool.Get(platform::CPUPlace());

  auto stack_value = op_->dyn_cast<pir::HasElementsOp>().operand_source(0);
  auto var_array = value_exe_info_->GetVarByValue(stack_value);
  auto stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();
}

void HasElementsInstruction::Run() {
  bool is_empty = stack_element_var_array_->size();
  auto bool_var = value_exe_info_->GetVarByValue(op_->result(0));
  auto* bool_tensor = bool_var->GetMutable<phi::DenseTensor>();

  bool* bool_ptr = dev_ctx_->Alloc<bool>(bool_tensor);
  *bool_ptr = is_empty;
}
}  // namespace framework
}  // namespace paddle
