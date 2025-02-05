// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/yield_instruction.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

namespace paddle {
namespace framework {

YieldInstruction::YieldInstruction(size_t id,
                                   const phi::Place &place,
                                   ::pir::Operation *op,
                                   ValueExecutionInfo *value_exe_info)
    : InstructionBase(id, place), op_(op) {
  VLOG(6) << "construct yield instruction";

  auto parent_op = op->GetParentOp();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    // Skip the first input (cond) when the parent op is a while op.
    if (parent_op->isa<paddle::dialect::WhileOp>() && i == 0) {
      continue;
    }
    auto in = op->operand_source(i);
    if (in && in.type()) {
      inputs.emplace(in, GetValueIds(in, *value_exe_info));
      input_vars_.push_back(value_exe_info->GetVarByValue(in));
    }
  }
  SetInputs(inputs);

  for (size_t i = 0; i < parent_op->num_results(); ++i) {
    if (parent_op->result(i) && parent_op->result(i).type()) {
      output_vars_.push_back(
          value_exe_info->GetVarByValue(parent_op->result(i)));
    }
  }

  PADDLE_ENFORCE_EQ(
      input_vars_.size(),
      output_vars_.size(),
      common::errors::InvalidArgument("The number of inputs in YieldOp and "
                                      "outputs of parent op must be equal."
                                      "But received %d and %d.",
                                      input_vars_.size(),
                                      output_vars_.size()));
}

void YieldInstruction::Run() {
  for (size_t i = 0; i < input_vars_.size(); ++i) {
    if (input_vars_[i]->IsType<phi::DenseTensor>()) {
      output_vars_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
          input_vars_[i]->Get<phi::DenseTensor>());
    } else if (input_vars_[i]->IsType<phi::TensorArray>()) {
      const auto &inner_array = input_vars_[i]->Get<phi::TensorArray>();
      auto *output_array = output_vars_[i]->GetMutable<phi::TensorArray>();
      *output_array = inner_array;
    } else {
      PADDLE_THROW(common::errors::Unimplemented("unsupported type %d",
                                                 input_vars_[i]->Type()));
    }
  }
}

}  // namespace framework
}  // namespace paddle
