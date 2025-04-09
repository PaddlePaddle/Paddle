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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/assert_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace paddle::framework {
AssertInstruction::AssertInstruction(size_t id,
                                     const phi::Place& place,
                                     ::pir::Operation* op,
                                     ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place),
      op_(op),
      type_(OpFuncType::kCpuSync),
      value_exe_info_(value_exe_info) {
  PADDLE_ENFORCE(op->isa<paddle::dialect::AssertOp>(),
                 common::errors::PreconditionNotMet(
                     "Assert instruction only support assert op"));

  auto assert_op = op->dyn_cast<paddle::dialect::AssertOp>();
  VLOG(6) << "construct assert instruction for: " << assert_op->name();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  inputs.emplace(assert_op.cond(),
                 GetValueIds(assert_op.cond(), *value_exe_info_));
  inputs.emplace(assert_op.data(),
                 GetValueIds(assert_op.data(), *value_exe_info_));
  SetInputs(inputs);

  op_ = op;
  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto cond_value = assert_op.operand_source(0);
  cond_var_ = value_exe_info_->GetVarByValue(cond_value);
  auto data_value = assert_op.operand_source(1);
  data_var_ = value_exe_info_->GetVarByValue(data_value);
  VLOG(6) << "finish process cond_var and data_var";
}

void AssertInstruction::Run() {
  DeviceContext().Wait();
  const phi::DenseTensor& cond = cond_var_->Get<phi::DenseTensor>();

  PADDLE_ENFORCE_EQ(
      cond.numel(),
      1,
      common::errors::InvalidArgument(
          "The numel of Input(Condition) of AssertOp must be 1. But now "
          "the Condition's shape is %s.",
          cond.dims().to_str()));

  bool cond_data = GetCondData(cond);
  if (cond_data) {
    return;
  }

  funcs::TensorFormatter formatter;
  formatter.SetSummarize(
      op_->attribute<::pir::Int64Attribute>("summarize").data());

  const std::vector<pir::Value>& inputs_data_val =
      op_->dyn_cast<paddle::dialect::AssertOp>()
          .data()
          .defining_op<::pir::CombineOp>()
          .inputs();
  for (pir::Value val : inputs_data_val) {
    const std::string& name = value_exe_info_->GetVarName(val);
    const phi::DenseTensor& tensor =
        value_exe_info_->GetVarByValue(val)->Get<phi::DenseTensor>();
    formatter.Print(tensor, name);
  }
  const std::string& error_msg = [&]() -> std::string {
    if (op_->HasAttribute(paddle::dialect::AssertOp::ERROR_INFO_ATTR_NAME)) {
      return op_
          ->attribute<pir::StrAttribute>(
              paddle::dialect::AssertOp::ERROR_INFO_ATTR_NAME)
          .AsString();
    }
    return {};
  }();
  PADDLE_THROW(common::errors::InvalidArgument(
      "The condition variable '%s' of AssertOp must be "
      "true, but received false. %s",
      value_exe_info_->GetVarName(cond_var_),
      error_msg));
}

}  // namespace paddle::framework
