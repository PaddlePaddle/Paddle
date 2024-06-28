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

#include "paddle/fluid/framework/new_executor/instruction/register_hook_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/pir/include/core/builtin_attribute.h"
// #include "pybind11/gil.h"
// #include "pybind11/pytypes.h"
// #include "pybind11/cast.h"

namespace paddle {
namespace framework {

RegisterHookInstruction::RegisterHookInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place),
      op_(op),
      type_(OpFuncType::kCpuSync),
      value_exe_info_(value_exe_info) {
  PADDLE_ENFORCE(
      op->isa<paddle::dialect::RegisterHookOp>(),
      phi::errors::PreconditionNotMet(
          "register_hook instruction only support register_hook op"));

  auto register_hook_op = op->dyn_cast<paddle::dialect::RegisterHookOp>();
  VLOG(6) << "construct register_hook instruction for: "
          << register_hook_op->name();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  inputs.emplace(register_hook_op.input(),
                 GetValueIds(register_hook_op.input(), *value_exe_info_));
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  outputs.emplace(register_hook_op.out(),
                  GetValueIds(register_hook_op.out(), *value_exe_info_));
  SetOutputs(outputs);

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto input_value = register_hook_op.operand_source(0);
  input_var_ = value_exe_info_->GetVarByValue(input_value);
  auto output_value = register_hook_op.result(0);
  output_var_ = value_exe_info_->GetVarByValue(output_value);

  VLOG(6) << "finish process input_var_ and output_var_";
}

void RegisterHookInstruction::Run() {
  namespace py = pybind11;
  //  Maybe it needs to be added GIL
  py::gil_scoped_acquire guard;
  const phi::DenseTensor& input = input_var_->Get<phi::DenseTensor>();

  PADDLE_ENFORCE_EQ(
      input.numel(),
      1,
      platform::errors::InvalidArgument(
          "The numel of Input of RegisterHookOp must be 1. But now "
          "the input's shape is %s.",
          input.dims().to_str()));

  void* func_void_ptr =
      op_->attribute<::pir::PointerAttribute>("forward_hook_func").data();

  // Note: The func when performing the grad, forward_hook_func it's grad func.
  PADDLE_ENFORCE_NOT_NULL(func_void_ptr,
                          platform::errors::InvalidArgument(
                              "The func of RegisterHookOp must not be NULL."));

  // func return Variable
  auto callable = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject*>(func_void_ptr));

  if (callable.is_none()) {
    // 或许应该使用深拷贝
    output_var_ = input_var_;
    return;
  }

  //  py::object result = callable(input_var_);
  py::object result = callable(input);
  // output_var_ = py::handle(result).cast<Variable*>();
}

}  // namespace framework
}  // namespace paddle
