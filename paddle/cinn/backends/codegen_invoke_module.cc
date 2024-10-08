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

#include "paddle/cinn/backends/codegen_invoke_module.h"

#include <vector>
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {

llvm::Value* CodeGenInvokeModule::LowerInvokeFunc(
    const ir::_LoweredFunc_* func) {
  // Create the function
  // @{
  auto* function_type = GenFunctionTypeFromCinnFunction(func, true);
  f_ = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage, func->name, m_);
  f_->setCallingConv(llvm::CallingConv::C);
  f_->setHasUWTable();

  std::vector<llvm::Value*> ll_function_args;
  std::transform(f_->arg_begin(),
                 f_->arg_end(),
                 std::back_inserter(ll_function_args),
                 [](auto& arg) { return std::addressof(arg); });
  // @}

  // Set local scope table
  PADDLE_ENFORCE_EQ(ll_function_args.size(),
                    func->args.size(),
                    ::common::errors::InvalidArgument(
                        "The number of arguments is not equal to the number of "
                        "function arguments"));
  for (int i = 0; i < ll_function_args.size(); ++i) {
    SetVar(func->args[i].name(), ll_function_args[i]);
  }
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/f_,
      /*InsertBefore=*/nullptr);
  b_->SetInsertPoint(entry);
  CodeGenLLVM::Visit(&func->body);

  // Reset local scope table
  for (const ir::Argument& func_arg : func->args) {
    symbol_table_->Erase(func_arg.name());
  }
  RetVoid();

  return f_;
}

llvm::Value* CodeGenSwitchHost::LowerInnerCaseCall(const ir::Call* op) {
  std::vector<llvm::Value*> ll_function_args;
  std::transform(f_->arg_begin(),
                 f_->arg_end(),
                 std::back_inserter(ll_function_args),
                 [](auto& arg) { return std::addressof(arg); });
  // TODO(Hongqing-work): Add check for parameter type
  llvm::Function* call_func = m_->getFunction(op->name);
  PADDLE_ENFORCE_NOT_NULL(
      call_func,
      ::common::errors::InvalidArgument("Unknown function referenced. [%s]",
                                        op->name.c_str()));
  b_->CreateCall(call_func, ll_function_args);
  return nullptr;
}
}  // namespace backends
}  // namespace cinn
