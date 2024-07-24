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
                    phi::errors::InvalidArgument(
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

llvm::Value* CodeGenInvokeModule::LowerParseArgsValueCall(
    const ir::Call* call_ir) {
  auto ret_type = CinnTypeToLLVMType(Int(64), m_);
  std::vector<llvm::Type*> args_type;
  PADDLE_ENFORCE_EQ(
      call_ir->read_args.size(),
      2,
      phi::errors::InvalidArgument(
          "The number of arguments of ParseArgsValue should be 2"));
  CHECK(call_ir->read_args[0].is_var() &&
        call_ir->read_args[0].as_var()->type().is_cpp_handle());
  CHECK(call_ir->read_args[1].type().is_int(32));
  args_type.push_back(CinnTypeToLLVMType(type_of<void*>(), m_));
  args_type.push_back(CinnTypeToLLVMType(type_of<int32_t>(), m_));

  auto func_type = llvm::FunctionType::get(ret_type, args_type, false);
  auto call_func = m_->getOrInsertFunction(call_ir->name, func_type);

  std::vector<llvm::Value*> call_args;
  call_args.push_back(std::addressof(*f_->arg_begin()));
  call_args.push_back(b_->getInt32(call_ir->read_args[1].as_int32()));
  return b_->CreateCall(call_func, call_args);
}

llvm::Value* CodeGenSwitchHost::LowerInnerCaseCall(const ir::Call* op) {
  std::vector<llvm::Value*> ll_function_args;
  std::transform(f_->arg_begin(),
                 f_->arg_end(),
                 std::back_inserter(ll_function_args),
                 [](auto& arg) { return std::addressof(arg); });
  // TODO(Hongqing-work): Add check for parameter type
  llvm::Function* call_func = m_->getFunction(op->name);
  CHECK(call_func) << "Unknown function referenced. [" << op->name << "]";
  b_->CreateCall(call_func, ll_function_args);
  return nullptr;
}
}  // namespace backends
}  // namespace cinn
