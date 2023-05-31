// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/codegen_cuda_host.h"

#include <algorithm>
#include <string>
#include <unordered_map>

#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/extern_func_emitter_builtin.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

using cinn::common::bfloat16;
using cinn::common::float16;

const int kArgsArrayMaxLen = 20;

llvm::Value* CodeGenCUDA_Host::LowerGPUKernelLauncher(
    const ir::_LoweredFunc_* func) {
  auto body = func->body;
  auto* call_ir = body.As<ir::Call>();
  CHECK(call_ir);

  // Create the function
  // @{
  auto* function_type = GenFunctionTypeFromCinnFunction(func, true);
  llvm::Function* function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage, func->name, m_);
  function->setCallingConv(llvm::CallingConv::C);
  function->setHasUWTable();

  std::vector<llvm::Value*> ll_function_args;
  std::transform(function->arg_begin(),
                 function->arg_end(),
                 std::back_inserter(ll_function_args),
                 [](auto& arg) { return std::addressof(arg); });
  // @}

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function,
      /*InsertBefore=*/nullptr);
  b_->SetInsertPoint(entry);

  auto* kernel_args = ll_function_args[0];
  auto* kernel_args_count = ll_function_args[1];
  llvm::Value* kernel_stream = nullptr;
  if (ll_function_args.size() == 3) {
    kernel_stream = ll_function_args[2];
    CHECK_EQ(kernel_stream->getType(), ll_void_p_ty());  // void* stream
  }
  CHECK_EQ(kernel_args->getType(), ll_void_p_ty());       // void* args
  CHECK_EQ(kernel_args_count->getType(), ll_int32_ty());  // int32

  std::unordered_map<std::string, llvm::Value*> global_args = {
      {KERNEL_ARGS, kernel_args},
      {KERNEL_ARGS_NUM, kernel_args_count},
      {KERNEL_STREAM, kernel_stream}};

  auto ret_type = CinnTypeToLLVMType(Void(), m_);
  std::vector<llvm::Type*> args_type;
  for (auto r_arg : call_ir->read_args) {
    if (r_arg.is_var()) {
      if (r_arg.as_var()->type().is_cpp_handle() ||
          r_arg.as_var()->type().is_string()) {
        args_type.push_back(CinnTypeToLLVMType(type_of<void*>(), m_));
      } else if (r_arg.as_var()->type().is_int(32)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<int32_t>(), m_));
      } else {
        CINN_NOT_IMPLEMENTED;
      }
    } else {
      if (r_arg.type().is_bool()) {
        args_type.push_back(CinnTypeToLLVMType(type_of<bool>(), m_));
      } else if (r_arg.type().is_uint(8)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<uint8_t>(), m_));
      } else if (r_arg.type().is_uint(16)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<uint16_t>(), m_));
      } else if (r_arg.type().is_uint(32)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<uint32_t>(), m_));
      } else if (r_arg.type().is_uint(64)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<uint64_t>(), m_));
      } else if (r_arg.type().is_int(8)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<int8_t>(), m_));
      } else if (r_arg.type().is_int(16)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<int16_t>(), m_));
      } else if (r_arg.type().is_int(32)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<int32_t>(), m_));
      } else if (r_arg.type().is_int(64)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<int64_t>(), m_));
      } else if (r_arg.type().is_float(32)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<float>(), m_));
      } else if (r_arg.type().is_float(64)) {
        args_type.push_back(CinnTypeToLLVMType(type_of<double>(), m_));
      } else if (r_arg.type().is_bfloat16()) {
        args_type.push_back(CinnTypeToLLVMType(type_of<bfloat16>(), m_));
      } else if (r_arg.type().is_float16()) {
        args_type.push_back(CinnTypeToLLVMType(type_of<float16>(), m_));
      } else {
        CINN_NOT_IMPLEMENTED;
      }
    }
  }
  auto func_type = llvm::FunctionType::get(ret_type, args_type, false);
  auto call_func = m_->getOrInsertFunction(call_ir->name, func_type);

  std::vector<llvm::Value*> call_args;
  for (auto& r_arg : call_ir->read_args) {
    if (r_arg.is_var()) {
      if (r_arg.as_var()->type().is_string()) {
        auto kvalue = m_->getOrInsertGlobal(r_arg.as_var()->name + "_ptr_",
                                            b_->getInt8PtrTy());
        call_args.push_back(b_->CreateLoad(
            b_->getInt8PtrTy(), kvalue, r_arg.as_var()->name + "_ptr_load"));
      } else if (r_arg.as_var()->type().is_cpp_handle() ||
                 r_arg.as_var()->type().is_int(32)) {
        CHECK(global_args.count(r_arg.as_var()->name));
        call_args.push_back(global_args[r_arg.as_var()->name]);
      } else {
        CINN_NOT_IMPLEMENTED;
      }
    } else {
      if (r_arg.type().is_bool()) {
        call_args.push_back(b_->getInt1(r_arg.as_bool()));
      } else if (r_arg.type().is_int(8)) {
        call_args.push_back(b_->getInt8(r_arg.as_int8()));
      } else if (r_arg.type().is_int(16)) {
        call_args.push_back(b_->getInt16(r_arg.as_int16()));
      } else if (r_arg.type().is_int(32)) {
        call_args.push_back(b_->getInt32(r_arg.as_int32()));
      } else if (r_arg.type().is_int(64)) {
        call_args.push_back(b_->getInt64(r_arg.as_int64()));
      } else if (r_arg.type().is_uint(8)) {
        call_args.push_back(b_->getInt8(r_arg.as_uint8()));
      } else if (r_arg.type().is_uint(16)) {
        call_args.push_back(b_->getInt16(r_arg.as_uint16()));
      } else if (r_arg.type().is_uint(32)) {
        call_args.push_back(b_->getInt32(r_arg.as_uint32()));
      } else if (r_arg.type().is_uint(64)) {
        call_args.push_back(b_->getInt64(r_arg.as_uint64()));
      } else if (r_arg.type().is_float(32)) {
        call_args.push_back(llvm::ConstantFP::get(
            b_->getFloatTy(), llvm::APFloat(r_arg.as_float())));
      } else if (r_arg.type().is_float(64)) {
        call_args.push_back(llvm::ConstantFP::get(
            b_->getDoubleTy(), llvm::APFloat(r_arg.as_double())));
      } else if (r_arg.type().is_bfloat16()) {
        call_args.push_back(llvm::ConstantFP::get(
            b_->getBFloatTy(),
            llvm::APFloat(static_cast<float>(r_arg.as_bfloat16()))));
      } else if (r_arg.type().is_float16()) {
        call_args.push_back(llvm::ConstantFP::get(
            b_->getHalfTy(),
            llvm::APFloat(static_cast<float>(r_arg.as_float16()))));
      } else {
        CINN_NOT_IMPLEMENTED;
      }
    }
  }

  b_->CreateCall(call_func, call_args);
  RetVoid();

  return function;
}

}  // namespace backends
}  // namespace cinn
