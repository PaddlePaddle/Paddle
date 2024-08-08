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

#include "paddle/cinn/backends/extern_func_emitter_builtin.h"

#include <glog/logging.h>

#include "paddle/cinn/backends/llvm/ir_builder_mixin.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {

void ExternFunctionLLVMEmitter::BindCodeGen(void* codegen) {
  codegen_ = reinterpret_cast<CodeGenLLVM*>(codegen);
}

const char* ExternFunctionLLVMEmitter::func_name() const {
  return fn_name_.c_str();
}

bool ExternFunctionLLVMEmitter::RetValuePacked() const {
  return fn_proto().ret_type.is_void();
}

FunctionProto& ExternFunctionLLVMEmitter::fn_proto() const {
  auto* proto = ExternFunctionProtoRegistry::Global().Lookup(fn_name_);
  PADDLE_ENFORCE_NOT_NULL(proto,
                          ::common::errors::InvalidArgument(
                              "No function prototype found for %s.", fn_name_));
  return *proto;
}
llvm::FunctionType* ExternFunctionLLVMEmitter::llvm_fn_type() const {
  auto* proto = ExternFunctionProtoRegistry::Global().Lookup(fn_name_);
  PADDLE_ENFORCE_NOT_NULL(proto,
                          ::common::errors::InvalidArgument(
                              "No function prototype found for %s.", fn_name_));

  auto* llvm_ret_type = CinnTypeToLLVMType(proto->ret_type, codegen_->m());
  std::vector<llvm::Type*> arg_types;
  for (auto& t : proto->readonly_arg_types) {
    arg_types.push_back(CinnTypeToLLVMType(t, codegen_->m()));
  }
  for (auto& t : proto->mutable_arg_types) {
    arg_types.push_back(CinnTypeToLLVMType(t, codegen_->m()));
  }
  auto* fn_type = llvm::FunctionType::get(llvm_ret_type, arg_types, false);
  return fn_type;
}

const char* ExternFunctionLLVMEmitter::backend_kind() const { return nullptr; }

void ExternFunctionLLVMEmitter::EmitImpl(const ir::Call* op) {
  PADDLE_ENFORCE_NOT_NULL(
      codegen_,
      ::common::errors::InvalidArgument("Code not generate, please check."));
  CodeGenLLVMforEmitter codegen_for_emitter(codegen_);
  llvm::Function* custom_function = llvm::dyn_cast<llvm::Function>(
      codegen_for_emitter.m()
          ->getOrInsertFunction(fn_name_, llvm_fn_type())
          .getCallee());
  PADDLE_ENFORCE_NOT_NULL(
      custom_function,
      ::common::errors::InvalidArgument(
          "No function registered in JIT called %s.", fn_name_));
  custom_function->setCallingConv(llvm::CallingConv::C);

  std::vector<llvm::Value*> args;
  for (auto& v : op->read_args) {
    if (v.as_tensor()) {
      args.push_back(
          codegen_for_emitter.GetVar(v.as_tensor()->buffer->name, false));
    } else {
      auto* arg = codegen_for_emitter.Visit(&v);
      args.push_back(arg);
    }
  }
  for (auto& v : op->write_args) {
    if (v.as_tensor()) {
      args.push_back(
          codegen_for_emitter.GetVar(v.as_tensor()->buffer->name, false));
    } else {
      auto* arg = codegen_->Visit(&v);
      args.push_back(arg);
    }
  }

  VLOG(3) << "function type " << op->name << ": "
          << DumpToString(*custom_function);

  auto* command = codegen_for_emitter.b()->CreateCall(custom_function, args);
  codegen_->extern_func_emit_res_ = command;
  VLOG(3) << "call: " << DumpToString(*command);
}

}  // namespace backends
}  // namespace cinn
