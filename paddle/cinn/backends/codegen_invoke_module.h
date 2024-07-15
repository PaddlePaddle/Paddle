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

#pragma once

#include <memory>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

/**
 * CINN jit instructions support two kinds of invoke function, which can be
 * represented like this:
 * InvokeFunc = HostFunc | SwitchHostFunc
 * HostFunc = X86Func | CudaHostFunc | HipHostFunc | ......
 * CodeGenInvokeModule takes a CINN invoke Module(a module that only contains
 * functions that jit instructions actually invoke) and output a LLVM module.
 */
class CodeGenInvokeModule : public CodeGenLLVM {
 public:
  explicit CodeGenInvokeModule(
      llvm::Module *m,
      llvm::IRBuilder<> *b,
      const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenLLVM(m, b, vars) {}

  using CodeGenLLVM::Visit;
  llvm::Value *Visit(const ir::_LoweredFunc_ *func) override {
    return LowerInvokeFunc(func);
  }

  llvm::Value *Visit(const ir::Call *op) override {
    // TODO(Hongqing-work): change intrinsic name to get_value_in_kernel_args
    if (op->name == runtime::intrinsic::get_value_in_cuda_kernel_args) {
      return LowerParseArgsValueCall(op);
    } else {
      return CodeGenLLVM::Visit(op);
    }
  }

 protected:
  llvm::Value *LowerInvokeFunc(const ir::_LoweredFunc_ *func);

  llvm::Value *LowerParseArgsValueCall(const ir::Call *call_ir);
};

class CodeGenHost : public CodeGenInvokeModule {
 public:
  explicit CodeGenHost(llvm::Module *m,
                       llvm::IRBuilder<> *b,
                       const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenInvokeModule(m, b, vars) {}
};

/**
 * In the SwitchHostFunc pattern, InvokeFunc is a switch statement where
 * every case is a call of HostFunc. All the callee functions have the same
 * parameters with the caller function.
 */
class CodeGenSwitchHost : public CodeGenInvokeModule {
 public:
  explicit CodeGenSwitchHost(llvm::Module *m,
                             llvm::IRBuilder<> *b,
                             const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenInvokeModule(m, b, vars) {}
  // only support call of args get function and inner case host function call
  llvm::Value *Visit(const ir::Call *op) override {
    if (op->name == runtime::intrinsic::get_value_in_cuda_kernel_args) {
      return CodeGenInvokeModule::LowerParseArgsValueCall(op);
    } else {
      return LowerInnerCaseCall(op);
    }
  }

 private:
  llvm::Value *LowerInnerCaseCall(const ir::Call *op);
};

}  // namespace backends
}  // namespace cinn
