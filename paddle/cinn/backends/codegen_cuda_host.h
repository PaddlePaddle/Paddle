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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/runtime/intrinsic.h"

PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace backends {

/**
 * CodeGenCUDA takes a CINN Module with host functions and output a LLVM module.
 */
class CodeGenCUDA_Host : public CodeGenLLVM {
 public:
  explicit CodeGenCUDA_Host(llvm::Module *m,
                            llvm::IRBuilder<> *b,
                            const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenLLVM(m, b, vars) {}

  using CodeGenLLVM::Visit;
  llvm::Value *Visit(const ir::_LoweredFunc_ *func) override {
    if (FLAGS_cinn_bucket_compile) {
      return LowerHostFunc(func);
    }
    return LowerGPUKernelLauncher(func);
  }

  llvm::Value *Visit(const ir::Call *op) override {
    if (op->name == runtime::intrinsic::get_value_in_cuda_kernel_args) {
      return LowerParseArgsValueCall(op);
    } else if (op->name == runtime::intrinsic::call_cuda_kernel) {
      return LowerCUDAKernelCall(op);
    } else if (op->name.find("_leaf") != std::string::npos) {
      return LowerWrappedCall(op);
    } else {
      return CodeGenLLVM::Visit(op);
    }
  }

 private:
  /**
   * Lower a CUDA kernel launcher.
   *
   * We launch a CUDA kernel in the following way:
   *
   * 1. a GPU function (called fn) will compiled to PTX and lower by CUDA driver
   * to a function pointer, which we store as a `void*` type global variable
   * [fn_kernel_ptr] in LLVM module.
   * 2. when lower the host launcher, we replace the Call of the original kernel
   * [fn] to a Call of `cinn_call_cuda_kernel` method which is registered as an
   * external function.
   *
   */
  llvm::Value *LowerGPUKernelLauncher(const ir::_LoweredFunc_ *func);

  llvm::Value *LowerHostFunc(const ir::_LoweredFunc_ *func);

  llvm::Value *LowerParseArgsValueCall(const ir::Call *call_ir);

  llvm::Value *LowerCUDAKernelCall(const ir::Call *op);

  llvm::Value *LowerWrappedCall(const ir::Call *op);
};

}  // namespace backends
}  // namespace cinn
