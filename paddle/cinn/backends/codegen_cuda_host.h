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
#include <memory>

#include "paddle/cinn/backends/codegen_invoke_module.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

/**
 * CodeGenGpuHost takes a CINN Module with CUDA/HIP host functions and output a
 * LLVM module.
 */
class CodeGenGpuHost : public CodeGenHost {
 public:
  explicit CodeGenGpuHost(llvm::Module *m,
                          llvm::IRBuilder<> *b,
                          const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenHost(m, b, vars) {}

  // TODO(Hongqing-work): remove this after we clear some old codes.
  llvm::Value *Visit(const ir::_LoweredFunc_ *func) {
    return CodeGenHost::Visit(func);
  }

  llvm::Value *Visit(const ir::Call *op) override {
    return common::DefaultDeviceTarget().arch.Match(
        [&](common::UnknownArch) { return CodeGenHost::Visit(op); },
        [&](common::X86Arch) { return CodeGenHost::Visit(op); },
        [&](common::ARMArch) { return CodeGenHost::Visit(op); },
        [&](common::NVGPUArch) {
          if (op->name == runtime::intrinsic::call_cuda_kernel) {
            return LowerGPUKernelCall(op);
          } else {
            return CodeGenHost::Visit(op);
          }
        },
        [&](common::HygonDCUArchHIP) {
          if (op->name == runtime::intrinsic::call_hip_kernel) {
            return LowerGPUKernelCall(op);
          } else {
            return CodeGenHost::Visit(op);
          }
        },
        [&](common::HygonDCUArchSYCL) {
          if (op->name == runtime::intrinsic::call_sycl_kernel) {
            return LowerGPUKernelCall(op);
          } else {
            return CodeGenHost::Visit(op);
          }
        });
  }

 private:
  llvm::Value *LowerGPUKernelCall(const ir::Call *op);
};

}  // namespace backends
}  // namespace cinn
