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

#include "paddle/cinn/backends/codegen_host_base.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

/**
 * CodeGenSwitchHost takes a CINN Module with switch<NaiveHostFunction>
 * functions and output a LLVM module.
 */
class CodeGenSwitchHost : public CodeGenHostBase {
 public:
  explicit CodeGenSwitchHost(llvm::Module *m,
                             llvm::IRBuilder<> *b,
                             const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenHostBase(m, b, vars) {}

  llvm::Value *Visit(const ir::Call *op) override {
    if (op->name == runtime::intrinsic::get_value_in_cuda_kernel_args) {
      return CodeGenHostBase::LowerParseArgsValueCall(op);
    } else {
      return LowerInnerCaseCall(op);
    }
  }

 private:
  llvm::Value *LowerInnerCaseCall(const ir::Call *op);
};

}  // namespace backends
}  // namespace cinn
