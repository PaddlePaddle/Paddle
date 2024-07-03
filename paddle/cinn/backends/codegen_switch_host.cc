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

#include "paddle/cinn/backends/codegen_switch_host.h"

#include <vector>
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {
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
