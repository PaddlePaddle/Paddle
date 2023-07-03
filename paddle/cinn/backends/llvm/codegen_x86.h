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
#include <llvm/IR/IRBuilder.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"

namespace cinn::backends {

class CodeGenX86 : public CodeGenLLVM {
 public:
  explicit CodeGenX86(llvm::Module* m,
                      llvm::IRBuilder<>* b,
                      const std::shared_ptr<SymbolTable>& vars = nullptr);
  virtual ~CodeGenX86();

  using LLVMIRVisitor::Visit;

  llvm::Value* Visit(const ir::For* op);

 private:
  // parallel information
  struct ParallelEnv {
    Expr task_id;
    Expr num_task;
    bool stride_pattern{false};
    bool in_parallel_loop{false};
    int parallel_loop_count{0};
    llvm::Value* penv{nullptr};
  };

  llvm::Value* ParallelLaunch();
  // Create parallel launch
  void CreateParallelLaunch(Expr body, int num_task);

  llvm::Value* PackVars(const std::vector<std::string>& vars,
                        uint64_t* num_bytes);
  void UnpackVars(const std::vector<std::string>& vars, llvm::Value* data);
  llvm::BasicBlock* CheckCallSuccess(llvm::Value* retcode);
  // Current parallel environment scope.
  ParallelEnv parallel_env_;
};

}  // namespace cinn::backends
