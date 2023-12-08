// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"

namespace cinn {
namespace hlir {
namespace framework {

class GroupCompilationContext {
 public:
  GroupCompilationContext(const Target& target,
                          const pir::GroupPtr& group,
                          std::shared_ptr<Scope> scope)
      : target_(target), group_(group), scope_(scope) {}

  void SetLoweredFuncs(
      std::vector<std::pair<ir::SymbolicPredicate, ir::LoweredFunc>>&& funcs);
  std::string PrintPredicate2Funcs() const;
  void* FuncPtr();
  std::shared_ptr<backends::Compiler> BackendCompiler();

 private:
  friend class CompilationTask;

  const Target& target_;
  const pir::GroupPtr& group_;
  std::shared_ptr<Scope> scope_;

  size_t func_size_ = 0;
  std::vector<ir::SymbolicPredicate> predicates_;
  std::vector<ir::LoweredFunc> lowered_funcs_;
  std::string host_func_name_;
  std::string host_code_;
  std::vector<std::string> device_code_;
  std::shared_ptr<backends::Compiler> backend_compiler_;
};

class CompilationTask {
 public:
  explicit CompilationTask(GroupCompilationContext* context)
      : context_(context) {}

  void operator()();

  void Lowering();
  void CodegenAndJit();
  std::unique_ptr<Instruction> BuildInstruction();
  pir::CINNKernelInfo BuildPirCINNKernelInfo();

 private:
  GroupCompilationContext* context_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
