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
#include "paddle/cinn/hlir/framework/pir/compilation_cache.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace hlir {
namespace framework {
class CompilationTask;

class GroupCompilationContext {
 public:
  GroupCompilationContext(const Target& target,
                          const pir::OpLoweringGroupPtr& group)
      : target_(target),
        group_(group),
        module_builder_(cinn::common::UniqName("module"), target),
        CX86_module_builder_(cinn::common::UniqName("module"),
                             common::DefaultHostTarget()) {}

  const pir::OpLoweringGroupPtr& GetGroup() const { return group_; }
  void SetLoweredFuncs(BucketLoweredFuncsWrapper&& funcs);
  void PrepareModuleBuilder();
  std::string PrintPredicate2Funcs() const;
  bool NeedCompileCX86Kernel() const { return need_x86_kernel_; }

 private:
  friend class CompilationTask;
  friend void UnifyBroadcastGroupFuncArgs(
      std::vector<GroupCompilationContext>* contexts,
      pir::OpLoweringGroupPtr origin_group,
      std::unordered_map<int, ir::Var>* symbolic_shape_var_index);
  const Target& target_;
  const pir::OpLoweringGroupPtr& group_;
  ir::SymbolicPredicate broadcast_condition_;
  std::vector<ir::SymbolicPredicate> predicates_;
  std::vector<int> priorities_;
  std::vector<ir::LoweredFunc> lowered_funcs_;
  std::vector<ir::SymbolicPredicate> CX86_predicates_;
  std::vector<ir::LoweredFunc> CX86_lowered_funcs_;
  ir::LoweredFunc infer_shape_lowered_func_;
  ir::Module::Builder module_builder_;
  ir::Module::Builder CX86_module_builder_;
  bool need_x86_kernel_{false};
};

class CompilationTask {
 public:
  explicit CompilationTask(GroupCompilationContext* context)
      : context_(context) {}

  std::shared_ptr<pir::CompilationResult> operator()();
  void Lowering();
  std::shared_ptr<pir::CompilationResult> CompileBroadcastModules(
      std::vector<GroupCompilationContext>* leaf_group_contexts,
      const std::unordered_map<int, ir::Var>& symbolic_shape_var_index);

 private:
  std::shared_ptr<pir::CompilationResult> CodegenAndJit();
  std::shared_ptr<pir::CompilationResult> BuildPirCINNKernelInfo(
      const ir::Module& module,
      const ir::Module& CX86module,
      bool need_x86_kernel = false);

  GroupCompilationContext* context_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
