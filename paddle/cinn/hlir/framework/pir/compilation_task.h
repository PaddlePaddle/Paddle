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
class LoweringTask;
class CompilationTask;
class GroupCompilationContext;

class GroupCompilationContext {
 public:
  GroupCompilationContext(const Target& target,
                          const pir::OpLoweringGroupPtr& group)
      : target_(target), group_(group) {}

  void SetLoweredFuncs(BucketLoweredFuncsWrapper&& funcs);
  std::string PrintPredicate2Funcs() const;

 private:
  friend class LoweringTask;
  friend class CompilationTask;
  friend GroupCompilationContext ContextReduction(
      std::vector<GroupCompilationContext>* contexts);
  const Target& target_;
  const pir::OpLoweringGroupPtr& group_;
  std::vector<ir::SymbolicPredicate> broadcast_predicates_;
  std::vector<ir::SymbolicPredicate> bucket_predicates_;
  std::vector<int> priorities_;
  std::vector<ir::LoweredFunc> lowered_funcs_;
  std::vector<ir::SymbolicPredicate> CX86_bucket_predicates_;
  std::vector<ir::LoweredFunc> CX86_lowered_funcs_;
  std::vector<ir::LoweredFunc> infer_shape_lowered_funcs_;
};

class LoweringTask {
 public:
  explicit LoweringTask(GroupCompilationContext* context) : context_(context) {}

  void operator()();

 private:
  void Lowering();
  GroupCompilationContext* context_;
};

class CompilationTask {
 public:
  explicit CompilationTask(GroupCompilationContext* context)
      : context_(context) {}

  std::shared_ptr<pir::CompilationResult> operator()();

 private:
  std::shared_ptr<pir::CompilationResult> CodegenAndJit();
  std::shared_ptr<pir::CompilationResult> BuildPirCINNKernelInfo(
      const ir::Module& module, const ir::Module& CX86module);

  GroupCompilationContext* context_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
