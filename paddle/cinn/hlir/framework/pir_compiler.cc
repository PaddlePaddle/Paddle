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

#include "paddle/cinn/hlir/framework/pir_compiler.h"

#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/common/enforce.h"

namespace cinn::hlir::framework {

class CompilationContextMapper {
 public:
  CompilationContextMapper(const Target& target,
                           const std::vector<pir::OpLoweringGroupPtr>& groups)
      : groups_(groups) {
    Construct(target);
  }
  std::vector<GroupCompilationContext>& UniqueCompilationContexts() {
    return group_compilation_contexts_;
  }
  std::vector<std::shared_ptr<pir::CompilationResult>>&
  MutableCompilationResult() {
    return compilation_results_;
  }

  std::vector<pir::CINNKernelInfo> RecoverKernelInfos();
  void UpdateGlobalCache();
  void SetFinalize(bool val) { is_finalized_ = val; }

 private:
  void Construct(const Target& target);

  const std::vector<pir::OpLoweringGroupPtr>& groups_;
  std::vector<size_t> mapper_index_;
  std::vector<GroupCompilationContext> group_compilation_contexts_;
  std::vector<std::shared_ptr<pir::CompilationResult>> compilation_results_;

  bool is_finalized_{false};
};

std::vector<pir::CINNKernelInfo> PirCompiler::Build(
    const std::vector<pir::OpLoweringGroupPtr>& groups) {
  CompilationContextMapper ctx_mapper(target_, groups);
  auto& group_compilation_contexts = ctx_mapper.UniqueCompilationContexts();
  auto& compilation_results = ctx_mapper.MutableCompilationResult();

  auto worker_fn = [&](int index) {
    CompilationTask task(&group_compilation_contexts[index]);
    compilation_results[index] = task();
  };
  utils::parallel_run(
      worker_fn, utils::SequenceDispatcher(0, groups.size()), -1);
  ctx_mapper.SetFinalize(true);
  ctx_mapper.UpdateGlobalCache();

  return ctx_mapper.RecoverKernelInfos();
}

void CompilationContextMapper::Construct(const Target& target) {
  // Step 1: Generate unqiue group_compilation_contexts_;
  std::unordered_map<pir::FusionInfo, size_t> unique_infos;
  for (size_t i = 0; i < groups_.size(); ++i) {
    GroupCompilationContext ctx(target, groups_[i]);
    if (unique_infos.find(ctx.FusionInfo()) == unique_infos.end()) {
      unique_infos[ctx.FusionInfo()] = i;
      group_compilation_contexts_.push_back(ctx);
    }
    mapper_index_.push_back(unique_infos[ctx.FusionInfo()]);
  }

  // Step 2: Generate empty compilation_results_;
  for (size_t i = 0; i < group_compilation_contexts_.size(); ++i) {
    compilation_results_.push_back(
        std::make_shared<pir::CompilationResult>(target));
  }
}

std::vector<pir::CINNKernelInfo>
CompilationContextMapper::RecoverKernelInfos() {
  PADDLE_ENFORCE_EQ(
      is_finalized_,
      true,
      ::common::errors::PreconditionNotMet(
          "Required is_finalized_ = true, please call SetFinalize() firstly."));
  PADDLE_ENFORCE_EQ(
      groups_.size(),
      compilation_results_.size(),
      ::common::errors::PreconditionNotMet(
          "Required groups_.size() = compilation_results_.size()."));

  std::vector<pir::CINNKernelInfo> kernel_infos(groups_.size());
  for (size_t i = 0; i < mapper_index_.size(); ++i) {
    size_t index = mapper_index_[i];
    kernel_infos[i] = compilation_results_[index]->GetKernelInfo();
  }
  return kernel_infos;
}

void CompilationContextMapper::UpdateGlobalCache() {
  PADDLE_ENFORCE_EQ(
      is_finalized_,
      true,
      ::common::errors::PreconditionNotMet(
          "Required is_finalized_ = true, please call SetFinalize() firstly."));
  for (size_t i = 0; i < compilation_results_.size(); ++i) {
    const auto& fusion_info = group_compilation_contexts_[i].FusionInfo();
    CompilationCache::Instance().Insert(fusion_info, compilation_results_[i]);
  }
}
}  // namespace cinn::hlir::framework
