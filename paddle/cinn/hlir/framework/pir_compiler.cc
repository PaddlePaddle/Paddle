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
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"

#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/arch_device.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"

PD_DECLARE_bool(enable_cinn_compile_cache);
PD_DECLARE_int64(cinn_compile_thread_num);

namespace cinn::hlir::framework {
class CompilationContextMapper {
 public:
  CompilationContextMapper(const Target& target,
                           const std::vector<pir::OpLoweringGroupPtr>& groups) {
    Construct(target, groups);
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
  void Construct(const Target& target,
                 const std::vector<pir::OpLoweringGroupPtr>& groups);
  std::vector<size_t> mapper_index_;
  std::vector<pir::FusionInfo> fusion_infos_;
  std::vector<GroupCompilationContext> group_compilation_contexts_;
  std::vector<std::shared_ptr<pir::CompilationResult>> compilation_results_;

  bool is_finalized_{false};
};

static size_t GetThreadNum(size_t task_size) {
  size_t thread_size = task_size;
  if (!FLAGS_enable_cinn_compile_cache) {
    thread_size = 1;
  } else if (FLAGS_cinn_compile_thread_num > 0) {
    thread_size = FLAGS_cinn_compile_thread_num;
  }
  return thread_size;
}

std::vector<pir::CINNKernelInfo> PirCompiler::Build(
    const std::vector<pir::OpLoweringGroupPtr>& groups) {
  CompilationContextMapper ctx_mapper(target_, groups);
  auto& group_compilation_contexts = ctx_mapper.UniqueCompilationContexts();
  auto& compilation_results = ctx_mapper.MutableCompilationResult();

  const size_t task_size = group_compilation_contexts.size();
  const size_t thread_size = GetThreadNum(task_size);
  VLOG(5) << "Found " << task_size << " new groups parsed from "
          << groups.size() << " and compiles with " << thread_size;
  cinn::ir::InitScheduleConfig();
  if (task_size > 0) {
    // See
    // https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
    // for details.
    const auto device_id = runtime::GetArchDevice(target_);
    auto worker_fn = [&](int index) {
      runtime::SetArchDevice(target_, device_id);
      CompilationTask task(&group_compilation_contexts[index]);
      compilation_results[index] = task();
      // Triggering llvm compilation in thread
      compilation_results[index]->GetKernelInfo();
    };
    utils::parallel_run(worker_fn,
                        utils::SequenceDispatcher(0, task_size),
                        /*thread_num=*/thread_size);
  }
  VLOG(5) << "Finished compiling " << task_size << " Cinn Kernel info.";
  ctx_mapper.SetFinalize(true);
  ctx_mapper.UpdateGlobalCache();
  return ctx_mapper.RecoverKernelInfos();
}

void CompilationContextMapper::Construct(
    const Target& target, const std::vector<pir::OpLoweringGroupPtr>& groups) {
  std::unordered_set<size_t> unique_infos;
  const auto IsNewAndUnique =
      [&unique_infos](const pir::FusionInfo& info) -> bool {
    const bool is_unique = unique_infos.find(info.hash()) == unique_infos.end();
    const bool is_new = !CompilationCache::Instance().Has(info);
    return is_new && is_unique;
  };

  for (size_t i = 0; i < groups.size(); ++i) {
    fusion_infos_.emplace_back(*groups[i]);
    VLOG(5) << "Construct FusionInfo: " << fusion_infos_[i]
            << " for group: " << *groups[i];
    // If FLAGS_enable_cinn_compile_cache=False, Cache strategy will not take
    // effects.
    if (IsNewAndUnique(fusion_infos_[i]) || !FLAGS_enable_cinn_compile_cache) {
      mapper_index_.push_back(i);
      group_compilation_contexts_.emplace_back(target, groups[i]);
      compilation_results_.push_back(
          std::make_shared<pir::CompilationResult>(target));
    }
    unique_infos.insert(fusion_infos_[i].hash());
  }
}

std::vector<pir::CINNKernelInfo>
CompilationContextMapper::RecoverKernelInfos() {
  PADDLE_ENFORCE_EQ(
      is_finalized_,
      true,
      ::common::errors::PreconditionNotMet(
          "Required is_finalized_ = true, please call SetFinalize() firstly."));
  PADDLE_ENFORCE_EQ(group_compilation_contexts_.size(),
                    compilation_results_.size(),
                    ::common::errors::PreconditionNotMet(
                        "Required group_compilation_contexts_.size() = "
                        "compilation_results_.size()."));

  std::vector<pir::CINNKernelInfo> kernel_infos(fusion_infos_.size());
  for (size_t i = 0; i < fusion_infos_.size(); ++i) {
    const auto& compilation_result =
        FLAGS_enable_cinn_compile_cache
            ? CompilationCache::Instance().Get(fusion_infos_[i])
            : compilation_results_[i];
    kernel_infos[i] = compilation_result->GetKernelInfo();
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
    PADDLE_ENFORCE_LT(mapper_index_[i],
                      fusion_infos_.size(),
                      ::common::errors::PreconditionNotMet(
                          "Required mapper_index < fusion_infos_.size()."));
    const auto& fusion_info = fusion_infos_[mapper_index_[i]];
    VLOG(5) << "Insert new compiled result into cache, fusion_info: "
            << fusion_info;
    CompilationCache::Instance().Insert(fusion_info, compilation_results_[i]);
  }
}
}  // namespace cinn::hlir::framework
