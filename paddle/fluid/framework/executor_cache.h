// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

PHI_DECLARE_bool(enable_pir_in_executor);
PHI_DECLARE_bool(enable_pir_with_pt_in_dy2st);

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}

class InterpreterCore;

namespace details {
void AppendSkipDeletionVars(const std::vector<std::string>& append_vars,
                            std::vector<std::string>* all_vars);

void ParseSafeEagerDeletionSkipVars(
    const ProgramDesc& program,
    int64_t forward_op_nums,
    const std::vector<std::string>& output_var_names,
    std::vector<std::string>* skip_eager_delete_vars);

void AppendSkipDeletionVars(const std::vector<std::string>& append_vars,
                            std::set<std::string>* all_vars);

// TODO(Aurelius84) : Need remove skip_no_need_buffer after cinn fix this
// problem.
std::set<std::string> ParseSafeEagerDeletionSkipVarsSet(
    const ProgramDesc& backward_program, bool skip_no_need_buffer = false);

}  // namespace details

class ExecutorInfo {
 public:
  struct CacheValue {
    std::shared_ptr<ParallelExecutor> executor_{nullptr};
    std::shared_ptr<ir::Graph> graph_{nullptr};

    std::vector<std::string> skip_eager_delete_vars_;
  };

  bool IsAvailable(bool is_grad) {
    const auto& executor =
        is_grad ? backward_info_.executor_ : forward_info_.executor_;
    return executor != nullptr;
  }

  CacheValue& GetMutable(bool is_grad) {
    return is_grad ? backward_info_ : forward_info_;
  }

 private:
  CacheValue forward_info_;
  CacheValue backward_info_;
};

class ExecutorInfoCache {
 public:
  static ExecutorInfoCache& Instance();

  const BuildStrategy& GetBuildStrategy(int64_t program_id) {
    // If not found, insert build_strategy with default value.
    return strategy_map_[program_id];
  }

  void SetBuildStrategy(int64_t program_id,
                        const BuildStrategy& build_strategy) {
    PADDLE_ENFORCE_EQ(
        strategy_map_.count(program_id),
        0,
        platform::errors::PreconditionNotMet(
            "program_id: %s already exist in ExecutorInfoCache", program_id));
    strategy_map_[program_id] = build_strategy;
  }

  bool Has(int64_t program_id, bool is_grad) {
    return info_map_.find(program_id) != info_map_.end() &&
           info_map_[program_id].IsAvailable(is_grad);
  }

  ExecutorInfo::CacheValue& GetMutable(int64_t program_id, bool is_grad) {
    return info_map_[program_id].GetMutable(is_grad);
  }

  void UpdateSkipEagerDeleteVars(int64_t program_id,
                                 bool is_grad,
                                 const std::vector<std::string>& skip_vars) {
    auto& cached_value = GetMutable(program_id, is_grad);
    cached_value.skip_eager_delete_vars_ = std::move(skip_vars);
  }

  std::vector<std::string>& SkipEagerDeleteVars(int64_t program_id,
                                                bool is_grad) {
    auto& cached_value = GetMutable(program_id, is_grad);
    return cached_value.skip_eager_delete_vars_;
  }

  size_t Size() const { return info_map_.size(); }

  void Finalize() {
    // NOTE(Aurelius84): DO NOT perform finalize in destructor
    // to avoid problems caused by destructor order of static
    // object.
    info_map_.clear();
    strategy_map_.clear();
  }

 private:
  std::unordered_map<int64_t, ExecutorInfo> info_map_;
  std::unordered_map<int64_t, BuildStrategy> strategy_map_;
};

using CacheInfo =
    std::pair<std::shared_ptr<ParallelExecutor>, bool /*is_new_created*/>;

using PEAndGraphPair =
    std::pair<std::shared_ptr<ParallelExecutor>, std::shared_ptr<ir::Graph>>;

CacheInfo GetExecutorInfoFromCache(const ProgramDesc& program_desc,
                                   const platform::Place& place,
                                   int64_t start_op_index,
                                   int64_t end_op_index,
                                   bool is_grad,
                                   int64_t program_id,
                                   framework::Scope* scope);

PEAndGraphPair CreateFixOrderExecutorInfo(const ProgramDesc& program_desc,
                                          const platform::Place& place,
                                          int64_t start_op_index,
                                          int64_t end_op_index,
                                          framework::Scope* scope);

int64_t hash_with_seed(int64_t value, int64_t seed);

class InterpreterCoreInfo {
 public:
  struct CacheValue {
    std::shared_ptr<InterpreterCore> core_{nullptr};
    std::set<std::string> skip_eager_delete_vars_;
    std::unique_ptr<::pir::Program> ir_prog_{nullptr};
  };

  bool IsAvailable(bool is_grad) {
    const auto& core = is_grad ? backward_info_.core_ : forward_info_.core_;
    return core != nullptr;
  }

  CacheValue& GetMutable(bool is_grad) {
    return is_grad ? backward_info_ : forward_info_;
  }

 private:
  CacheValue forward_info_;
  CacheValue backward_info_;
};

class InterpreterCoreInfoCache {
 public:
  static InterpreterCoreInfoCache& Instance();

  bool Has(int64_t program_id,
           const framework::Scope* scope,
           const std::vector<int64_t>& seeds,
           bool is_grad) {
    if (FLAGS_enable_pir_in_executor || FLAGS_enable_pir_with_pt_in_dy2st) {
      int64_t scope_i = reinterpret_cast<int64_t>(scope);
      program_id = hash_with_seed(program_id, scope_i);
      for (int64_t seed : seeds) {
        program_id = hash_with_seed(program_id, seed);
      }
    }
    return info_map_.find(program_id) != info_map_.end() &&
           info_map_[program_id].IsAvailable(is_grad);
  }

  InterpreterCoreInfo::CacheValue& GetMutable(int64_t program_id,
                                              const framework::Scope* scope,
                                              const std::vector<int64_t>& seeds,
                                              bool is_grad) {
    if (FLAGS_enable_pir_in_executor || FLAGS_enable_pir_with_pt_in_dy2st) {
      int64_t scope_i = reinterpret_cast<int64_t>(scope);
      program_id = hash_with_seed(program_id, scope_i);
      for (int64_t seed : seeds) {
        program_id = hash_with_seed(program_id, seed);
      }
    }
    return info_map_[program_id].GetMutable(is_grad);
  }

  void UpdateSkipEagerDeleteVars(int64_t program_id,
                                 const framework::Scope* scope,
                                 const std::vector<int64_t>& seeds,
                                 bool is_grad,
                                 const std::set<std::string>& skip_vars) {
    auto& cached_value = GetMutable(program_id, scope, seeds, is_grad);
    cached_value.skip_eager_delete_vars_ = std::move(skip_vars);
  }

  std::set<std::string>& GetSkipEagerDeleteVars(
      int64_t program_id,
      const framework::Scope* scope,
      const std::vector<int64_t>& seeds,
      bool is_grad) {
    auto& cached_value = GetMutable(program_id, scope, seeds, is_grad);
    return cached_value.skip_eager_delete_vars_;
  }

  size_t Size() const { return info_map_.size(); }

  void Finalize() {
    // NOTE(Aurelius84): DO NOT perform finalize in destructor
    // to avoid problems caused by destructor order of static
    // object.
    info_map_.clear();
  }

 private:
  std::unordered_map<int64_t, InterpreterCoreInfo> info_map_;
};

std::shared_ptr<InterpreterCore> CreateProgramInterpreterCoreInfoToCache(
    const ProgramDesc& program_desc,
    const platform::Place& place,
    bool is_grad,
    int64_t program_id,
    framework::Scope* scope,
    const std::vector<int64_t>& seeds);

std::shared_ptr<InterpreterCore> CreatePirInterpreterCoreInfoToCache(
    std::unique_ptr<::pir::Program> ir_prog,
    const platform::Place& place,
    bool is_grad,
    int64_t program_id,
    framework::Scope* scope,
    const std::vector<int64_t>& seeds);

std::unique_ptr<::pir::Program> ApplyIrPass(::pir::Program* program,
                                            phi::Place place);

std::unique_ptr<::pir::Program> ConstructFowardIrProgram(
    const paddle::framework::BlockDesc* forward_global_block,
    const paddle::framework::BlockDesc* backward_global_block,
    const std::vector<std::string>& output_names,
    const std::vector<paddle::Tensor>& x,
    const std::vector<std::string>& x_names,
    const std::vector<paddle::Tensor>& params,
    const phi::Place& place);

std::unique_ptr<::pir::Program> ConstructBackwardIrProgram(
    const paddle::framework::BlockDesc* backward_global_block,
    const std::vector<paddle::Tensor>& out_grad,
    const std::vector<paddle::Tensor*>& x_grad,
    const std::vector<paddle::Tensor*>& params_grad,
    const paddle::framework::Scope* scope,
    const phi::Place& place);

}  // namespace framework
}  // namespace paddle
