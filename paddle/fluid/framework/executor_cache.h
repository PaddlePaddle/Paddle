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

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}

namespace details {
void AppendSkipDeletionVars(const std::vector<std::string>& append_vars,
                            std::vector<std::string>* all_vars);

void ParseSafeEagerDeletionSkipVars(
    const ProgramDesc& program, int64_t forward_op_nums,
    const std::vector<std::string>& output_var_names,
    std::vector<std::string>* skip_eager_delete_vars);

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
        strategy_map_.count(program_id), 0,
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

  void UpdateSkipEagerDeleteVars(int64_t program_id, bool is_grad,
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

CacheInfo GetExecutorInfoFromCache(const ProgramDesc& program_desc,
                                   const platform::Place& place,
                                   int64_t start_op_index, int64_t end_op_index,
                                   bool is_grad, int64_t program_id,
                                   framework::Scope* scope);

}  // namespace framework
}  // namespace paddle
