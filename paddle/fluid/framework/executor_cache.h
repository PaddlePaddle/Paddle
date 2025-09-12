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

#include "paddle/common/macros.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/utils/string/string_helper.h"

#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

COMMON_DECLARE_bool(enable_pir_in_executor);
COMMON_DECLARE_bool(enable_pir_with_pt_in_dy2st);

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}

class InterpreterCore;

namespace details {
void AppendSkipDeletionVars(const std::vector<std::string>& append_vars,
                            std::set<std::string>* all_vars);

// TODO(Aurelius84) : Need remove skip_no_need_buffer after cinn fix this
// problem.
std::set<std::string> ParseSafeEagerDeletionSkipVarsSet(
    const ProgramDesc& backward_program, bool skip_no_need_buffer = false);

}  // namespace details

int64_t hash_with_seed(int64_t value, int64_t seed);

class InterpreterCoreInfo {
 public:
  struct CacheValue {
    std::shared_ptr<InterpreterCore> core_{nullptr};
    std::set<std::string> skip_eager_delete_vars_;
    std::unique_ptr<::pir::Program> ir_prog_{nullptr};
  };

  bool IsAvailable(bool is_grad) const {
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

class InterpreterCoreInfoCacheKey {
 public:
  InterpreterCoreInfoCacheKey(int64_t program_id,
                              const framework::Scope* scope,
                              int64_t place_hash_key,
                              bool use_cuda_graph = false,
                              int64_t cuda_graph_dispatch_key = 0,
                              bool is_grad = false,
                              bool in_pir_mode = false)
      : program_id_(program_id),
        scope_(scope),
        place_hash_key_(place_hash_key),
        use_cuda_graph_(use_cuda_graph),
        cuda_graph_dispatch_key_(cuda_graph_dispatch_key),
        is_grad_(is_grad),
        in_pir_mode_(in_pir_mode) {}
  int64_t hash() const {
    int64_t hash_result = program_id_;
    if (in_pir_mode_) {
      int64_t scope_i = reinterpret_cast<int64_t>(scope_);
      hash_result = hash_with_seed(hash_result, scope_i);
      hash_result = hash_with_seed(hash_result, place_hash_key_);
      // CUDA Graph is available in pir mode only
      hash_result =
          hash_with_seed(hash_result, static_cast<int64_t>(use_cuda_graph_));
      hash_result = hash_with_seed(hash_result, cuda_graph_dispatch_key_);
    }
    return hash_result;
  }

  bool is_grad() const { return is_grad_; }

  InterpreterCoreInfoCacheKey with_pir_mode(bool in_pir_mode) const {
    // Create a new key with the specified PIR mode if the current key's
    // PIR mode is different from the specified one. Otherwise, return
    // the current key itself. This function is used to switch legacy IR
    // and PT mode.
    if (in_pir_mode == in_pir_mode_) {
      return *this;
    }
    return InterpreterCoreInfoCacheKey(program_id_,
                                       scope_,
                                       place_hash_key_,
                                       use_cuda_graph_,
                                       cuda_graph_dispatch_key_,
                                       is_grad_,
                                       in_pir_mode);
  }

 private:
  int64_t program_id_;
  const framework::Scope* scope_;
  int64_t place_hash_key_;
  bool use_cuda_graph_;
  int64_t cuda_graph_dispatch_key_;
  bool is_grad_;
  bool in_pir_mode_;
};

class InterpreterCoreInfoCache {
 public:
  static InterpreterCoreInfoCache& Instance();
  bool Has(const InterpreterCoreInfoCacheKey& key) const {
    int64_t hash_key = key.hash();
    return info_map_.find(hash_key) != info_map_.end() &&
           info_map_.at(hash_key).IsAvailable(key.is_grad());
  }

  InterpreterCoreInfo::CacheValue& GetMutable(
      const InterpreterCoreInfoCacheKey& key) {
    int64_t hash_key = key.hash();
    return info_map_[hash_key].GetMutable(key.is_grad());
  }

  void UpdateSkipEagerDeleteVars(const InterpreterCoreInfoCacheKey& key,
                                 const std::set<std::string>& skip_vars) {
    auto& cached_value = GetMutable(key);
    cached_value.skip_eager_delete_vars_ = std::move(skip_vars);
  }

  std::set<std::string>& GetSkipEagerDeleteVars(
      const InterpreterCoreInfoCacheKey& key) {
    auto& cached_value = GetMutable(key);
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
    const phi::Place& place,
    framework::Scope* scope,
    const InterpreterCoreInfoCacheKey& key);

std::shared_ptr<InterpreterCore> CreatePirInterpreterCoreInfoToCache(
    std::unique_ptr<::pir::Program> ir_prog,
    const phi::Place& place,
    framework::Scope* scope,
    const InterpreterCoreInfoCacheKey& key,
    bool used_for_sot);

std::unique_ptr<::pir::Program> ApplyIrPass(
    ::pir::Program* program,
    phi::Place place,
    const std::set<std::string>& no_need_buffer_names);

std::unique_ptr<::pir::Program> ApplyRemoveShadowFeedPass(
    const std::unique_ptr<::pir::Program> program,
    const pir::Block* block,
    const phi::Place& place,
    const paddle::framework::Scope* scope);

std::unique_ptr<::pir::Program> ConstructForwardIrProgram(
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
