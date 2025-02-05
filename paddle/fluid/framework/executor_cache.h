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
           const int64_t& place_hash_key,
           bool is_grad,
           bool in_pir_mode) {
    if (in_pir_mode) {
      int64_t scope_i = reinterpret_cast<int64_t>(scope);
      program_id = hash_with_seed(program_id, scope_i);
      program_id = hash_with_seed(program_id, place_hash_key);
    }
    return info_map_.find(program_id) != info_map_.end() &&
           info_map_[program_id].IsAvailable(is_grad);
  }

  InterpreterCoreInfo::CacheValue& GetMutable(int64_t program_id,
                                              const framework::Scope* scope,
                                              const int64_t& place_hash_key,
                                              bool is_grad,
                                              bool in_pir_mode) {
    if (in_pir_mode) {
      int64_t scope_i = reinterpret_cast<int64_t>(scope);
      program_id = hash_with_seed(program_id, scope_i);
      program_id = hash_with_seed(program_id, place_hash_key);
    }
    return info_map_[program_id].GetMutable(is_grad);
  }

  void UpdateSkipEagerDeleteVars(int64_t program_id,
                                 const framework::Scope* scope,
                                 const int64_t& place_hash_key,
                                 bool is_grad,
                                 bool in_pir_mode,
                                 const std::set<std::string>& skip_vars) {
    auto& cached_value =
        GetMutable(program_id, scope, place_hash_key, is_grad, in_pir_mode);
    cached_value.skip_eager_delete_vars_ = std::move(skip_vars);
  }

  std::set<std::string>& GetSkipEagerDeleteVars(int64_t program_id,
                                                const framework::Scope* scope,
                                                const int64_t& place_hash_key,
                                                bool in_pir_mode,
                                                bool is_grad) {
    auto& cached_value =
        GetMutable(program_id, scope, place_hash_key, is_grad, in_pir_mode);
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
    bool is_grad,
    int64_t program_id,
    framework::Scope* scope,
    const int64_t& place_hash_key);

std::shared_ptr<InterpreterCore> CreatePirInterpreterCoreInfoToCache(
    std::unique_ptr<::pir::Program> ir_prog,
    const phi::Place& place,
    bool is_grad,
    int64_t program_id,
    framework::Scope* scope,
    const int64_t& place_hash_key,
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
