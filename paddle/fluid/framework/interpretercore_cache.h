// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpretercore.h"

namespace paddle {
namespace framework {

class InterpreterCoreInfo {
 public:
  struct CacheValue {
    std::shared_ptr<InterpreterCore> core_{nullptr};
    std::shared_ptr<ProgramDesc> program_{nullptr};
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

  bool Has(int64_t program_id, bool is_grad) {
    return info_map_.find(program_id) != info_map_.end() &&
           info_map_[program_id].IsAvailable(is_grad);
  }

  InterpreterCoreInfo::CacheValue& GetMutable(int64_t program_id,
                                              bool is_grad) {
    return info_map_[program_id].GetMutable(is_grad);
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

using CacheInfo =
    std::pair<std::shared_ptr<InterpreterCore>, bool /*is_new_created*/>;

CacheInfo GetInterpreterCoreInfoFromCache(const ProgramDesc& program_desc,
                                          const platform::Place& place,
                                          bool is_grad,
                                          int64_t program_id,
                                          framework::Scope* scope);

}  // namespace framework
}  // namespace paddle
