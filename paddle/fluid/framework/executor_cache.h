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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

class ExecutorInfoCache {
 public:
  /*
   * The ExecutorPrepareContext is different while running forward program and
   * backward program. We add bool value into cached key to distinguish this.
   */
  using KeyType = std::pair<const framework::ProgramDesc*, /*is_grad*/ bool>;

  static ExecutorInfoCache& Instance();

  std::shared_ptr<framework::ExecutorPrepareContext> Get(
      const KeyType& key) const {
    PADDLE_ENFORCE_EQ(
        Has(key), true,
        platform::errors::NotFound(
            "(programDesc: %s, is_grad: %s) doesn't exist in ExecutorInfoCache",
            key.first, key.second));
    return info_map_.at(key);
  }

  bool Has(const KeyType& key) const {
    return info_map_.find(key) != info_map_.end();
  }

  void Insert(const KeyType& key,
              std::shared_ptr<framework::ExecutorPrepareContext> exe_ctx) {
    PADDLE_ENFORCE_NE(
        Has(key), true,
        platform::errors::NotFound(
            "(programDesc: %s, is_grad: %s) has existed in ExecutorInfoCache",
            key.first, key.second));

    info_map_.insert(std::make_pair(key, exe_ctx));
  }

 private:
  struct HashPair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const noexcept {
      auto hash1 = std::hash<T1>{}(p.first);
      auto hash2 = std::hash<T2>{}(p.second);
      return hash1 ^ hash2;
    }
  };
  ExecutorInfoCache() = default;

  std::unordered_map<
      KeyType, std::shared_ptr<framework::ExecutorPrepareContext>, HashPair>
      info_map_;
  DISABLE_COPY_AND_ASSIGN(ExecutorInfoCache);
};

}  // namespace framework
}  // namespace paddle
