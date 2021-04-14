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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}

namespace details {
void AppendSkipDeletionVars(const std::vector<std::string>& append_vars,
                            std::vector<std::string>* all_vars);

std::vector<std::string> ParseSafeEagerDeletionSkipVars(
    const ProgramDesc& program, int64_t forward_op_nums,
    const std::vector<std::string>& output_var_names);

}  // namespace details
class ExecutorInfoCache {
 public:
  using KeyInfo = std::tuple<const ProgramDesc*, /*device_type*/ int,
                             /*start_op_index*/ int64_t,
                             /*end_op_index*/ int64_t, /*is_grad*/ bool>;
  using KeyType = size_t;

  using ValueType =
      std::pair<std::shared_ptr<ParallelExecutor>, std::shared_ptr<ir::Graph>>;

  struct HashTuple {
    size_t operator()(const KeyInfo& key) const noexcept {
      size_t seed = 10;
      auto* prog_desc = std::get<0>(key);
      /*
       * Note(Aurelius84): DO NOT use only ProgramDesc* to calculate hash value
       * because a new program will hold same pointer address after an older
       * program is destructed with a small probability. Add op size while
       * hashing because program may contains at least one block.
       */
      hash_combine(&seed, prog_desc);
      for (size_t i = 0; i < prog_desc->Size(); ++i) {
        hash_combine(&seed, &prog_desc->Block(i));
        hash_combine(&seed, prog_desc->Block(i).OpSize());
      }
      hash_combine(&seed, std::get<1>(key));
      hash_combine(&seed, std::get<2>(key));
      hash_combine(&seed, std::get<3>(key));
      hash_combine(&seed, std::get<4>(key));
      VLOG(1) << "hash value is : " << seed << " of pointer " << prog_desc;
      return seed;
    }

    template <typename T>
    void hash_combine(size_t* seed, const T& val) const {
      std::hash<T> hasher;
      (*seed) ^= hasher(val) + 0x9e3779b9 + ((*seed) << 6) + ((*seed >> 2));
    }
  };

  static ExecutorInfoCache& Instance();

  ValueType GetMutable(const KeyInfo& key) {
    KeyType key_value = key_hash_func_(key);
    PADDLE_ENFORCE_EQ(
        Has(key_value), true,
        platform::errors::NotFound(
            "(programDesc: %s, is_grad: %s) doesn't exist in ExecutorInfoCache",
            std::get<const ProgramDesc*>(key), std::get<bool>(key)));
    return info_map_[key_value];
  }

  bool Has(const KeyInfo& key) const {
    KeyType key_value = key_hash_func_(key);
    return Has(key_value);
  }

  bool Has(const KeyType& key) const {
    return info_map_.find(key) != info_map_.end();
  }

  void Insert(const KeyInfo& key, ValueType value) {
    KeyType key_value = key_hash_func_(key);
    PADDLE_ENFORCE_NE(
        Has(key_value), true,
        platform::errors::NotFound(
            "(programDesc: %s, is_grad: %s) has existed in ExecutorInfoCache",
            std::get<const ProgramDesc*>(key), std::get<bool>(key)));
    info_map_.insert({key_value, value});
  }

 private:
  ExecutorInfoCache() = default;

  HashTuple key_hash_func_;

  // Note: we shall avoid using raw pointer as key but use hash code,
  // beacause pointer doesn't hold resource indeed.
  std::unordered_map<KeyType, ValueType> info_map_;
  DISABLE_COPY_AND_ASSIGN(ExecutorInfoCache);
};

std::shared_ptr<ParallelExecutor> GetExecutorInfoFromCache(
    const ProgramDesc* program, int64_t start_op_index, int64_t end_op_index,
    const platform::Place& place, framework::Scope* scope,
    const std::vector<std::string>& output_var_names, bool is_grad);

}  // namespace framework
}  // namespace paddle
