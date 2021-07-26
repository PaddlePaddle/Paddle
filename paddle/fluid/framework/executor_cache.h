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
class ExecutorInfoCache {
 public:
  struct CacheKey {
    CacheKey(const ProgramDesc* program_desc, const platform::Place& place,
             int64_t start_op_index, int64_t end_op_index, bool is_grad)
        : program_desc_(program_desc),
          place_(place),
          start_op_index_(start_op_index),
          end_op_index_(end_op_index),
          is_grad_(is_grad) {
      device_type_ = platform::Place2DeviceType(place);
      PADDLE_ENFORCE_NOT_NULL(program_desc_,
                              "program_desc should not be null.");
    }

    std::string DebugString() const {
      std::stringstream ss;

      ss << "\n CacheKey(program_desc: " << program_desc_;
      ss << ", start_op_index: " << start_op_index_;
      ss << ", end_op_index: " << end_op_index_;
      ss << ", is_grad: " << is_grad_;
      ss << ", device_type: " << device_type_ << ")";

      return ss.str();
    }

    const ProgramDesc* program_desc_;
    platform::Place place_;
    int64_t start_op_index_;
    int64_t end_op_index_;
    bool is_grad_;
    platform::DeviceType device_type_;
  };

  using KeyType = size_t;
  using ValueType =
      std::pair<std::shared_ptr<ParallelExecutor>, std::shared_ptr<ir::Graph>>;

  struct KeyHasher {
    size_t operator()(const CacheKey& key) const noexcept {
      size_t seed = 10;
      auto* prog_desc = key.program_desc_;
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
      hash_combine(&seed, static_cast<int>(key.device_type_));
      hash_combine(&seed, key.start_op_index_);
      hash_combine(&seed, key.end_op_index_);
      hash_combine(&seed, key.is_grad_);
      VLOG(3) << "hash value is : " << seed
              << " of key:  " << key.DebugString();
      return seed;
    }

    template <typename T>
    void hash_combine(size_t* seed, const T& val) const {
      std::hash<T> hasher;
      (*seed) ^= hasher(val) + 0x9e3779b9 + ((*seed) << 6) + ((*seed >> 2));
    }
  };

  static ExecutorInfoCache& Instance();

  ValueType GetMutable(const CacheKey& key) {
    auto key_val = key_hash_func_(key);
    PADDLE_ENFORCE_EQ(
        Has(key_val), true,
        platform::errors::NotFound("%s doesn't exist in ExecutorInfoCache",
                                   key.DebugString()));
    return info_map_[key_val];
  }

  bool Has(const CacheKey& key) const {
    auto key_val = key_hash_func_(key);
    return Has(key_val);
  }

  bool Has(const KeyType& key) const {
    return info_map_.find(key) != info_map_.end();
  }

  void Insert(const CacheKey& key, ValueType value) {
    auto key_val = key_hash_func_(key);
    PADDLE_ENFORCE_EQ(
        Has(key_val), false,
        platform::errors::NotFound("%s has existed in ExecutorInfoCache",
                                   key.DebugString()));
    info_map_.insert({key_val, value});
  }

  size_t Size() const { return info_map_.size(); }

  void Finalize();

 private:
  ExecutorInfoCache() = default;
  DISABLE_COPY_AND_ASSIGN(ExecutorInfoCache);

  KeyHasher key_hash_func_;
  std::unordered_map<KeyType, ValueType> info_map_;
};

using CacheInfo =
    std::pair<std::shared_ptr<ParallelExecutor>, bool /*is_new_created*/>;

CacheInfo GetExecutorInfoFromCache(const ExecutorInfoCache::CacheKey& cache_key,
                                   framework::Scope* scope);

}  // namespace framework
}  // namespace paddle
