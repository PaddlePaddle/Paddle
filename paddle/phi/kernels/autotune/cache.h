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
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

inline void HashCombine(std::size_t* seed) {}

// combine hash value
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <typename T, typename... Rest>
inline void HashCombine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  HashCombine(seed, rest...);
}

// custom specialization of std::hash can be injected in namespace std
// ref: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(std::vector<T> const& vec) const noexcept {
    std::size_t seed = 0;
    for (auto val : vec) {
      HashCombine(&seed, val);
    }
    return seed;
  }
};
}  // namespace std

namespace phi {
namespace autotune {

template <typename... Args>
size_t GetKey(Args&&... args) {
  size_t seed = 0;
  HashCombine(&seed, std::forward<Args>(args)...);
  return seed;
}

// Define the cache key of operator
static size_t ConvKey(const std::vector<int64_t>& x_dims,
                      const std::vector<int64_t>& w_dims,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      phi::DataType dtype) {
  return GetKey(x_dims,
                w_dims,
                strides,
                paddings,
                dilations,
                static_cast<int64_t>(dtype));
}

template <typename AlgorithmT>
class AlgorithmsCache {
 public:
  AlgorithmsCache() : cache_mutex_(new std::mutex()) { hash_.clear(); }

  AlgorithmT Get(size_t key) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    PADDLE_ENFORCE_NE(
        hash_.find(key),
        hash_.end(),
        phi::errors::PreconditionNotMet("The key does not exist."));
    return hash_[key];
  }

  bool Find(size_t key) {
    bool ret = false;
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (hash_.find(key) != hash_.end()) {
      cache_hits_++;
      ret = true;
    } else {
      cache_misses_++;
    }
    return ret;
  }

  void Set(size_t key, AlgorithmT algo) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    hash_[key] = algo;
  }

  float CacheHitRate() const {
    int64_t num_accesses = cache_hits_ + cache_misses_;
    float cache_hit_rate =
        static_cast<float>(cache_hits_) / static_cast<float>(num_accesses);
    return cache_hit_rate;
  }

  int64_t Size() { return hash_.size(); }

 private:
  std::unordered_map<size_t, AlgorithmT> hash_;
  std::shared_ptr<std::mutex> cache_mutex_;
  int64_t cache_hits_ = 0;
  int64_t cache_misses_ = 0;
};

// AlgorithmsConfigKey -> AlgorithmsID
using AlgorithmsConfigKeyMap = AlgorithmsCache<int64_t>;
// AlgorithmsType -> AlgorithmsCache
using AlgorithmsTypeMap =
    std::unordered_map<std::string, AlgorithmsConfigKeyMap>;

class AutoTuneCache {
 public:
  static AutoTuneCache& Instance() {
    static AutoTuneCache autotune_cache;
    return autotune_cache;
  }

  AlgorithmsConfigKeyMap& RegisterOrGet(const std::string& algo_type) {
    std::lock_guard<std::mutex> lock(*autotune_cache_mutex_);
    if (auto_tune_map_.find(algo_type) == auto_tune_map_.end()) {
      AlgorithmsConfigKeyMap cache;
      auto_tune_map_[algo_type] = cache;
    }
    return auto_tune_map_[algo_type];
  }

  // The number of total config cached
  int64_t Size() {
    int64_t total = 0;
    for (auto& v : auto_tune_map_) {
      VLOG(3) << v.first << " " << v.second.Size();
      total += v.second.Size();
    }
    return total;
  }

 private:
  AutoTuneCache() : autotune_cache_mutex_(new std::mutex()) {}
  AlgorithmsTypeMap auto_tune_map_;
  std::shared_ptr<std::mutex> autotune_cache_mutex_;
};

}  // namespace autotune
}  // namespace phi
