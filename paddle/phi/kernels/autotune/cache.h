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

template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  AlgorithmsCache() { hash_.clear(); }

  template <typename... Args>
  size_t GetKey(Args&&... args) {
    size_t seed = 0;
    HashCombine(&seed, std::forward<Args>(args)...);
    return seed;
  }

  TAlgorithm Get(size_t key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    PADDLE_ENFORCE_NE(
        hash_.find(key),
        hash_.end(),
        phi::errors::PreconditionNotMet("The key does not exist."));
    return hash_[key];
  }

  bool Find(size_t key) {
    bool ret = false;
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (hash_.find(key) != hash_.end()) {
      cache_hits_++;
      ret = true;
    } else {
      cache_misses_++;
    }
    return ret;
  }

  void Set(size_t key, TAlgorithm algo) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    hash_[key] = algo;
  }

  float CacheHitRate() const {
    int64_t num_accesses = cache_hits_ + cache_misses_;
    float cache_hit_rate =
        static_cast<float>(cache_hits_) / static_cast<float>(num_accesses);
    return cache_hit_rate;
  }

 private:
  std::unordered_map<size_t, TAlgorithm> hash_;
  std::mutex cache_mutex;
  int64_t cache_hits_ = 0;
  int64_t cache_misses_ = 0;
};

}  // namespace phi
