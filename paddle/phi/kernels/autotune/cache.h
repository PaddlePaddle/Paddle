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

namespace autotune {

inline void HashCombine(std::size_t* seed) {}

template <typename T, typename... Rest>
inline void HashCombine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  HashCombine(seed, rest...);
}

}  // namespace autotune

// custom specialization of std::hash can be injected in namespace std
// ref: https://en.cppreference.com/w/cpp/utility/hash
template <typename T>
struct std::hash<std::vector<T>> {
  std::size_t operator()(std::vector<T> const& vec) const noexcept {
    std::size_t seed = 0;
    for (auto val : vec) {
      autotune::HashCombine(seed, val);
    }
    return seed;
  }
};

namespace phi {

template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  AlgorithmsCache() { hash_.clear(); }

  template <typename... Args>
  size_t GetKey(Args&&... args) {
    size_t seed = 0;
    autotune::HashCombine(&seed, std::forward<Args>(args)...);
    return seed;
  }

  TAlgorithm Get(size_t key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    PADDLE_ENFORCE_NE(hash_.find(key), hash_.end());
    return hash_[key];
  }

  bool Find(size_t key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return hash_.find(key) != hash_.end();
  }

  void Set(size_t key, TAlgorithm algo) {
    auto it = hash_.end();
    bool have_found = false;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      it = hash_.find(key);

      if (it != hash_.end()) {
        have_found = true;
      }
    }

    if (!have_found) {
      std::lock_guard<std::mutex> lock(cache_mutex);
      hash_[key] = algo;
    }
  }

 private:
  std::unordered_map<size_t, TAlgorithm> hash_;
  std::mutex cache_mutex;
};

/*
class AutoTuneCache {
 public:
  // AlgoType->KernelCache
  AutoTuneCache& AutoTuneCache::Instance() {
  static AutoTuneCache g_auto_tune_map_;
  return g_auto_tune_map_;
}

 private:
  int64_t cache_hits_;
  int64_t cache_misses_;
  std::vector<float> cache_hit_rates_;
  std::unordered_map<std::string, AlgorithmsCache> g_auto_tune_map_;
};
*/

}  // namespace phi
