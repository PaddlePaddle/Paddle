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
#include <numeric>
<<<<<<< HEAD
#include <unordered_map>
#include <vector>

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
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/autotune/cache_base.h"
#ifdef PADDLE_WITH_CUDNN_FRONTEND
#include "paddle/phi/kernels/autotune/cache_cudnn_frontend.h"
#endif
namespace phi {
namespace autotune {

<<<<<<< HEAD
template <typename... Args>
size_t GetKey(Args&&... args) {
  size_t seed = 0;
  HashCombine(&seed, std::forward<Args>(args)...);
  return seed;
}

// Define the cache key of operator
size_t ConvKey(const std::vector<int64_t>& x_dims,
               const std::vector<int64_t>& w_dims,
               const std::vector<int>& strides,
               const std::vector<int>& paddings,
               const std::vector<int>& dilations,
               phi::DataType dtype);

size_t TransposeKey(const std::vector<int64_t>& x_dims,
                    const std::vector<int32_t>& perm,
                    phi::DataType dtype);

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

  void Clean() {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    hash_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
  }

  void Set(size_t key, AlgorithmT algo) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    hash_[key] = algo;
  }

  int64_t CacheMisses() const { return cache_misses_; }

  int64_t CacheHits() const { return cache_hits_; }
=======
struct ConvAutoTuneResult {
  ConvAutoTuneResult() {}
  ConvAutoTuneResult(int64_t a, size_t size, bool search)
      : algo(a), workspace_size(size), exhaustive_search(search) {}
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

  int64_t algo;
  size_t workspace_size = 0;
  bool exhaustive_search = false;
};

size_t TransposeKey(const std::vector<int64_t>& x_dims,
                    const std::vector<int32_t>& perm,
                    phi::DataType dtype);

enum class AlgorithmType {
  kConvForward = 1,
  kConvBackwardData = 2,
  kConvBackwardFilter = 3,
<<<<<<< HEAD
  kTranspose = 4,
  kAlgorithmCount = 5
=======
#ifdef PADDLE_WITH_CUDNN_FRONTEND
  kConvForwardV8 = 4,
  kConvBackwardDataV8 = 5,
  kConvBackwardFilterV8 = 6,
  kTranspose = 7,
  kAlgorithmCount = 8
#else
  kTranspose = 4,
  kAlgorithmCount = 5
#endif
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
};

// AlgorithmsConfigKey -> AlgorithmsID
// (todo. hong) use cudnnConvolutionFwdAlgo_t
using AlgorithmsCacheMap = AlgorithmsCache<size_t, int64_t>;
// AlgorithmType -> AlgorithmsCache
using AlgorithmsTypeMap = std::unordered_map<int64_t, AlgorithmsCacheMap>;
using ConvAlgorithmsCacheMap = ConvAlgorithmsCache<ConvAutoTuneResult>;
using ConvAlgorithmsTypeMap =
    std::unordered_map<int64_t, ConvAlgorithmsCacheMap>;
#ifdef PADDLE_WITH_CUDNN_FRONTEND
using CudnnV8AlgorithmsTypeMap =
    std::unordered_map<int64_t, CudnnFrontendPlanCache>;
#endif
class AutoTuneCache {
 public:
  static AutoTuneCache& Instance() {
    static AutoTuneCache autotune_cache;
    return autotune_cache;
  }

  AlgorithmsCacheMap& Get(const AlgorithmType& algo_type) {
    return auto_tune_map_[static_cast<int64_t>(algo_type)];
  }

  ConvAlgorithmsCacheMap& GetConv(const AlgorithmType& algo_type) {
    return conv_auto_tune_map_[static_cast<int64_t>(algo_type)];
  }

#ifdef PADDLE_WITH_CUDNN_FRONTEND
  CudnnFrontendPlanCache& GetConvV8(const AlgorithmType& algo_type) {
    return cudnn_v8_auto_tune_map_[static_cast<int64_t>(algo_type)];
  }
#endif

  AlgorithmsCacheMap& GetTranspose() { return Get(AlgorithmType::kTranspose); }

  AlgorithmsCacheMap& GetTranspose() { return Get(AlgorithmType::kTranspose); }

  void Clean() {
    for (auto& v : auto_tune_map_) {
      v.second.Clean();
    }

    for (auto& v : conv_auto_tune_map_) {
      v.second.Clean();
    }

#ifdef PADDLE_WITH_CUDNN_FRONTEND
    for (auto& v : cudnn_v8_auto_tune_map_) {
      v.second.Clean();
    }
#endif
  }

  void UpdateStatus();

  // The number of total config cached
  int64_t Size() const { return total_size_; }

  int64_t CacheHits() const { return total_cache_hits_; }

  int64_t CacheMisses() const { return total_cache_misses_; }

  float CacheHitRate() const {
    float total_cache_hit_rate = 0.;
    int64_t total_num_accesses = total_cache_hits_ + total_cache_misses_;
    if (total_num_accesses != 0) {
      total_cache_hit_rate = static_cast<float>(total_cache_hits_) /
                             static_cast<float>(total_num_accesses);
    }
    return total_cache_hit_rate;
  }

 private:
  AutoTuneCache() : autotune_cache_mutex_(new std::mutex()) {
    for (int i = 1; i < static_cast<int>(AlgorithmType::kAlgorithmCount); ++i) {
      Register(static_cast<AlgorithmType>(i));
    }
  }

  void Register(const AlgorithmType& algo_type) {
    std::lock_guard<std::mutex> lock(*autotune_cache_mutex_);
    if (algo_type == AlgorithmType::kConvForward ||
        algo_type == AlgorithmType::kConvBackwardData ||
        algo_type == AlgorithmType::kConvBackwardFilter) {
      int64_t key = static_cast<int64_t>(algo_type);
      if (auto_tune_map_.find(key) == auto_tune_map_.end()) {
        ConvAlgorithmsCacheMap cache;
        conv_auto_tune_map_[key] = cache;
      }
#ifdef PADDLE_WITH_CUDNN_FRONTEND
    } else if (algo_type == AlgorithmType::kConvForwardV8 ||
               algo_type == AlgorithmType::kConvBackwardDataV8 ||
               algo_type == AlgorithmType::kConvBackwardFilterV8) {
      int64_t key = static_cast<int64_t>(algo_type);
      if (cudnn_v8_auto_tune_map_.find(key) == cudnn_v8_auto_tune_map_.end()) {
        CudnnFrontendPlanCache cache;
        cudnn_v8_auto_tune_map_[key] = cache;
      }
#endif
    } else {
      int64_t key = static_cast<int64_t>(algo_type);
      if (auto_tune_map_.find(key) == auto_tune_map_.end()) {
        AlgorithmsCacheMap cache;
        auto_tune_map_[key] = cache;
      }
    }
  }

  AlgorithmsTypeMap auto_tune_map_;
  ConvAlgorithmsTypeMap conv_auto_tune_map_;
#ifdef PADDLE_WITH_CUDNN_FRONTEND
  CudnnV8AlgorithmsTypeMap cudnn_v8_auto_tune_map_;
#endif
  std::shared_ptr<std::mutex> autotune_cache_mutex_;
  int64_t total_cache_hits_{0};
  int64_t total_cache_misses_{0};
  int64_t total_size_{0};
};

}  // namespace autotune
}  // namespace phi
