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
#include <numeric>
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

DECLARE_int32(search_cache_max_number);

inline void HashCombine(std::size_t* seed) {}

// combine hash value
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <typename T, typename... Rest>
inline void HashCombine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  *seed *= 0x00000100000001B3;
  HashCombine(seed, rest...);
}

// custom specialization of std::hash can be injected in namespace std
// ref: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(std::vector<T> const& vec) const noexcept {
    std::size_t seed = 0xcbf29ce484222325;
    for (auto val : vec) {
      HashCombine(&seed, val);
    }
    return seed;
  }
};
}  // namespace std

namespace phi {
namespace autotune {

struct DnnNode {
  DnnNode() {}
  explicit DnnNode(int64_t a, size_t size) : algo(a), workspace_size(size) {}

  int64_t algo;
  size_t workspace_size = 0;
};

template <typename... Args>
size_t GetKey(Args&&... args) {
  size_t seed = 0;
  HashCombine(&seed, std::forward<Args>(args)...);
  return seed;
}

struct ConvCacheKey {
  ConvCacheKey() {}
  explicit ConvCacheKey(const std::vector<int64_t>& x_dims,
                        const std::vector<int64_t>& w_dims,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        phi::DataType dtype,
                        int groups,
                        int64_t data_layout)
      : x_dims_(x_dims),
        w_dims_(w_dims),
        strides_(strides),
        paddings_(paddings),
        dilations_(dilations),
        dtype_(dtype),
        groups_(groups),
        data_layout_(data_layout) {}
  size_t hash_value() const {
    return GetKey(x_dims_,
                  w_dims_,
                  strides_,
                  paddings_,
                  dilations_,
                  static_cast<int64_t>(dtype_),
                  groups_,
                  data_layout_);
  }
  std::vector<int64_t> x_dims_;
  std::vector<int64_t> w_dims_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  phi::DataType dtype_;
  int groups_;
  int64_t data_layout_;
};

struct ConvCacheKeyHash {
  size_t operator()(const ConvCacheKey& cache) const {
    return cache.hash_value();
  }
};

struct ConvCacheKeyEqual {
  size_t operator()(const ConvCacheKey& first,
                    const ConvCacheKey& second) const {
    if (first.x_dims_ != second.x_dims_) return false;
    if (first.w_dims_ != second.w_dims_) return false;
    if (first.strides_ != second.strides_) return false;
    if (first.paddings_ != second.paddings_) return false;
    if (first.dilations_ != second.dilations_) return false;
    if (first.dtype_ != second.dtype_) return false;
    if (first.groups_ != second.groups_) return false;
    if (first.data_layout_ != second.data_layout_) return false;

    return true;
  }
};

class CudnnAlgorithmsCacheMap {
 public:
  CudnnAlgorithmsCacheMap() : cache_mutex_(new std::mutex()) { hash_.clear(); }

  DnnNode Get(const ConvCacheKey& key) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    PADDLE_ENFORCE_NE(
        hash_.find(key),
        hash_.end(),
        phi::errors::PreconditionNotMet("The key does not exist."));
    return hash_[key];
  }

  bool Find(const ConvCacheKey& key) {
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

  void Set(const ConvCacheKey& key, DnnNode algo) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (hash_.size() > static_cast<size_t>(FLAGS_search_cache_max_number)) {
      hash_.clear();
    }
    hash_[key] = algo;
  }

  int64_t CacheMisses() const { return cache_misses_; }

  int64_t CacheHits() const { return cache_hits_; }

  float CacheHitRate() const {
    int64_t num_accesses = cache_hits_ + cache_misses_;
    float cache_hit_rate = 0.;
    if (num_accesses != 0) {
      cache_hit_rate =
          static_cast<float>(cache_hits_) / static_cast<float>(num_accesses);
    }
    return cache_hit_rate;
  }

  int64_t Size() const { return hash_.size(); }

 private:
  std::unordered_map<ConvCacheKey, DnnNode, ConvCacheKeyHash, ConvCacheKeyEqual>
      hash_;
  std::shared_ptr<std::mutex> cache_mutex_;

  int64_t cache_hits_{0};
  int64_t cache_misses_{0};
};

size_t TransposeKey(const std::vector<int64_t>& x_dims,
                    const std::vector<int32_t>& perm,
                    phi::DataType dtype);

template <typename AlgorithmT>
class AlgorithmsCache {
 public:
  AlgorithmsCache() : cache_mutex_(new std::mutex()) { hash_.clear(); }

  AlgorithmT Get(const size_t& key) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    PADDLE_ENFORCE_NE(
        hash_.find(key),
        hash_.end(),
        phi::errors::PreconditionNotMet("The key does not exist."));
    return hash_[key];
  }

  bool Find(const size_t& key) {
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

  void Set(const size_t& key, AlgorithmT algo) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    hash_[key] = algo;
  }

  int64_t CacheMisses() const { return cache_misses_; }

  int64_t CacheHits() const { return cache_hits_; }

  float CacheHitRate() const {
    int64_t num_accesses = cache_hits_ + cache_misses_;
    float cache_hit_rate = 0.;
    if (num_accesses != 0) {
      cache_hit_rate =
          static_cast<float>(cache_hits_) / static_cast<float>(num_accesses);
    }
    return cache_hit_rate;
  }

  int64_t Size() const { return hash_.size(); }

 private:
  std::unordered_map<size_t, AlgorithmT> hash_;
  std::shared_ptr<std::mutex> cache_mutex_;

  int64_t cache_hits_{0};
  int64_t cache_misses_{0};
};

enum class AlgorithmType {
  kConvForward = 1,
  kConvBackwardData = 2,
  kConvBackwardFilter = 3,
  kTranspose = 4,
  kAlgorithmCount = 5
};

// AlgorithmsConfigKey -> AlgorithmsID
// (todo. hong) use cudnnConvolutionFwdAlgo_t
using AlgorithmsCacheMap = AlgorithmsCache<int64_t>;
// AlgorithmType -> AlgorithmsCache
using AlgorithmsTypeMap = std::unordered_map<int64_t, AlgorithmsCacheMap>;
using CudnnAlgorithmsTypeMap =
    std::unordered_map<int64_t, CudnnAlgorithmsCacheMap>;

class AutoTuneCache {
 public:
  static AutoTuneCache& Instance() {
    static AutoTuneCache autotune_cache;
    return autotune_cache;
  }

  AlgorithmsCacheMap& Get(const AlgorithmType& algo_type) {
    return auto_tune_map_[static_cast<int64_t>(algo_type)];
  }

  CudnnAlgorithmsCacheMap& GetConv(const AlgorithmType& algo_type) {
    return cudnn_auto_tune_map_[static_cast<int64_t>(algo_type)];
  }

  CudnnAlgorithmsCacheMap& GetConvForward() {
    return cudnn_auto_tune_map_[static_cast<int64_t>(
        AlgorithmType::kConvForward)];
  }

  CudnnAlgorithmsCacheMap& GetConvBackwardData() {
    return cudnn_auto_tune_map_[static_cast<int64_t>(
        AlgorithmType::kConvBackwardData)];
  }

  CudnnAlgorithmsCacheMap& GetConvBackwardFilter() {
    return cudnn_auto_tune_map_[static_cast<int64_t>(
        AlgorithmType::kConvBackwardFilter)];
  }

  AlgorithmsCacheMap& GetTranspose() { return Get(AlgorithmType::kTranspose); }

  void Clean() {
    for (auto& v : auto_tune_map_) {
      v.second.Clean();
    }

    for (auto& v : cudnn_auto_tune_map_) {
      v.second.Clean();
    }
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
        CudnnAlgorithmsCacheMap cache;
        cudnn_auto_tune_map_[key] = cache;
      }
    } else {
      int64_t key = static_cast<int64_t>(algo_type);
      if (auto_tune_map_.find(key) == auto_tune_map_.end()) {
        AlgorithmsCacheMap cache;
        auto_tune_map_[key] = cache;
      }
    }
  }

  AlgorithmsTypeMap auto_tune_map_;
  CudnnAlgorithmsTypeMap cudnn_auto_tune_map_;
  std::shared_ptr<std::mutex> autotune_cache_mutex_;
  int64_t total_cache_hits_{0};
  int64_t total_cache_misses_{0};
  int64_t total_size_{0};
};

}  // namespace autotune
}  // namespace phi
