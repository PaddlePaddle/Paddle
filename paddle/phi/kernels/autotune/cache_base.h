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

#include <mutex>
#include <unordered_map>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/common/hash_funcs.h"
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_int32(search_cache_max_number);

namespace phi {
namespace autotune {

template <typename... Args>
size_t GenKey(Args&&... args) {
  size_t seed = 0;
  HashCombine(&seed, std::forward<Args>(args)...);
  return seed;
}

struct ConvCacheKey {
  ConvCacheKey() {}
  ConvCacheKey(const std::vector<int64_t>& arg_x_dims,
               const std::vector<int64_t>& arg_w_dims,
               const std::vector<int>& arg_strides,
               const std::vector<int>& arg_paddings,
               const std::vector<int>& arg_dilations,
               phi::DataType arg_dtype,
               int arg_groups,
               int64_t arg_data_layout)
      : x_dims(arg_x_dims),
        w_dims(arg_w_dims),
        strides(arg_strides),
        paddings(arg_paddings),
        dilations(arg_dilations),
        dtype(arg_dtype),
        groups(arg_groups),
        data_layout(arg_data_layout) {}
  size_t hash_value() const {
    return GenKey(x_dims,
                  w_dims,
                  strides,
                  paddings,
                  dilations,
                  static_cast<int64_t>(dtype),
                  groups,
                  data_layout);
  }

  std::vector<int64_t> x_dims;
  std::vector<int64_t> w_dims;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> dilations;
  phi::DataType dtype;
  int groups;
  int64_t data_layout;
};

struct ConvCacheKeyHash {
  size_t operator()(const ConvCacheKey& cache) const {
    return cache.hash_value();
  }
};

struct ConvCacheKeyEqual {
  size_t operator()(const ConvCacheKey& first,
                    const ConvCacheKey& second) const {
    if (first.x_dims != second.x_dims) return false;
    if (first.w_dims != second.w_dims) return false;
    if (first.strides != second.strides) return false;
    if (first.paddings != second.paddings) return false;
    if (first.dilations != second.dilations) return false;
    if (first.dtype != second.dtype) return false;
    if (first.groups != second.groups) return false;
    if (first.data_layout != second.data_layout) return false;

    return true;
  }
};

template <typename KeyT,
          typename AlgorithmT,
          typename HashT = std::hash<KeyT>,
          typename KeyEqualT = std::equal_to<KeyT>>
class AlgorithmsCache {
 public:
  AlgorithmsCache() : cache_mutex_(new std::mutex()) {}

  AlgorithmT Get(const KeyT& key) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    PADDLE_ENFORCE_NE(
        hash_.find(key),
        hash_.end(),
        common::errors::PreconditionNotMet("The key does not exist."));
    return hash_[key];
  }

  bool Find(const KeyT& key) {
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

  void Set(const KeyT& key, AlgorithmT algo) {
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

 protected:
  std::unordered_map<KeyT, AlgorithmT, HashT, KeyEqualT> hash_;
  std::shared_ptr<std::mutex> cache_mutex_;

  int64_t cache_hits_{0};
  int64_t cache_misses_{0};
};

template <typename AlgorithmT>
class ConvAlgorithmsCache : public AlgorithmsCache<ConvCacheKey,
                                                   AlgorithmT,
                                                   ConvCacheKeyHash,
                                                   ConvCacheKeyEqual> {
 public:
  using AlgorithmsCacheBase = AlgorithmsCache<ConvCacheKey,
                                              AlgorithmT,
                                              ConvCacheKeyHash,
                                              ConvCacheKeyEqual>;

  ConvAlgorithmsCache()
      : AlgorithmsCache<ConvCacheKey,
                        AlgorithmT,
                        ConvCacheKeyHash,
                        ConvCacheKeyEqual>() {}

  void Set(const ConvCacheKey& key, AlgorithmT algo) {
    std::lock_guard<std::mutex> lock(*AlgorithmsCacheBase::cache_mutex_);
    if (AlgorithmsCacheBase::hash_.size() >
        static_cast<size_t>(FLAGS_search_cache_max_number)) {
      AlgorithmsCacheBase::hash_.clear();
    }
    AlgorithmsCacheBase::hash_[key] = algo;
  }
};

template <typename KeyT, typename AlgorithmT>
class MatmulAlgorithmsCache : public AlgorithmsCache<KeyT, AlgorithmT> {
 public:
  MatmulAlgorithmsCache() : AlgorithmsCache<KeyT, AlgorithmT>() {}

  bool FindSubKey(const KeyT& sub_key) {
    std::lock_guard<std::mutex> lock(*(this->cache_mutex_));
    bool ret = (sub_hash_.find(sub_key) != sub_hash_.end()) ? true : false;
    return ret;
  }

  void SetSubKey(const KeyT& sub_key, void* algo) {
    std::lock_guard<std::mutex> lock(*(this->cache_mutex_));
    sub_hash_[sub_key] = algo;
  }

  void* GetSubKey(const KeyT& sub_key) {
    std::lock_guard<std::mutex> lock(*(this->cache_mutex_));
    PADDLE_ENFORCE_NE(
        sub_hash_.find(sub_key),
        sub_hash_.end(),
        common::errors::PreconditionNotMet("The key does not exist."));
    return sub_hash_[sub_key];
  }

 private:
  std::unordered_map<KeyT, void*> sub_hash_;
};

}  // namespace autotune
}  // namespace phi
