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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/autotune/cache_base.h"

namespace phi {
namespace autotune {

struct ConvAutoTuneResult {
  ConvAutoTuneResult() {}
  ConvAutoTuneResult(int64_t a, size_t size, bool search)
      : algo(a), workspace_size(size), exhaustive_search(search) {}

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
  kTranspose = 4,
  kAlgorithmCount = 5
};

// AlgorithmsConfigKey -> AlgorithmsID
// (todo. hong) use cudnnConvolutionFwdAlgo_t
using AlgorithmsCacheMap = AlgorithmsCache<size_t, int64_t>;
// AlgorithmType -> AlgorithmsCache
using AlgorithmsTypeMap = std::unordered_map<int64_t, AlgorithmsCacheMap>;
using ConvAlgorithmsCacheMap = ConvAlgorithmsCache<ConvAutoTuneResult>;
using ConvAlgorithmsTypeMap =
    std::unordered_map<int64_t, ConvAlgorithmsCacheMap>;

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

  AlgorithmsCacheMap& GetTranspose() { return Get(AlgorithmType::kTranspose); }

  void Clean() {
    for (auto& v : auto_tune_map_) {
      v.second.Clean();
    }

    for (auto& v : conv_auto_tune_map_) {
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
        ConvAlgorithmsCacheMap cache;
        conv_auto_tune_map_[key] = cache;
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
  ConvAlgorithmsTypeMap conv_auto_tune_map_;
  std::shared_ptr<std::mutex> autotune_cache_mutex_;
  int64_t total_cache_hits_{0};
  int64_t total_cache_misses_{0};
  int64_t total_size_{0};
};

}  // namespace autotune
}  // namespace phi
