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
#ifdef PADDLE_WITH_CUDNN_FRONTEND
#include "paddle/phi/kernels/autotune/cache_cudnn_frontend.h"
#endif
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
  kMatmul = 5,
  kGatherGemmScatterFP16NN = 6,
  kGatherGemmScatterFP32NN = 7,
  kGatherGemmScatterFP32TN = 8,
  kGatherGemmScatterFP32NT = 9,
#if !defined(PADDLE_WITH_CUDNN_FRONTEND)
  kAlgorithmCount = 10
#else
  kConvForwardV8 = 10,
  kConvBackwardDataV8 = 11,
  kConvBackwardFilterV8 = 12,
  kAlgorithmCount = 13
#endif
};

// AlgorithmsConfigKey -> AlgorithmsID
// AlgorithmType -> AlgorithmsCache
using AlgorithmsCacheMap = AlgorithmsCache<size_t, int64_t>;
using AlgorithmsTypeMap = std::unordered_map<int64_t, AlgorithmsCacheMap>;

// (todo. hong) use cudnnConvolutionFwdAlgo_t
using ConvAlgorithmsCacheMap = ConvAlgorithmsCache<ConvAutoTuneResult>;
using ConvAlgorithmsTypeMap =
    std::unordered_map<int64_t, ConvAlgorithmsCacheMap>;

using MatmulAlgorithmsCacheMap = MatmulAlgorithmsCache<size_t, int64_t>;
#ifdef PADDLE_WITH_CUDNN_FRONTEND
using CudnnV8AlgorithmsTypeMap =
    std::unordered_map<int64_t, CudnnFrontendPlanCache>;
#endif

#define DEFINE_GET_GATHER_GEMM_SCATTER(                    \
    dtype, transpose_a, transpose_b, algo_type)            \
  template <typename T, bool TransposeA, bool TransposeB>  \
  typename std::enable_if<std::is_same<T, dtype>::value && \
                              TransposeA == transpose_a && \
                              TransposeB == transpose_b,   \
                          AlgorithmsCacheMap&>::type       \
  GetGatherGemmScatter() {                                 \
    return Get(algo_type);                                 \
  }

class AutoTuneCache {
 public:
  static AutoTuneCache& Instance() {
    static AutoTuneCache autotune_cache;
    return autotune_cache;
  }

  AlgorithmsCacheMap& Get(const AlgorithmType& algo_type) {
    return auto_tune_map_[static_cast<int64_t>(algo_type)];
  }

  MatmulAlgorithmsCacheMap& GetMatmul() { return matmul_auto_tune_map_; }

  ConvAlgorithmsCacheMap& GetConv(const AlgorithmType& algo_type) {
    return conv_auto_tune_map_[static_cast<int64_t>(algo_type)];
  }
  DEFINE_GET_GATHER_GEMM_SCATTER(phi::dtype::float16,
                                 false,
                                 false,
                                 AlgorithmType::kGatherGemmScatterFP16NN);
  DEFINE_GET_GATHER_GEMM_SCATTER(float,
                                 false,
                                 false,
                                 AlgorithmType::kGatherGemmScatterFP32NN);
  DEFINE_GET_GATHER_GEMM_SCATTER(float,
                                 true,
                                 false,
                                 AlgorithmType::kGatherGemmScatterFP32TN);
  DEFINE_GET_GATHER_GEMM_SCATTER(float,
                                 false,
                                 true,
                                 AlgorithmType::kGatherGemmScatterFP32NT);

#ifdef PADDLE_WITH_CUDNN_FRONTEND
  CudnnFrontendPlanCache& GetConvV8(const AlgorithmType& algo_type) {
    return cudnn_v8_auto_tune_map_[static_cast<int64_t>(algo_type)];
  }
#endif

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
  MatmulAlgorithmsCacheMap matmul_auto_tune_map_;
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
