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

#include "paddle/phi/kernels/autotune/cache.h"

#include <iomanip>
#include <map>

#include "glog/logging.h"

namespace phi {
namespace autotune {

size_t TransposeKey(const std::vector<int64_t>& x_dims,
                    const std::vector<int32_t>& perm,
                    phi::DataType dtype) {
  const auto rank = perm.size();
  return GenKey(x_dims, perm, rank, static_cast<int>(dtype));
}

std::string AlgorithmTypeString(int64_t algo_type) {
  static const std::map<AlgorithmType, std::string> algo_map = {
      {AlgorithmType::kConvForward, "conv_forward"},
      {AlgorithmType::kConvBackwardData, "conv_backward_data"},
      {AlgorithmType::kConvBackwardFilter, "conv_backward_filter"},
#ifdef PADDLE_WITH_CUDNN_FRONTEND
      {AlgorithmType::kConvForwardV8, "conv_forward_v8"},
      {AlgorithmType::kConvBackwardDataV8, "conv_backward_data_v8"},
      {AlgorithmType::kConvBackwardFilterV8, "conv_backward_filter_v8"},
      {AlgorithmType::kScaleBiasReluConvBNstats,
       "scale_bias_relu_conv_bnstats"},
      {AlgorithmType::kBNFinalize, "bn_finalize"},
      {AlgorithmType::kScaleBiasAddRelu, "scale_bias_add_relu"},
      {AlgorithmType::kDgradDreluBnBwdWeight, "dgrad_drelu_bnbwdweight"},
      {AlgorithmType::kBnActWgrad, "bn_act_wgrad"},
      {AlgorithmType::kDbnApply, "dbn_apply"},
#endif
  };

  auto it = algo_map.find(static_cast<AlgorithmType>(algo_type));
  if (it != algo_map.end()) {
    return it->second;
  } else {
    return std::to_string(algo_type);
  }
}

void AutoTuneCache::UpdateStatus() {
  int64_t size = 0;
  int64_t cache_hits = 0;
  int64_t cache_misses = 0;
  int name_width = 24;
  std::cout.setf(std::ios::left);
  for (auto& v : auto_tune_map_) {
    VLOG(4) << "AlgoType: " << std::setfill(' ') << std::setw(name_width)
            << AlgorithmTypeString(v.first)
            << " Cache Size: " << v.second.Size()
            << " Hits: " << v.second.CacheHits()
            << " Misses: " << v.second.CacheMisses()
            << " Hit Rate: " << v.second.CacheHitRate();
    size += v.second.Size();
    cache_hits += v.second.CacheHits();
    cache_misses += v.second.CacheMisses();
  }

  for (auto& v : conv_auto_tune_map_) {
    VLOG(4) << "AlgoType: " << std::setfill(' ') << std::setw(name_width)
            << AlgorithmTypeString(v.first)
            << " Cache Size: " << v.second.Size()
            << " Hits: " << v.second.CacheHits()
            << " Misses: " << v.second.CacheMisses()
            << " Hit Rate: " << v.second.CacheHitRate();
    size += v.second.Size();
    cache_hits += v.second.CacheHits();
    cache_misses += v.second.CacheMisses();
  }

#ifdef PADDLE_WITH_CUDNN_FRONTEND
  for (auto& v : cudnn_v8_auto_tune_map_) {
    VLOG(4) << "AlgoType: " << std::setfill(' ') << std::setw(name_width)
            << AlgorithmTypeString(v.first)
            << " Cache Size: " << v.second.Size()
            << " Hits: " << v.second.CacheHits()
            << " Misses: " << v.second.CacheMisses()
            << " Hit Rate: " << v.second.CacheHitRate();
    size += v.second.Size();
    cache_hits += v.second.CacheHits();
    cache_misses += v.second.CacheMisses();
  }
#endif

  total_size_ = size;
  total_cache_hits_ = cache_hits;
  total_cache_misses_ = cache_misses;
}

}  // namespace autotune
}  // namespace phi
