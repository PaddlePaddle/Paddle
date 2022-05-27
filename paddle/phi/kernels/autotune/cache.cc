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
#include "glog/logging.h"

namespace phi {
namespace autotune {

// Define the cache key of operator
size_t ConvKey(const std::vector<int64_t>& x_dims,
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

std::string AlgorithmTypeString(int64_t algo_type) {
  if (algo_type == static_cast<int64_t>(AlgorithmType::kConvForward)) {
    return "conv_forward";
  } else if (algo_type ==
             static_cast<int64_t>(AlgorithmType::kConvBackwardData)) {
    return "conv_backward_data";
  } else if (algo_type ==
             static_cast<int64_t>(AlgorithmType::kConvBackwardFilter)) {
    return "conv_backward_filter";
  }
  return std::to_string(algo_type);
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
  total_size_ = size;
  total_cache_hits_ = cache_hits;
  total_cache_misses_ = cache_misses;
}

}  // namespace autotune
}  // namespace phi
