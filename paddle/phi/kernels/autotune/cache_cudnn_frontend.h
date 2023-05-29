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

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "paddle/phi/backends/dynload/cudnn_frontend.h"

DECLARE_int32(cudnn_cache_saturation_count);

namespace phi {
namespace autotune {

class CudnnFrontendPlanCache {
 public:
  CudnnFrontendPlanCache() : cache_mutex_(new std::mutex()) {
    map_.clear();
    tracker_.clear();
    saturation_count_ = FLAGS_cudnn_cache_saturation_count;
  }

  int64_t Size() const { return map_.size(); }

  int64_t CacheHits() const { return cache_hits_; }

  int64_t CacheMisses() const { return cache_misses_; }

  float CacheHitRate() const {
    int64_t num_accesses = cache_hits_ + cache_misses_;
    float cache_hit_rate = 0.;
    if (num_accesses != 0) {
      cache_hit_rate =
          static_cast<float>(cache_hits_) / static_cast<float>(num_accesses);
    }
    return cache_hit_rate;
  }

  void Clean() {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    map_.clear();
    tracker_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
  }

  bool FindPlan(const cudnn_frontend::OperationGraph& op_graph,
                bool use_addto = false) {
    bool ret = false;
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (map_.count(MakeKey(op_graph, use_addto)) > 0) {
      cache_hits_++;
      ret = true;
    } else {
      cache_misses_++;
    }
    return ret;
  }

  cudnn_frontend::ManagedOpaqueDescriptor GetConfig(
      const cudnn_frontend::OperationGraph& op_graph,
      cudnnHandle_t handle,
      bool use_addto = false) {
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto engine_config = map_[MakeKey(op_graph, use_addto)];
    return engine_config;
  }

  void InsertPlan(const cudnn_frontend::OperationGraph& op_graph,
                  const cudnn_frontend::ExecutionPlan& plan,
                  bool use_addto = false) {
    VLOG(4) << "[cudnn_frontend] cache: Insert graph tag: "
            << op_graph.getTag();
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    map_.insert(
        std::make_pair(MakeKey(op_graph, use_addto), plan.GetEngineConfig()));
  }

  bool IsStable(const cudnn_frontend::OperationGraph& op_graph,
                const std::string& tag,
                bool use_addto = false) {
    if (saturation_count_ == 1) {
      return true;
    }
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (map_.count(MakeKey(op_graph, use_addto))) {
      return false;
    }
    int cnt = tracker_[std::make_pair(MakeKey(op_graph, use_addto), tag)] += 1;
    VLOG(4) << "[cudnn_frontend] SaturationTracker: " << op_graph.getTag()
            << " " << tag << " " << cnt;
    return cnt >= saturation_count_;
  }

 private:
  static cudnn_frontend::feature_vector_t MakeKey(
      const cudnn_frontend::OperationGraph& op_graph, bool use_addto) {
    auto key = op_graph.getFeatureVector();
    key.push_back(static_cast<uint64_t>(use_addto));
    return key;
  }

  std::map<cudnn_frontend::feature_vector_t,
           cudnn_frontend::ManagedOpaqueDescriptor>
      map_;
  std::shared_ptr<std::mutex> cache_mutex_;
  int saturation_count_;

  using SaturationTracker =
      std::map<std::pair<cudnn_frontend::feature_vector_t, std::string>, int>;
  SaturationTracker tracker_;

  int64_t cache_hits_{0};
  int64_t cache_misses_{0};
};  // class CudnnFrontendPlanCache

}  // namespace autotune
}  // namespace phi
