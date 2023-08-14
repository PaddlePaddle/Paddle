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
#include <thread>
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

  int64_t Size() const {
    int64_t total_size = 0;
    for (auto it = map_.begin(); it != map_.end(); it++) {
      total_size += (it->second).size();
    }
    return total_size;
  }

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

  bool FindPlan(const cudnn_frontend::feature_vector_t &feature,
                cudnnHandle_t handle) {
    bool ret = false;
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    if (local_map.count(GetExtendedFeature(feature, handle)) > 0) {
      cache_hits_++;
      ret = true;
    } else {
      cache_misses_++;
    }
    return ret;
  }

  void GetPlan(const cudnn_frontend::feature_vector_t &feature,
               const cudnn_frontend::ExecutionPlan **plan,
               int64_t *workspace_size,
               cudnnHandle_t handle) {
    // Note(tizheng): CUDNNv8 execution plan is not thread-safe.
    // A shared plan being executed by different threads is
    // generally not safe (for now).
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];

    auto it = local_map.find(GetExtendedFeature(feature, handle));
    if (it == local_map.end()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "[cudnn_frontend] Cached Plan Not Found."));
      return;
    }
    *plan = &(it->second);
    *workspace_size = (*plan)->getWorkspaceSize();
    VLOG(4) << "Cached execution plan found." << (*plan)->getTag()
            << "; Require workspace: " << *workspace_size;
  }

  void InsertPlan(const cudnn_frontend::feature_vector_t &feature,
                  const cudnn_frontend::ExecutionPlan &plan,
                  cudnnHandle_t handle) {
    VLOG(4) << "[cudnn_frontend] cache: Insert plan: " << plan.getTag();
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    local_map.insert(std::make_pair(GetExtendedFeature(feature, handle), plan));
  }

  bool IsStable(const cudnn_frontend::feature_vector_t &feature,
                const std::string &tag,
                cudnnHandle_t handle) {
    if (saturation_count_ == 1) {
      return true;
    }
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    auto &local_tracker = tracker_[hasher(std::this_thread::get_id())];
    auto ext_feature = GetExtendedFeature(feature, handle);
    if (local_map.count(ext_feature)) {
      return false;
    }
    int cnt = local_tracker[std::make_pair(ext_feature, tag)] += 1;
    VLOG(4) << "[cudnn_frontend] SaturationTracker: " << tag << " " << cnt;
    return cnt >= saturation_count_;
  }

  bool FindPlan(const cudnn_frontend::OperationGraph &op_graph,
                cudnnHandle_t handle) {
    return FindPlan(op_graph.getFeatureVector(), handle);
  }

  void GetPlan(const cudnn_frontend::OperationGraph &op_graph,
               const cudnn_frontend::ExecutionPlan **plan,
               int64_t *workspace_size,
               cudnnHandle_t handle) {
    GetPlan(op_graph.getFeatureVector(), plan, workspace_size, handle);
  }

  void InsertPlan(const cudnn_frontend::OperationGraph &op_graph,
                  const cudnn_frontend::ExecutionPlan &plan,
                  cudnnHandle_t handle) {
    InsertPlan(op_graph.getFeatureVector(), plan, handle);
  }

  bool IsStable(const cudnn_frontend::OperationGraph &op_graph,
                const std::string &tag,
                cudnnHandle_t handle) {
    return IsStable(op_graph.getFeatureVector(), tag, handle);
  }

 private:
  cudnn_frontend::feature_vector_t GetExtendedFeature(
      cudnn_frontend::feature_vector_t feat, cudnnHandle_t handle) {
    int64_t val = 0;
    memcpy(&val, &handle, sizeof(int64_t));
    feat.push_back(val);
    return feat;
  }
  using FeatureVectorToPlanMap =
      std::map<cudnn_frontend::feature_vector_t, cudnn_frontend::ExecutionPlan>;
  std::map<std::size_t, FeatureVectorToPlanMap> map_;
  std::hash<std::thread::id> hasher;

  std::shared_ptr<std::mutex> cache_mutex_;
  int saturation_count_;

  using SaturationTracker =
      std::map<std::pair<cudnn_frontend::feature_vector_t, std::string>, int>;
  std::map<std::size_t, SaturationTracker> tracker_;

  int64_t cache_hits_{0};
  int64_t cache_misses_{0};
};  // class CudnnFrontendPlanCache

}  // namespace autotune
}  // namespace phi
