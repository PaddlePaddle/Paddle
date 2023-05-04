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

  bool FindPlan(const cudnn_frontend::feature_vector_t &feature) {
    bool ret = false;
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    if (local_map.count(feature) > 0) {
      cache_hits_++;
      ret = true;
    } else {
      cache_misses_++;
    }
    return ret;
  }

  void GetPlan(const cudnn_frontend::feature_vector_t &feature,
               const cudnn_frontend::ExecutionPlan **plan,
               int64_t *workspace_size) {
    // Note(tizheng): CUDNNv8 execution plan is not thread-safe.
    // A shared plan being executed by different threads with the
    // same CUDNN handle is generally not safe (for now).
    // But somehow having thread-local copies of plans work fine,
    // even with the same handle.
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];

    auto it = local_map.find(feature);
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
                  const cudnn_frontend::ExecutionPlan &plan) {
    VLOG(4) << "[cudnn_frontend] cache: Insert plan: " << plan.getTag();
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    local_map.insert(std::make_pair(feature, plan));
  }

  bool IsStable(const cudnn_frontend::feature_vector_t &feature,
                const std::string &tag) {
    if (saturation_count_ == 1) {
      return true;
    }
    std::lock_guard<std::mutex> lock(*cache_mutex_);
    auto &local_map = map_[hasher(std::this_thread::get_id())];
    auto &local_tracker = tracker_[hasher(std::this_thread::get_id())];
    if (local_map.count(feature)) {
      return false;
    }
    int cnt = local_tracker[std::make_pair(feature, tag)] += 1;
    VLOG(4) << "[cudnn_frontend] SaturationTracker: " << tag << " " << cnt;
    return cnt >= saturation_count_;
  }

  bool FindPlan(const cudnn_frontend::OperationGraph &op_graph) {
    return FindPlan(op_graph.getFeatureVector());
  }

  void GetPlan(const cudnn_frontend::OperationGraph &op_graph,
               const cudnn_frontend::ExecutionPlan **plan,
               int64_t *workspace_size) {
    GetPlan(op_graph.getFeatureVector(), plan, workspace_size);
  }

  void InsertPlan(const cudnn_frontend::OperationGraph &op_graph,
                  const cudnn_frontend::ExecutionPlan &plan) {
    InsertPlan(op_graph.getFeatureVector(), plan);
  }

  bool IsStable(const cudnn_frontend::OperationGraph &op_graph,
                const std::string &tag) {
    return IsStable(op_graph.getFeatureVector(), tag);
  }

 private:
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

template <typename T>
inline void BuildFeatureVectorSingle(cudnn_frontend::feature_vector_t *v,
                                     const T &value) {
  v->push_back(static_cast<int64_t>(value));
}

template <>
inline void BuildFeatureVectorSingle(cudnn_frontend::feature_vector_t *v,
                                     const float &value) {
  int64_t val = 0;
  memcpy(&val, &value, sizeof(float));
  v->push_back(val);
}

template <>
inline void BuildFeatureVectorSingle<std::vector<int64_t>>(
    cudnn_frontend::feature_vector_t *v, const std::vector<int64_t> &value) {
  v->insert(v->end(), value.begin(), value.end());
}

template <>
inline void BuildFeatureVectorSingle<std::vector<int>>(
    cudnn_frontend::feature_vector_t *v, const std::vector<int> &value) {
  for (auto &val : value) {
    v->push_back(static_cast<int64_t>(val));
  }
}

template <>
inline void BuildFeatureVectorSingle<std::string>(
    cudnn_frontend::feature_vector_t *v, const std::string &value) {
  v->push_back(std::hash<std::string>()(value));
}

inline void BuildFeatureVector(cudnn_frontend::feature_vector_t *v) { return; }

template <typename T, typename... Args>
inline void BuildFeatureVector(cudnn_frontend::feature_vector_t *v,
                               const T &value,
                               Args... args) {
  BuildFeatureVectorSingle(v, value);
  BuildFeatureVector(v, args...);
}

}  // namespace autotune
}  // namespace phi
