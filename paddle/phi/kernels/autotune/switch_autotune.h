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
#include <cmath>
#include <mutex>
#include <numeric>
#include "glog/logging.h"
#include "paddle/phi/kernels/autotune/cache.h"

namespace phi {
namespace autotune {

class AutoTuneStatus {
 public:
  static AutoTuneStatus& Instance() {
    static AutoTuneStatus switch_autotune;
    return switch_autotune;
  }

  bool UseAutoTune() { return use_autotune_; }

  // EnableAutoTune and DisableAutoTune Should be used for debug only.
  void EnableAutoTune() {
    use_autotune_ = true;
    Init();
  }

  void DisableAutoTune() {
    use_autotune_ = false;
    Init();
  }

  void Update() {
    current_steps_id_ += 1;

    if (!use_autotune_ && !update_use_autotune_) {
      return;
    }

    if (current_steps_id_ < start_step_id_) {
      use_autotune_ = false;
    } else if (current_steps_id_ >= start_step_id_ &&
               current_steps_id_ < stop_step_id_) {
      use_autotune_ = true;
      AutoTuneCache::Instance().UpdateStatus();
      step_hit_rates_.push_back(StepHitRate());
      VLOG(3) << "Step ID " << current_steps_id_
              << ", Accumulative Cache Hit Rate: "
              << AutoTuneCache::Instance().CacheHitRate()
              << ", Cache Size: " << AutoTuneCache::Instance().Size()
              << ", Current Step Hit Rate: " << StepHitRate();
    } else if (current_steps_id_ == stop_step_id_) {
      use_autotune_ = false;
      update_use_autotune_ = false;
      // clean cache according miss rate
      float miss_rate = static_cast<float>(1) - RecentHitRate();
      AutoTuneCache::Instance().Clean(miss_rate);
      VLOG(3) << "Recent Miss Rate: " << miss_rate;
    }
  }

  int64_t StepID() { return current_steps_id_; }

  float RecentHitRate() {
    int recent_step_nums = std::ceil(step_hit_rates_.size() * 0.3);
    float sum = std::accumulate(step_hit_rates_.rbegin(),
                                step_hit_rates_.rbegin() + recent_step_nums,
                                0.0);
    float mean = sum / recent_step_nums;
    return mean;
  }

  // Hit Rate of Current Step
  float StepHitRate() {
    int64_t current_hits = AutoTuneCache::Instance().CacheHits();
    int64_t current_misses = AutoTuneCache::Instance().CacheMisses();
    int64_t step_hits_ = current_hits - previous_hits_;
    int64_t step_misses_ = current_misses - previous_misses_;
    float step_hit_rate = 0.;
    int64_t step_num_accesses = step_hits_ + step_misses_;
    if (step_num_accesses != 0) {
      step_hit_rate = static_cast<float>(step_hits_) /
                      static_cast<float>(step_num_accesses);
    }
    previous_hits_ = current_hits;
    previous_misses_ = current_misses;
    return step_hit_rate;
  }

  void SetAutoTuneRange(int64_t start, int64_t stop) {
    start_step_id_ = start;
    stop_step_id_ = stop;
  }

 private:
  AutoTuneStatus() = default;

  void Init() {
    update_use_autotune_ = use_autotune_;
    current_steps_id_ = -1;
    previous_hits_ = 0;
    previous_misses_ = 0;
    step_hit_rates_.clear();
    AutoTuneCache::Instance().Clean(1.0);
  }

  int64_t start_step_id_ = 0;
  int64_t stop_step_id_ = 10;
  int64_t current_steps_id_ = -1;
  bool use_autotune_ = false;
  bool update_use_autotune_ = false;
  int64_t previous_hits_ = 0;
  int64_t previous_misses_ = 0;
  std::vector<float> step_hit_rates_;
};

}  // namespace autotune
}  // namespace phi
