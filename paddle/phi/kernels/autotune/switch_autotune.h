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

  // EnableAutoTune and DisableAutoTune should be used for debug only.
  void EnableAutoTune();
  void DisableAutoTune();

  void Update();

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
    static int64_t last_step_id = -2;

    if (last_step_id != current_steps_id_) {
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
      current_step_hit_rate_ = step_hit_rate;
      last_step_id = current_steps_id_;
    }
    return current_step_hit_rate_;
  }

  void SetAutoTuneRange(int64_t start, int64_t stop) {
    start_step_id_ = start;
    stop_step_id_ = stop;
  }

 private:
  AutoTuneStatus() = default;

  void Init() {
    use_autotune_ = false;
    current_steps_id_ = -1;
    previous_hits_ = 0;
    previous_misses_ = 0;
    step_hit_rates_.clear();
    AutoTuneCache::Instance().Clean();
  }

  bool use_autotune_{false};
  int64_t start_step_id_{1};
  int64_t stop_step_id_{10};
  int64_t current_steps_id_{-1};
  int64_t previous_hits_{0};
  int64_t previous_misses_{0};
  float current_step_hit_rate_{0.f};
  std::vector<float> step_hit_rates_;
};

}  // namespace autotune
}  // namespace phi
