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

#include "paddle/phi/kernels/autotune/switch_autotune.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_autotune);

namespace phi::autotune {

void AutoTuneStatus::EnableAutoTune() {
  FLAGS_use_autotune = true;
  Init();
}

void AutoTuneStatus::DisableAutoTune() {
  FLAGS_use_autotune = false;
  use_autotune_ = false;
  Init();
}

void AutoTuneStatus::Update() {
  current_steps_id_ += 1;
  if (!FLAGS_use_autotune) {
    return;
  }

  // This function is called when each iter finished.
  if (current_steps_id_ + 1 < start_step_id_) {
    use_autotune_ = false;
  } else if (current_steps_id_ + 1 >= start_step_id_ &&
             current_steps_id_ + 1 < stop_step_id_) {
    use_autotune_ = true;
    AutoTuneCache::Instance().UpdateStatus();
    step_hit_rates_.push_back(StepHitRate());
    VLOG(3) << "Step ID: " << current_steps_id_
            << ", Accumulative Cache Hit Rate: "
            << static_cast<int>(AutoTuneCache::Instance().CacheHitRate() * 100)
            << "%, Cache Size: " << AutoTuneCache::Instance().Size()
            << ", Current Step Hit Rate: "
            << static_cast<int>(StepHitRate() * 100) << "%";
  } else {
    use_autotune_ = false;
    // Set a small tolerance to avoid performance degradation
    // due to large cache size under dynamic shape.
    // TODO(limingshu): Currently works for conv op only, this
    // method shall be optimized when more ops involved in.
    // float miss_rate = static_cast<float>(1) - RecentHitRate();
    // if (current_steps_id_ == stop_step_id_) {
    //   AutoTuneCache::Instance().Clean(miss_rate);
    // }
    if (VLOG_IS_ON(4)) {
      AutoTuneCache::Instance().UpdateStatus();
      VLOG(4) << "Step ID: " << current_steps_id_ << ", Current Step Hit Rate: "
              << static_cast<int>(StepHitRate() * 100) << "%";
    }
  }
}

}  // namespace phi::autotune
