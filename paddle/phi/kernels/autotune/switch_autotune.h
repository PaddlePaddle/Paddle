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
#include "glog/logging.h"
#include "paddle/phi/kernels/autotune/cache.h"

namespace phi {
namespace autotune {

class SwitchAutoTune {
 public:
  static SwitchAutoTune& Instance() {
    static SwitchAutoTune switch_autotune;
    return switch_autotune;
  }

  bool UseAutoTune() { return enable_autotune_; }

  void UpdateAutoTuneStatus() {
    current_steps_id_ += 1;
    if (enable_autotune_ == false) {
      return;
    }

    if (current_steps_id_ < auto_tune_start_step_id_) {
      enable_autotune_ = false;
    } else if (current_steps_id_ >= auto_tune_start_step_id_ &&
               current_steps_id_ <
                   auto_tune_start_step_id_ + auto_tune_steps_) {
      enable_autotune_ = true;
      VLOG(3) << "Current Step ID: " << current_steps_id_ << " Cache Hit Rate: "
              << AutoTuneCache::Instance().AutoTuneCacheHitRate()
              << " Cache Size: " << AutoTuneCache::Instance().Size();
    } else {
      enable_autotune_ = false;
    }
  }

  void SetAutoTuneSteps(int steps) { auto_tune_steps_ = steps; }

  void EnableAutoTune() { enable_autotune_ = true; }

  void DisableAutoTune() { enable_autotune_ = false; }

  int64_t StepID() { return current_steps_id_; }

 private:
  SwitchAutoTune() = default;
  int64_t auto_tune_start_step_id_ = 0;
  int64_t auto_tune_steps_ = 10;
  int64_t current_steps_id_ = -1;
  bool enable_autotune_ = true;
};

}  // namespace autotune
}  // namespace phi
