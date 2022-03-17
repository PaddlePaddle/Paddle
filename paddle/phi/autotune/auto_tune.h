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

namespace phi {

class AutoTuneResult {
  //
};

class AutoTunerBase {
 public:
  template <typename Context, typename Callback>
  AutoTuneResult PickBestAlgorithm(Context ctx,
                                   Callback&& kernels,
                                   bool need_workspace = false) {
    if (need_workspace) {
      // 查询可用显存信息
    }
    float min_time = -1.f;
    for (&kernel : kernels) {
      if (need_workspace) {
        /* TODO: Currently, only conv need workspace and workspace query had
           beed added into the conv, if more op need workspace, this function
           will be achieved. */
      }
      // kernel_time = RunKernelSync(ctx, kernel);
      if (min_time > 0 && kernel_time < min_time) {
        min_time = kernel_time;
        selected_kernel = kernel;
      }
    }
  }

  template <typename Context, typename Callback>
  double RunKernelSync(Context ctx, Callback&& kernel_func) {
    /*
    ctx.Wait();
    time_start = profiler.start();
    kernel_func(...);
    time_end = profiler.end();
    ctx.Wait();
    return time_end - time_start;
    */
  }
}

/* This class is the main control abstraction of op auto-tune, to judge whether
   the op needed auto-tune, and choose out the best performance kernel
   implement.
   The main function of this class is below :
        1. Judge whether the op need auto-tune.
        2. Tunning the op with different kernels
        3. Choose out and cache the best kernel implement.
*/
class AutoTunerMap {
 public:
  AutoTunerMap& Instance() {
    static AutoTunerMap op_tune_map;
    return op_tune_map;
  }

  AutoTuneResult OpTuner(const std::string& op_type) {
    auto need_tune = Has(op_type);
    if (need_tune) {
      auto iter = tune_map.find(op_type);
      auto op_tuner = &iter->second;
      // return op_tuner::PickBestAlgorithm();
    }
  }

 private:
  bool Has(const std::string& op_type) const {
    return tune_map.find(op_type) != tune_map.end();
  }

  const AutoTunerBase& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
  }

  std::unordered_map<std::string, AutoTunerBase> tune_map;
  size_t cached_counts;   // recording the total cached count in one iter
  float cached_hit_rate;  //
  size_t iter_idx;        // recording the iter no.
};

}  // namespace phi
