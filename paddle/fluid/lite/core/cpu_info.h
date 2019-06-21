// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdarg>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/lite_tensor.h"
#include "paddle/fluid/lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM

typedef enum {
  LITE_POWER_HIGH = 0,
  LITE_POWER_LOW = 1,
  LITE_POWER_FULL = 2,
  LITE_POWER_NO_BIND = 3,
  LITE_POWER_RAND_HIGH = 4,
  LITE_POWER_RAND_LOW = 5
} PowerMode;

typedef enum {
  kAPPLE = 0,
  kA53 = 53,
  kA55 = 55,
  kA57 = 57,
  kA72 = 72,
  kA73 = 73,
  kA75 = 75,
  kA76 = 76,
  kARMArch_UNKOWN = -1
} ARMArch;

class DeviceInfo {
 public:
  static DeviceInfo& Global() {
    static auto* x = new DeviceInfo;
    return *x;
  }

  static int Init() {
    static int ret = Global().Setup();
    return ret;
  }

  int Setup();

  void SetRunMode(PowerMode mode, int thread_num);
  void SetCache(int l1size, int l2size, int l3size);
  void SetArch(ARMArch arch) { arch_ = arch; }

  PowerMode mode() const { return mode_; }
  int threads() const { return active_ids_.size(); }
  ARMArch arch() const { return arch_; }
  int l1_cache_size() const { return L1_cache_[active_ids_[0]]; }
  int l2_cache_size() const { return L2_cache_[active_ids_[0]]; }
  int l3_cache_size() const { return L3_cache_[active_ids_[0]]; }

  template <typename T>
  T* workspace_data() {
    return workspace_.mutable_data<T>();
  }
  bool ExtendWorkspace(DDimLite dims);

 private:
  int core_num_;
  std::vector<int> max_freqs_;
  std::vector<int> min_freqs_;
  int mem_size_;
  std::string dev_name_;

  std::vector<int> L1_cache_;
  std::vector<int> L2_cache_;
  std::vector<int> L3_cache_;
  std::vector<int> core_ids_;
  std::vector<int> big_core_ids_;
  std::vector<int> little_core_ids_;
  std::vector<int> cluster_ids_;
  std::vector<ARMArch> archs_;

  ARMArch arch_;
  // LITE_POWER_HIGH stands for using big cores,
  // LITE_POWER_LOW stands for using small core,
  // LITE_POWER_FULL stands for using all cores
  PowerMode mode_;
  std::vector<int> active_ids_;
  TensorLite workspace_;
  int64_t count_{0};

  void SetCacheInfo(int cache_id, int argc, ...);
  void SetArchInfo(int argc, ...);
  bool SetCPUInfoByName();
  void SetCPUInfoByProb();
  void RequestPowerFullMode(const int thread_num);
  void RequestPowerHighMode(const int thread_num);
  void RequestPowerLowMode(const int thread_num);
  void RequestPowerNoBindMode(const int thread_num);
  void RequestPowerRandHighMode(const int shift_num, const int thread_num);
  void RequestPowerRandLowMode(const int shift_num, const int thread_num);

  DeviceInfo() = default;
};

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
