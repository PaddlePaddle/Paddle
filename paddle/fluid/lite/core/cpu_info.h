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

#include <string>
#include <vector>
#include "paddle/fluid/lite/utils/cp_logging.h"

#ifdef LITE_WITH_ANDROID
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // TARGET_OS_IPHONE
#endif  // __APPLE__

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
  int idx_;
  int max_freq_;
  int min_freq_;
  int generate_arch_;
  int compute_core_num_;
  int max_memory_;
  int sharemem_size_;

  std::string device_name_;
  std::string compute_ability_;

  std::vector<int> L1_cache_;
  std::vector<int> L2_cache_;
  std::vector<int> L3_cache_;
  std::vector<int> core_ids_;
  std::vector<int> big_core_ids_;
  std::vector<int> little_core_ids_;
  std::vector<int> cluster_ids_;
  std::vector<ARMArch> archs_;

  static DeviceInfo& Global() {
    static auto* x = new DeviceInfo;
    return *x;
  }

  static void Init() {
    auto& info = Global();
    InitInternal(&info);
  }

 private:
  DeviceInfo() = default;
  static void InitInternal(DeviceInfo* dev);
};

size_t arm_get_meminfo();

int arm_get_cpucount();

void arm_get_cpu_arch(std::vector<ARMArch>* archs);

bool get_cpu_info_from_name(DeviceInfo* cpu_info, std::string hardware_name);

#ifdef LITE_WITH_ANDROID

void set_default_cache(DeviceInfo* dev);

std::string arm_get_cpu_name();

int get_max_freq_khz(int cpuid);

int arm_sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>* cpuids,
                                    const std::vector<int>& cpu_freq,
                                    std::vector<int>* cluster_ids);
int check_online(const std::vector<int>& core_ids);
int set_sched_affinity(const std::vector<int>& cpuids);

#endif  // LITE_WITH_ANDROID

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
