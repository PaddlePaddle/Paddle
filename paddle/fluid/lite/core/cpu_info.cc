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

#ifdef LITE_WITH_LINUX
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

#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

#include <cstdarg>
#include "paddle/fluid/lite/core/cpu_info.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM

void DeviceInfo::InitInternal(DeviceInfo* dev) {
  set_default_cache(dev);
  dev->compute_core_num_ = arm_get_cpucount();
  dev->max_memory_ = arm_get_meminfo();

// get max freq
#ifdef LITE_WITH_LINUX
  std::vector<int> max_freq(dev->compute_core_num_);
  for (int i = 0; i < dev->compute_core_num_; ++i) {
    max_freq[i] = get_max_freq_khz(i) / 1000;
  }
  std::string cpu_name = arm_get_cpu_name();
  if (get_cpu_info_from_name(dev, cpu_name) != true) {
    arm_sort_cpuid_by_max_frequency(dev->compute_core_num_, &dev->core_ids_,
                                    max_freq, &dev->cluster_ids_);
    dev->big_core_ids_.clear();
    dev->little_core_ids_.clear();
    for (int i = 0; i < dev->cluster_ids_.size(); ++i) {
      if (dev->cluster_ids_[i] == 0) {
        dev->big_core_ids_.push_back(dev->core_ids_[i]);
      } else {
        dev->little_core_ids_.push_back(dev->core_ids_[i]);
      }
    }
    arm_get_cpu_arch(&dev->archs_);
  }

  LOG(INFO) << "ARM multiprocessors number: " << dev->compute_core_num_;
  for (int i = 0; i < dev->compute_core_num_; ++i) {
    LOG(INFO) << "ARM multiprocessors ID: " << dev->core_ids_[i]
              << ", frequence: " << max_freq[i]
              << ", cluster ID: " << dev->cluster_ids_[dev->core_ids_[i]]
              << ", CPU ARCH: A" << dev->archs_[i];
  }
  VLOG(1) << "L1 DataCache size is: ";
  for (int i = 0; i < dev->compute_core_num_; ++i) {
    VLOG(1) << dev->L1_cache_[i] / 1024 << " KB";
  }
  VLOG(1) << "L2 Cache size is: ";
  for (int i = 0; i < dev->compute_core_num_; ++i) {
    VLOG(1) << dev->L2_cache_[i] / 1024 << " KB";
  }
  VLOG(1) << "Total memory: " << dev->max_memory_ << "KB";

  dev->max_freq_ = max_freq[0];
  for (int j = 1; j < dev->compute_core_num_; ++j) {
    if (dev->max_freq_ < max_freq[j]) {
      dev->max_freq_ = max_freq[j];
    }
  }
#elif defined(TARGET_IOS)
  arm_get_cpu_arch(&dev->archs_);
#endif
  dev->active_ids_ = {0};
  dev->mode_ = LITE_POWER_HIGH;
  dev->workspace_.Resize({static_cast<int64_t>(
      dev->L2_cache_[dev->active_ids_[0]] / sizeof(float))});
#ifdef TARGET_IOS
  dev->arch_ = APPLE;  // use 6x8
#else
  if (dev->big_core_ids_.size() > 0) {
    dev->arch_ = dev->archs_[dev->big_core_ids_[0]];
  }
#endif
}

void DeviceInfo::SetCache(int l1size, int l2size, int l3size) {
  int cpu_count = arm_get_cpucount();
  L1_cache_.resize(cpu_count);
  L2_cache_.resize(cpu_count);
  L3_cache_.resize(cpu_count);
  for (int i = 0; i < cpu_count; ++i) {
    L1_cache_[i] = l1size;
    L2_cache_[i] = l2size;
    L3_cache_[i] = l3size;
  }
  workspace_.Resize({2 * (l1size + l2size)});
}

void DeviceInfo::BindDev() {
#ifdef ARM_WITH_OMP
  int num_threads = active_ids_.size();
  omp_set_num_threads(num_threads);
#ifdef LITE_WITH_LINUX
  std::vector<int> ssarets;
  for (int j = 0; j < num_threads; ++j) {
    ssarets.push_back(0);
  }
#pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    ssarets[i] = set_sched_affinity(active_ids_);
  }
  for (int i = 0; i < num_threads; i++) {
    if (ssarets[i] != 0) {
      LOG(ERROR) << "set cpu affinity failed, cpuID: " << active_ids_[i];
      return;
    }
  }
#endif  // LITE_WITH_LINUX
#else   // ARM_WITH_OMP
#ifdef LITE_WITH_LINUX
  std::vector<int> cpuid1;
  cpuid1.push_back(active_ids_[0]);
  int ssaret = set_sched_affinity(cpuid1);
  if (ssaret != 0) {
    printf("set cpu affinity failed, cpuID: %d\n", active_ids_[0]);
    return;
  }
#endif  // LITE_WITH_LINUX
#endif  // ARM_WITH_OMP
}

void DeviceInfo::SetRunMode(PowerMode mode, int threads) {
  LOG(INFO) << "ARM SetRunMode called";
  int big_core_size = big_core_ids_.size();
  int small_core_size = little_core_ids_.size();
  if (threads > big_core_size + small_core_size) {
    threads = big_core_size + small_core_size;
  }
#ifdef ARM_WITH_OMP
  count_++;
  int shift_num = (count_ / 10) % big_core_size;
  switch (mode) {
    case LITE_POWER_FULL:
      mode_ = mode;
      active_ids_.clear();
      for (int i = 0; i < threads; ++i) {
        if (i < big_core_size) {
          active_ids_.push_back(big_core_ids_[i]);
        } else {
          active_ids_.push_back(little_core_ids_[i - big_core_size]);
        }
      }
      if (active_ids_.size() == 0) {
        active_ids_.push_back(0);
      }
      break;
    case LITE_POWER_HIGH:
      active_ids_.clear();
      if (big_core_size > 0) {
        mode_ = LITE_POWER_HIGH;
        if (threads > big_core_size) {
          LOG(ERROR) << "threads: " << threads
                     << ", exceed the big cores size: " << big_core_size;
          active_ids_ = big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(big_core_ids_[i]);
          }
        }
      } else {
        mode_ = LITE_POWER_LOW;
        LOG(ERROR) << "HIGH POWER MODE is not support, switch to little cores.";
        if (threads > small_core_size) {
          active_ids_ = little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(little_core_ids_[i]);
          }
        }
      }
      if (active_ids_.size() == 0) {
        active_ids_.push_back(0);
      }
      break;
    case LITE_POWER_LOW:
      active_ids_.clear();
      if (small_core_size > 0) {
        mode_ = LITE_POWER_LOW;
        if (threads > small_core_size) {
          LOG(WARNING) << "threads: " << threads
                       << ", exceed the little cores size: " << small_core_size;
          active_ids_ = little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(little_core_ids_[i]);
          }
        }
      } else {
        mode_ = LITE_POWER_HIGH;
        LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores";
        if (threads > big_core_size) {
          active_ids_ = big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(big_core_ids_[i]);
          }
        }
      }
      if (active_ids_.size() == 0) {
        active_ids_.push_back(0);
      }
      break;
    case LITE_POWER_NO_BIND:
      mode_ = LITE_POWER_NO_BIND;
      active_ids_.clear();
      if (threads > core_ids_.size()) {
        active_ids_.resize(core_ids_.size());
      } else {
        active_ids_.resize(threads);
      }
      break;
    case LITE_POWER_RAND_HIGH:
      active_ids_.clear();
      if (big_core_size > 0) {
        mode_ = LITE_POWER_RAND_HIGH;
        if (threads > big_core_size) {
          LOG(WARNING) << "threads: " << threads
                       << ", exceed the big cores size: " << big_core_size;
          active_ids_ = big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(
                big_core_ids_[(i + shift_num) % big_core_size]);
          }
        }
      } else {
        mode_ = LITE_POWER_LOW;
        LOG(WARNING)
            << "HIGH POWER MODE is not support, switch to little cores.";
        if (threads > small_core_size) {
          active_ids_ = little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(little_core_ids_[i]);
          }
        }
      }
      if (active_ids_.size() == 0) {
        active_ids_.push_back(0);
      }
      break;
    case LITE_POWER_RAND_LOW:
      active_ids_.clear();
      if (small_core_size > 0) {
        mode_ = LITE_POWER_RAND_LOW;
        if (threads > small_core_size) {
          LOG(WARNING) << "threads: " << threads
                       << ", exceed the little cores size: " << small_core_size;
          active_ids_ = little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(
                little_core_ids_[(i + shift_num) % small_core_size]);
          }
        }
      } else {
        mode_ = LITE_POWER_HIGH;
        LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores.";
        if (threads > big_core_size) {
          active_ids_ = big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(big_core_ids_[i]);
          }
        }
      }
      if (active_ids_.size() == 0) {
        active_ids_.push_back(0);
      }
      break;
  }
  //! fix multi-threads LITE_POWER_HIGH mode
  if (mode_ == LITE_POWER_NO_BIND || threads > 1) {
    int threads = active_ids_.size();
    omp_set_num_threads(threads);
  } else {
    if (check_online(active_ids_)) {
      BindDev();
    } else {
      LOG(WARNING) << "core id " << active_ids_[0]
                   << " is offline, switch to NO BIND MODE";
      int threads = active_ids_.size();
      omp_set_num_threads(threads);
    }
  }
#else
  if (big_core_size > 0) {
    active_ids_ = {big_core_ids_[0]};
  } else {
    active_ids_ = {0};
  }
#endif
  //! alloc memory for sgemm in this context
  int temp_mem_size = L2_cache_[active_ids_[0]] / sizeof(float);
  workspace_.Resize({temp_mem_size});
  arch_ = archs_[active_ids_[0]];
}

bool DeviceInfo::ExtendWorkspace(DDimLite dims) {
  auto count = dims.product();
  auto old = workspace_.dims();
  if (count == old.product()) {
    return false;
  }

  workspace_.Resize({static_cast<int64_t>(
      count + L2_cache_[active_ids_[0]] / sizeof(float))});
  return true;
}

// cache_id : 0 -> L1, 1 -> L2, 2 -> L3
void set_cache_info(DeviceInfo* cpu_info, int cache_id, int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  std::vector<int>* cache;
  switch (cache_id) {
    case 0:
      cache = &cpu_info->L1_cache_;
      break;
    case 1:
      cache = &cpu_info->L2_cache_;
      break;
    case 2:
      cache = &cpu_info->L3_cache_;
      break;
    default:
      break;
  }
  int core_num = cpu_info->compute_core_num_;
  cache->resize(core_num);
  if (argc == 1) {
    int cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < core_num; ++i) {
      (*cache)[i] = cache_size;
    }
  } else {
    int big_core_num = cpu_info->big_core_ids_.size();
    int little_core_num = cpu_info->little_core_ids_.size();
    int big_core_cache_size = va_arg(arg_ptr, int);
    int little_core_cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < big_core_num; ++i) {
      (*cache)[cpu_info->big_core_ids_[i]] = big_core_cache_size;
    }
    for (int i = 0; i < little_core_num; ++i) {
      (*cache)[cpu_info->little_core_ids_[i]] = little_core_cache_size;
    }
  }
  va_end(arg_ptr);
}

void set_arch_info(DeviceInfo* cpu_info, int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  int core_num = cpu_info->compute_core_num_;
  cpu_info->archs_.resize(core_num);
  if (argc == 1) {
    ARMArch arch = (ARMArch)va_arg(arg_ptr, int);
    for (int i = 0; i < core_num; ++i) {
      cpu_info->archs_[i] = arch;
    }
  } else {
    ARMArch big_core_arch = (ARMArch)va_arg(arg_ptr, int);
    ARMArch little_core_arch = (ARMArch)va_arg(arg_ptr, int);
    int big_core_num = cpu_info->big_core_ids_.size();
    int little_core_num = cpu_info->little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      cpu_info->archs_[cpu_info->big_core_ids_[i]] = big_core_arch;
    }
    for (int i = 0; i < little_core_num; ++i) {
      cpu_info->archs_[cpu_info->little_core_ids_[i]] = little_core_arch;
    }
  }
  va_end(arg_ptr);
}

bool get_cpu_info_from_name(DeviceInfo* cpu_info, std::string hardware_name) {
  /* Snapdragon */
  if (hardware_name.find("SDM845") != std::string::npos) {  // 845
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 2, kA75, kA55);
    set_cache_info(cpu_info, 0, 1, 32 * 1024);
    set_cache_info(cpu_info, 1, 2, 256 * 1024, 128 * 1024);
    set_cache_info(cpu_info, 2, 1, 2048 * 1024);
    return true;

  } else if (hardware_name.find("SDM710") != std::string::npos) {  // 710
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3, 4, 5};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 1, 1, 0, 0};
    set_arch_info(cpu_info, 2, kA75, kA55);
    return true;
  } else if (hardware_name.find("MSM8998") != std::string::npos) {  // 835
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 2, kA73, kA53);
    set_cache_info(cpu_info, 0, 2, 64 * 1024);
    set_cache_info(cpu_info, 1, 2, 1024 * 1024,
                   /*real cache size is 2M, while that will get bad performace
                      on conv3x3s1 or gemm, set to 1M or 512K*/
                   1024 * 1024);
    return true;

  } else if (hardware_name.find("MSM8996") != std::string::npos) {  // 820
    cpu_info->compute_core_num_ = 4;
    cpu_info->core_ids_ = {0, 1, 2, 3};
    cpu_info->big_core_ids_ = {2, 3};
    cpu_info->little_core_ids_ = {0, 1};
    cpu_info->cluster_ids_ = {1, 1, 0, 0};
    set_arch_info(cpu_info, 1, kA72);
    set_cache_info(cpu_info, 0, 1, 24 * 1024);
    set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
    return true;

  } else if (hardware_name.find("SDM660") != std::string::npos ||
             hardware_name.find("SDM636") != std::string::npos) {  // 660, 636
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 1, kA73);
    set_cache_info(cpu_info, 0, 2, 64 * 1024, 32 * 1024);
    set_cache_info(cpu_info, 1, 1, 1024 * 1024);
    return true;

  } else if (hardware_name.find("MSM8976") != std::string::npos) {  // 652,653
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 2, kA72, kA53);
    set_cache_info(cpu_info, 0, 1, 32 * 1024);
    set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
    return true;

  } else if (hardware_name.find("MSM8953") != std::string::npos) {  // 625
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->little_core_ids_ = {};
    cpu_info->cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    set_arch_info(cpu_info, 1, kA53);
    set_cache_info(cpu_info, 0, 1, 32 * 1024);
    set_cache_info(cpu_info, 1, 1, 1024 * 1024);
    return true;

  } else if (hardware_name.find("MSM8939") != std::string::npos) {  // 615
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {0, 1, 2, 3};
    cpu_info->little_core_ids_ = {4, 5, 6, 7};
    cpu_info->cluster_ids_ = {0, 0, 0, 0, 1, 1, 1, 1};
    set_arch_info(cpu_info, 1, kA53);
    set_cache_info(cpu_info, 0, 1, 32 * 1024);
    set_cache_info(cpu_info, 1, 2, 512 * 1024, 256 * 1024);
    return true;

    /* MediaTek */

  } else if (hardware_name.find("MT6797") !=
             std::string::npos) {  // X20/X23/X25/X27
    cpu_info->compute_core_num_ = 10;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cpu_info->big_core_ids_ = {8, 9};
    cpu_info->little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    set_arch_info(cpu_info, 2, kA72, kA53);
    set_cache_info(cpu_info, 0, 1, 32 * 1024);
    set_cache_info(cpu_info, 1, 2, 1024 * 1024, 512 * 1024);
    return true;

  } else if (hardware_name.find("MT6799") != std::string::npos) {  // X30
    cpu_info->compute_core_num_ = 10;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cpu_info->big_core_ids_ = {8, 9};
    cpu_info->little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    set_arch_info(cpu_info, 2, kA73, kA53);
    return true;

  } else if (hardware_name.find("MT6795") != std::string::npos ||
             hardware_name.find("MT6762") != std::string::npos ||
             hardware_name.find("MT6755T") != std::string::npos ||
             hardware_name.find("MT6755S") != std::string::npos ||
             hardware_name.find("MT6753") != std::string::npos ||
             hardware_name.find("MT6752") != std::string::npos ||
             hardware_name.find("MT6750") != std::string::npos) {
    // X10, P22, P15/P18, MT6753, MT6752/MT6752M, MT6750
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->little_core_ids_ = {};
    cpu_info->cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    set_arch_info(cpu_info, 1, kA53);
    return true;

  } else if (hardware_name.find("MT6758") != std::string::npos ||
             hardware_name.find("MT6757") != std::string::npos ||
             hardware_name.find("MT6763") != std::string::npos ||
             hardware_name.find("MT6755M") != std::string::npos ||
             hardware_name.find("MT6755") !=
                 std::string::npos) {  // P30, P20/P25, P23, P10
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 1, kA53);
    return true;

  } else if (hardware_name.find("MT6771") != std::string::npos) {  // P60
    cpu_info->compute_core_num_ = 8;
    cpu_info->core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cpu_info->big_core_ids_ = {4, 5, 6, 7};
    cpu_info->little_core_ids_ = {0, 1, 2, 3};
    cpu_info->cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    set_arch_info(cpu_info, 2, kA73, kA53);
    return true;

  } else if (hardware_name.find("MT6765") != std::string::npos ||
             hardware_name.find("MT6739") != std::string::npos ||
             hardware_name.find("MT6738") != std::string::npos ||
             hardware_name.find("MT6737") !=
                 std::string::npos) {  // A22, MT6739, MT6738, MT6767
    cpu_info->compute_core_num_ = 4;
    cpu_info->core_ids_ = {0, 1, 2, 3};
    cpu_info->big_core_ids_ = {0, 0, 0, 0};
    cpu_info->little_core_ids_ = {};
    cpu_info->cluster_ids_ = {0, 0, 0, 0};
    set_arch_info(cpu_info, 1, kA53);
    return true;
  }
  return false;
}

size_t arm_get_meminfo() {
#ifdef LITE_WITH_LINUX
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/meminfo", "rb");
  if (!fp) {
    return 1;
  }

  size_t memsize = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    sscanf(s, "MemTotal:        %d kB", &memsize);
  }

  fclose(fp);

  return memsize;
#elif defined(TARGET_IOS)
  // to be implemented
  printf("not implemented\n");
  return 0;
#endif
}

int arm_get_cpucount() {
#ifdef LITE_WITH_LINUX
  // get cpu count from /sys/devices/system/cpu/cpunum/uevent
  int max_cpu_count = 20;
  int count = 0;
  for (int i = 0; i < max_cpu_count; ++i) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", i);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
      break;
    }
    count++;
    fclose(fp);
  }
  if (count < 1) {
    count = 1;
  }
  return count;
#elif defined(TARGET_IOS)
  int count = 0;
  size_t len = sizeof(count);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
  if (count < 1) {
    count = 1;
  }
  return count;
#else
  return 1;
#endif
}

void arm_get_cpu_arch(std::vector<ARMArch>* archs) {
#ifdef LITE_WITH_LINUX
  archs->clear();
  //! get CPU ARCH
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return;
  }
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "part") != NULL) {
      int arch_id = 0;
      sscanf(s, "CPU part\t: %x", &arch_id);
      switch (arch_id) {
        case 0xd03:
          archs->push_back(kA53);
          break;
        case 0xd05:
          archs->push_back(kA55);
          break;
        case 0xd07:
          archs->push_back(kA57);
          break;
        case 0xd08:
          archs->push_back(kA72);
          break;
        case 0xd09:
          archs->push_back(kA73);
          break;
        case 0xd0a:
          archs->push_back(kA75);
          break;
        case 0x800:
          // 835
          archs->push_back(kA73);
          break;
        case 0x205:
          // 820
          archs->push_back(kA72);
          break;
        default:
          LOG(ERROR) << "unknow type";
          archs->push_back(kARMArch_UNKOWN);
      }
    }
  }
  fclose(fp);
  int cpu_count = arm_get_cpucount();
  if (archs->size() < cpu_count) {
    for (int i = archs->size(); i < cpu_count; ++i) {
      archs->push_back(archs->at(i - 1));
    }
  }
#endif
#ifdef TARGET_IOS
  int cpu_count = arm_get_cpucount();
  for (int i = 0; i < cpu_count; ++i) {
    archs->push_back(APPLE);
  }
#endif
}

#ifdef LITE_WITH_LINUX

void set_default_cache(DeviceInfo* dev) {
  int cpu_count = arm_get_cpucount();
  dev->L1_cache_.resize(cpu_count);
  dev->L2_cache_.resize(cpu_count);
  dev->L3_cache_.resize(cpu_count);
#ifdef TARGET_IOS
  for (int i = 0; i < cpu_count; ++i) {
    dev->L1_cache_[i] = 64 * 1024;
    dev->L2_cache_[i] = 2048 * 1024;
    dev->L3_cache_[i] = 0;
  }
#else
  for (int i = 0; i < cpu_count; ++i) {
    dev->L1_cache_[i] = 32 * 1024;
    dev->L2_cache_[i] = 512 * 1024;
    dev->L3_cache_[i] = 0;
  }
#endif
}
std::string arm_get_cpu_name() {
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return "";
  }
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "Hardware") != NULL) {
      fclose(fp);
      return std::string(line);
    }
  }
  fclose(fp);
  return "";
}

int get_max_freq_khz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

  FILE* fp = fopen(path, "rb");

  if (!fp) {
    // second try, for online cpu
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpuid);
    fp = fopen(path, "rb");

    if (!fp) {
      // third try, for online cpu
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
      fp = fopen(path, "rb");

      if (!fp) {
        return -1;
      }

      int max_freq_khz = -1;
      fscanf(fp, "%d", &max_freq_khz);

      fclose(fp);

      return max_freq_khz;
    }
  }

  int max_freq_khz = 0;
  while (!feof(fp)) {
    int freq_khz = 0;
    int nscan = fscanf(fp, "%d %*d", &freq_khz);
    if (nscan != 1) {
      break;
    }

    if (freq_khz > max_freq_khz) {
      max_freq_khz = freq_khz;
    }
  }

  fclose(fp);

  return max_freq_khz;
}

int arm_sort_cpuid_by_max_frequency(int cpu_count, std::vector<int>* cpuids,
                                    const std::vector<int>& cpu_freq,
                                    std::vector<int>* cluster_ids) {
  if (cpu_count == 0) {
    return 0;
  }

  cpuids->resize(cpu_count);
  cluster_ids->resize(cpu_count);

  for (int i = 0; i < cpu_count; i++) {
    cpuids->at(i) = i;
  }

  // sort cpuid as big core first
  // simple bubble sort

  for (int i = 0; i < cpu_count; i++) {
    for (int j = i + 1; j < cpu_count; j++) {
      if (cpu_freq[i] < cpu_freq[j]) {
        // swap
        int tmp = cpuids->at(i);
        cpuids->at(i) = cpuids->at(j);
        cpuids->at(j) = tmp;
      }
    }
  }
  // SMP
  int mid_max_freq_khz =
      (cpu_freq[cpuids->at(0)] + cpu_freq[cpuids->at(cpu_count - 1)]) / 2;

  for (int i = 0; i < cpu_count; i++) {
    cpuids->at(i) = i;
    if (cpu_freq[i] >= mid_max_freq_khz) {
      cluster_ids->at(i) = 0;
    } else {
      cluster_ids->at(i) = 1;
    }
  }
  return 0;
}

int check_online(const std::vector<int>& core_ids) {
  if (core_ids.size() == 0) {
    return 0;
  }
  char path[256];
  int online = 1;
  for (int i = 0; i < core_ids.size(); ++i) {
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online",
             core_ids[i]);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
      return 0;
    }
    int cur_online = 0;
    fscanf(fp, "%d", &cur_online);
    online &= cur_online;
    fclose(fp);
  }
  return online;
}

int set_sched_affinity(const std::vector<int>& cpuids) {
// #define CPU_SETSIZE 1024
// #define __NCPUBITS  (8 * sizeof (unsigned long))
// typedef struct
// {
//    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
// } cpu_set_t;

// set affinity for thread
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < cpuids.size(); i++) {
    CPU_SET(cpuids[i], &mask);
  }

  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    LOG(ERROR) << "syscall error " << syscallret;
    return -1;
  }

  return 0;
}

#endif  // LITE_WITH_LINUX

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
