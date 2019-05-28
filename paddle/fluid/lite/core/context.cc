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

#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/cpu_info.h"

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

void ARMContext::SetCache(int l1size, int l2size, int l3size) {
  DeviceInfo& dev = DeviceInfo::Global();
  int cpu_count = arm_get_cpucount();
  dev.L1_cache_.resize(cpu_count);
  dev.L2_cache_.resize(cpu_count);
  dev.L3_cache_.resize(cpu_count);
  for (int i = 0; i < cpu_count; ++i) {
    dev.L1_cache_[i] = l1size;
    dev.L2_cache_[i] = l2size;
    dev.L3_cache_[i] = l3size;
  }
  workspace_.Resize({2 * (l1size + l2size)});
}

ARMContext::ARMContext() {
  active_ids_ = {0};
  mode_ = LITE_POWER_HIGH;
  DeviceInfo& dev = DeviceInfo::Global();
  workspace_.Resize(
      {static_cast<int64_t>(dev.L2_cache_[active_ids_[0]] / sizeof(float))});
#ifdef TARGET_IOS
  arch_ = APPLE;  // use 6x8
#else
  if (dev.big_core_ids_.size() > 0) {
    arch_ = dev.archs_[dev.big_core_ids_[0]];
  }
#endif
}

PowerMode ARMContext::mode() const { return mode_; }

int ARMContext::threads() const { return active_ids_.size(); }

ARMContext::ARMContext(const ARMContext& ctx) {
  mode_ = ctx.mode_;
  active_ids_ = ctx.active_ids_;
  workspace_ = ctx.workspace_;
  arch_ = ctx.arch_;
  count_ = ctx.count_;
}

ARMContext& ARMContext::operator=(const ARMContext& ctx) {
  mode_ = ctx.mode_;
  active_ids_ = ctx.active_ids_;
  workspace_ = ctx.workspace_;
  arch_ = ctx.arch_;
  count_ = ctx.count_;
  return *this;
}

void ARMContext::BindDev() {
#ifdef USE_OPENMP
  int num_threads = active_ids_.size();
  omp_set_num_threads(num_threads);
#ifdef LITE_WITH_ANDROID
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
      LOGE("set cpu affinity failed, cpuID: %d\n", active_ids_[i]);
      return;
    }
  }
#endif  // LITE_WITH_ANDROID
#else   // USE_OPENMP
#ifdef LITE_WITH_ANDROID
  std::vector<int> cpuid1;
  cpuid1.push_back(active_ids_[0]);
  int ssaret = set_sched_affinity(cpuid1);
  if (ssaret != 0) {
    printf("set cpu affinity failed, cpuID: %d\n", active_ids_[0]);
    return;
  }
#endif  // LITE_WITH_ANDROID
#endif  // USE_OPENMP
}

void ARMContext::SetRunMode(PowerMode mode, int threads) {
  DeviceInfo& dev = DeviceInfo::Global();
  int big_core_size = dev.big_core_ids_.size();
  int small_core_size = dev.little_core_ids_.size();
  if (threads > big_core_size + small_core_size) {
    threads = big_core_size + small_core_size;
  }
#ifdef USE_OPENMP
  count_++;
  int shift_num = (count_ / 10) % big_core_size;
  switch (mode) {
    case LITE_POWER_FULL:
      mode_ = mode;
      active_ids_.clear();
      for (int i = 0; i < threads; ++i) {
        if (i < big_core_size) {
          active_ids_.push_back(dev.big_core_ids_[i]);
        } else {
          active_ids_.push_back(dev.little_core_ids_[i - big_core_size]);
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
          LOGE("threads: %d, exceed the big cores size: %d\n", threads,
               big_core_size);
          active_ids_ = dev.big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.big_core_ids_[i]);
          }
        }
      } else {
        mode_ = LITE_POWER_LOW;
        LOGE("HIGH POWER MODE is not support, switch to little cores\n");
        if (threads > small_core_size) {
          active_ids_ = dev.little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.little_core_ids_[i]);
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
          LOGW("threads: %d, exceed the little cores size: %d\n", threads,
               small_core_size);
          active_ids_ = dev.little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.little_core_ids_[i]);
          }
        }
      } else {
        mode_ = LITE_POWER_HIGH;
        LOGW("LOW POWER MODE is not support, switch to big cores\n");
        if (threads > big_core_size) {
          active_ids_ = dev.big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.big_core_ids_[i]);
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
      if (threads > dev.core_ids_.size()) {
        active_ids_.resize(dev.core_ids_.size());
      } else {
        active_ids_.resize(threads);
      }
      break;
    case LITE_POWER_RAND_HIGH:
      active_ids_.clear();
      if (big_core_size > 0) {
        mode_ = LITE_POWER_RAND_HIGH;
        if (threads > big_core_size) {
          LOGW("threads: %d, exceed the big cores size: %d\n", threads,
               big_core_size);
          active_ids_ = dev.big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(
                dev.big_core_ids_[(i + shift_num) % big_core_size]);
          }
        }
      } else {
        mode_ = LITE_POWER_LOW;
        LOGW("HIGH POWER MODE is not support, switch to little cores\n");
        if (threads > small_core_size) {
          active_ids_ = dev.little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.little_core_ids_[i]);
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
          LOGW("threads: %d, exceed the little cores size: %d\n", threads,
               small_core_size);
          active_ids_ = dev.little_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(
                dev.little_core_ids_[(i + shift_num) % small_core_size]);
          }
        }
      } else {
        mode_ = LITE_POWER_HIGH;
        LOGW("LOW POWER MODE is not support, switch to big cores\n");
        if (threads > big_core_size) {
          active_ids_ = dev.big_core_ids_;
        } else {
          for (int i = 0; i < threads; ++i) {
            active_ids_.push_back(dev.big_core_ids_[i]);
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
      LOG(ERROR) << "core id " << active_ids_[0]
                 << " is offline, switch to NO BIND MODE";
      int threads = active_ids_.size();
      omp_set_num_threads(threads);
    }
  }
#else
  if (big_core_size > 0) {
    active_ids_ = {dev.big_core_ids_[0]};
  } else {
    active_ids_ = {0};
  }
#endif
  //! alloc memory for sgemm in this context
  int temp_mem_size =
      DeviceInfo::Global().L2_cache_[active_ids_[0]] / sizeof(float);
  workspace_.Resize({temp_mem_size});
  arch_ = DeviceInfo::Global().archs_[active_ids_[0]];
}

ARMArch ARMContext::arch() const { return arch_; }

void ARMContext::SetArch(ARMArch arch) { arch_ = arch; }

int ARMContext::l1_cache_size() const {
  DeviceInfo& dev = DeviceInfo::Global();
  return dev.L1_cache_[active_ids_[0]];
}

int ARMContext::l2_cache_size() const {
  DeviceInfo& dev = DeviceInfo::Global();
  return dev.L2_cache_[active_ids_[0]];
}

int ARMContext::l3_cache_size() const {
  DeviceInfo& dev = DeviceInfo::Global();
  return dev.L3_cache_[active_ids_[0]];
}

bool ARMContext::ExtendWorkspace(DDimLite dims) {
  auto count = dims.product();
  auto old = workspace_.dims();
  if (count == old.product()) {
    return false;
  }

  workspace_.Resize(
      {static_cast<int64_t>(count + l2_cache_size() / sizeof(float))});
  return true;
}
#endif  // LITE_WITH_ARM
}  // namespace lite
}  // namespace paddle
