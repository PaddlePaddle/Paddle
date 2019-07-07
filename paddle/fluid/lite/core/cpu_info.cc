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

#include <algorithm>
#include <limits>
#include "paddle/fluid/lite/core/cpu_info.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM

#ifdef TARGET_IOS
const int DEFAULT_L1_CACHE_SIZE = 64 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#else
const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 512 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#endif

int get_cpu_num() {
#ifdef LITE_WITH_LINUX
  // get cpu count from /sys/devices/system/cpu/cpunum/uevent
  int max_cpu_num = 20;
  int cpu_num = 0;
  for (int i = 0; i < max_cpu_num; ++i) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", i);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
      break;
    }
    cpu_num++;
    fclose(fp);
  }
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#elif defined(TARGET_IOS)
  int cpu_num = 0;
  size_t len = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &len, NULL, 0);
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#else
  return 1;
#endif
}

size_t get_mem_size() {
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
#endif
  return 0;
}

void get_cpu_arch(std::vector<ARMArch>* archs, const int cpu_num) {
  archs->resize(cpu_num);
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = kARMArch_UNKOWN;
  }
#ifdef LITE_WITH_LINUX
  //! get CPU ARCH
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return;
  }
  int cpu_idx = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "part") != NULL) {
      ARMArch arch_type = kARMArch_UNKOWN;
      int arch_id = 0;
      sscanf(s, "CPU part\t: %x", &arch_id);
      switch (arch_id) {
        case 0xd03:
          arch_type = kA53;
          break;
        case 0xd05:
          arch_type = kA55;
          break;
        case 0xd07:
          arch_type = kA57;
          break;
        case 0xd08:
          arch_type = kA72;
          break;
        case 0xd09:
          arch_type = kA73;
          break;
        case 0xd0a:
          arch_type = kA75;
          break;
        case 0xd40:
          arch_type = kA76;
          break;
        case 0x804:
          // 855
          arch_type = kA76;
          break;
        case 0x805:
          // 855
          arch_type = kA55;
          break;
        case 0x802:
          // 845
          arch_type = kA75;
          break;
        case 0x803:
          // 845
          arch_type = kA55;
          break;
        case 0x801:
          // 835
          arch_type = kA73;
          break;
        case 0x800:
          // 835
          arch_type = kA73;
          break;
        case 0x205:
          // 820
          arch_type = kA72;
          break;
        default:
          LOG(ERROR) << "Unknow cpu arch: " << arch_id;
      }
      archs->at(cpu_idx) = arch_type;
      cpu_idx++;
    }
  }
  fclose(fp);
  for (; cpu_idx > 0 && cpu_idx < cpu_num; ++cpu_idx) {
    archs->at(cpu_idx) = archs->at(cpu_idx - 1);
  }
#elif defined(TARGET_IOS)
  for (int i = 0; i < cpu_num; ++i) {
    archs->at(i) = APPLE;
  }
#endif
}

#ifdef LITE_WITH_LINUX

std::string get_cpu_name() {
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

void get_cpu_max_min_freq(int cpu_id, int* max_freq, int* min_freq) {
  *max_freq = 0;
  *min_freq = 0;
  // first try, for all possible cpu
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpu_id);
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    // second try, for online cpu
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpu_id);
    fp = fopen(path, "rb");
    if (!fp) {
      // third try, for online cpu
      // get max_freq
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
               cpu_id);
      fp = fopen(path, "rb");
      if (!fp) {
        return;
      }
      fscanf(fp, "%d", max_freq);
      fclose(fp);
      // get min_freq
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_min_freq",
               cpu_id);
      fp = fopen(path, "rb");
      if (!fp) {
        return;
      }
      fscanf(fp, "%d", min_freq);
      fclose(fp);
      return;
    }
  }
  *min_freq = std::numeric_limits<int>::max();
  while (!feof(fp)) {
    int freq = 0;
    int nscan = fscanf(fp, "%d %*d", &freq);
    if (nscan != 1) {
      break;
    }
    if (freq > *max_freq) {
      *max_freq = freq;
    }
    if (freq < *min_freq) {
      *min_freq = freq;
    }
  }
  fclose(fp);
}

void sort_cpuid_by_max_freq(const std::vector<int>& max_freqs,
                            std::vector<int>* cpu_ids,
                            std::vector<int>* cluster_ids) {
  int cpu_num = max_freqs.size();
  if (cpu_num == 0) {
    return;
  }
  cpu_ids->resize(cpu_num);
  cluster_ids->resize(cpu_num);
  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
  }
  // sort cpuid as big core first
  // simple bubble sort
  for (int i = 0; i < cpu_num; i++) {
    for (int j = i + 1; j < cpu_num; j++) {
      if (max_freqs[i] < max_freqs[j]) {
        // swap
        int tmp = cpu_ids->at(i);
        cpu_ids->at(i) = cpu_ids->at(j);
        cpu_ids->at(j) = tmp;
      }
    }
  }
  // SMP
  int mid_max_freq =
      (max_freqs[cpu_ids->at(0)] + max_freqs[cpu_ids->at(cpu_num - 1)]) / 2;

  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
    if (max_freqs[i] >= mid_max_freq) {
      cluster_ids->at(i) = 0;
    } else {
      cluster_ids->at(i) = 1;
    }
  }
}

void get_cpu_cache_size(int cpu_id, int* l1_cache_size, int* l2_cache_size,
                        int* l3_cache_size) {
  int max_cache_idx_num = 10;
  *l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  *l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  *l3_cache_size = DEFAULT_L3_CACHE_SIZE;
  for (int i = 0; i < max_cache_idx_num; i++) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu_id, i);
    FILE* fp = fopen(path, "rb");
    if (fp) {
      int level = -1;
      fscanf(fp, "%d", &level);
      fclose(fp);
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu_id, i);
      fp = fopen(path, "rb");
      if (fp) {
        int size = -1;
        fscanf(fp, "%d", &size);
        fclose(fp);
        if (size >= 0) {
          if (level == 1) {
            *l1_cache_size = size * 1024;
          } else if (level == 2) {
            *l2_cache_size = size * 1024;
          } else if (level == 3) {
            *l3_cache_size = size * 1024;
          }
        }
      }
    }
  }
}

bool check_cpu_online(const std::vector<int>& cpu_ids) {
  if (cpu_ids.size() == 0) {
    return false;
  }
  char path[256];
  bool all_online = true;
  for (int i = 0; i < cpu_ids.size(); ++i) {
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online",
             cpu_ids[i]);
    FILE* fp = fopen(path, "rb");
    int is_online = 0;
    if (fp) {
      fscanf(fp, "%d", &is_online);
      fclose(fp);
    } else {
      LOG(ERROR) << "Failed to query the online statue of CPU id:"
                 << cpu_ids[i];
    }
    if (is_online == 0) {
      all_online = false;
      LOG(ERROR) << "CPU id:" << cpu_ids[i] << " is offine";
    }
  }
  return all_online;
}

int set_sched_affinity(const std::vector<int>& cpu_ids) {
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
  for (int i = 0; i < cpu_ids.size(); ++i) {
    CPU_SET(cpu_ids[i], &mask);
  }
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    return -1;
  }
  return 0;
}

bool bind_threads(const std::vector<int> cpu_ids) {
#ifdef ARM_WITH_OMP
  int thread_num = cpu_ids.size();
  omp_set_num_threads(thread_num);
  std::vector<int> ssarets;
  for (int i = 0; i < thread_num; ++i) {
    ssarets.push_back(0);
  }
#pragma omp parallel for
  for (int i = 0; i < thread_num; i++) {
    ssarets[i] = set_sched_affinity(cpu_ids);
  }
  for (int i = 0; i < thread_num; i++) {
    if (ssarets[i] != 0) {
      LOG(ERROR) << "Set cpu affinity failed, core id: " << cpu_ids[i];
      return false;
    }
  }
#else   // ARM_WITH_OMP
  std::vector<int> first_cpu_id;
  first_cpu_id.push_back(cpu_ids[0]);
  int ssaret = set_sched_affinity(first_cpu_id);
  if (ssaret != 0) {
    LOG(ERROR) << "Set cpu affinity failed, core id: " << cpu_ids[0];
    return false;
  }
#endif  // ARM_WITH_OMP
  return true;
}

#endif  // LITE_WITH_LINUX

// cache_id : 0 -> L1, 1 -> L2, 2 -> L3
void DeviceInfo::SetCacheInfo(int cache_id, int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  std::vector<int>* cache;
  switch (cache_id) {
    case 0:
      cache = &L1_cache_;
      break;
    case 1:
      cache = &L2_cache_;
      break;
    case 2:
      cache = &L3_cache_;
      break;
    default:
      break;
  }
  cache->resize(core_num_);
  if (argc == 1) {
    int cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < core_num_; ++i) {
      (*cache)[i] = cache_size;
    }
  } else {
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    int big_core_cache_size = va_arg(arg_ptr, int);
    int little_core_cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < big_core_num; ++i) {
      (*cache)[big_core_ids_[i]] = big_core_cache_size;
    }
    for (int i = 0; i < little_core_num; ++i) {
      (*cache)[little_core_ids_[i]] = little_core_cache_size;
    }
  }
  va_end(arg_ptr);
}

void DeviceInfo::SetArchInfo(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  archs_.resize(core_num_);
  if (argc == 1) {
    ARMArch arch = (ARMArch)va_arg(arg_ptr, int);
    for (int i = 0; i < core_num_; ++i) {
      archs_[i] = arch;
    }
  } else {
    ARMArch big_core_arch = (ARMArch)va_arg(arg_ptr, int);
    ARMArch little_core_arch = (ARMArch)va_arg(arg_ptr, int);
    int big_core_num = big_core_ids_.size();
    int little_core_num = little_core_ids_.size();
    for (int i = 0; i < big_core_num; ++i) {
      archs_[big_core_ids_[i]] = big_core_arch;
    }
    for (int i = 0; i < little_core_num; ++i) {
      archs_[little_core_ids_[i]] = little_core_arch;
    }
  }
  va_end(arg_ptr);
}

bool DeviceInfo::SetCPUInfoByName() {
  /* Snapdragon */
  if (dev_name_.find("SM8150") != std::string::npos) {  // 855
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    return true;
  } else if (dev_name_.find("SDM845") != std::string::npos) {  // 845
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    return true;
  } else if (dev_name_.find("SDM710") != std::string::npos) {  // 710
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {6, 7};
    little_core_ids_ = {0, 1, 2, 3, 4, 5};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8998") != std::string::npos) {  // 835
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024,
                 /*real cache size is 2M, while that will get bad performace
                    on conv3x3s1 or gemm, set to 1M or 512K*/
                 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8996") != std::string::npos) {  // 820
    core_num_ = 4;
    core_ids_ = {0, 1, 2, 3};
    big_core_ids_ = {2, 3};
    little_core_ids_ = {0, 1};
    cluster_ids_ = {1, 1, 0, 0};
    SetArchInfo(1, kA72);
    SetCacheInfo(0, 1, 24 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("SDM660") != std::string::npos ||
             dev_name_.find("SDM636") != std::string::npos) {  // 660, 636
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA73);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8976") != std::string::npos) {  // 652,653
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("MSM8953") != std::string::npos) {  // 625
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (dev_name_.find("MSM8939") != std::string::npos) {  // 615
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3};
    little_core_ids_ = {4, 5, 6, 7};
    cluster_ids_ = {0, 0, 0, 0, 1, 1, 1, 1};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 256 * 1024);
    return true;
    /* MediaTek */
  } else if (dev_name_.find("MT6797") !=
             std::string::npos) {  // X20/X23/X25/X27
    core_num_ = 10;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    big_core_ids_ = {8, 9};
    little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (dev_name_.find("MT6799") != std::string::npos) {  // X30
    core_num_ = 10;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    big_core_ids_ = {8, 9};
    little_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    cluster_ids_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (dev_name_.find("MT6795") != std::string::npos ||
             dev_name_.find("MT6762") != std::string::npos ||
             dev_name_.find("MT6755T") != std::string::npos ||
             dev_name_.find("MT6755S") != std::string::npos ||
             dev_name_.find("MT6753") != std::string::npos ||
             dev_name_.find("MT6752") != std::string::npos ||
             dev_name_.find("MT6750") != std::string::npos) {
    // X10, P22, P15/P18, MT6753, MT6752/MT6752M, MT6750
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (dev_name_.find("MT6758") != std::string::npos ||
             dev_name_.find("MT6757") != std::string::npos ||
             dev_name_.find("MT6763") != std::string::npos ||
             dev_name_.find("MT6755M") != std::string::npos ||
             dev_name_.find("MT6755") !=
                 std::string::npos) {  // P30, P20/P25, P23, P10
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (dev_name_.find("MT6771") != std::string::npos) {  // P60
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (dev_name_.find("MT6765") != std::string::npos ||
             dev_name_.find("MT6739") != std::string::npos ||
             dev_name_.find("MT6738") != std::string::npos ||
             dev_name_.find("MT6737") !=
                 std::string::npos) {  // A22, MT6739, MT6738, MT6767
    core_num_ = 4;
    core_ids_ = {0, 1, 2, 3};
    big_core_ids_ = {0, 1, 2, 3};
    little_core_ids_ = {};
    cluster_ids_ = {0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  }
  return false;
}

void DeviceInfo::SetCPUInfoByProb() {
#ifdef LITE_WITH_LINUX
  // get big.LITTLE cores by sorting CPU frequency
  sort_cpuid_by_max_freq(max_freqs_, &core_ids_, &cluster_ids_);
  big_core_ids_.clear();
  little_core_ids_.clear();
  for (int i = 0; i < cluster_ids_.size(); ++i) {
    if (cluster_ids_[i] == 0) {
      big_core_ids_.push_back(core_ids_[i]);
    } else {
      little_core_ids_.push_back(core_ids_[i]);
    }
  }
  // get l1, l2, l3 cache size for each core
  for (int i = 0; i < core_num_; i++) {
    get_cpu_cache_size(i, &(L1_cache_[i]), &(L2_cache_[i]), &(L3_cache_[i]));
  }
#endif  // LITE_WITH_LINUX
}

void DeviceInfo::RequestPowerFullMode(const int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  for (int i = 0; i < thread_num; ++i) {
    if (i < big_core_size) {
      active_ids_.push_back(big_core_ids_[i]);
    } else if (i < big_core_size + little_core_size) {
      active_ids_.push_back(little_core_ids_[i - big_core_size]);
    }
  }
  mode_ = LITE_POWER_FULL;
}

void DeviceInfo::RequestPowerHighMode(const int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (big_core_size > 0) {
    mode_ = LITE_POWER_HIGH;
    if (thread_num > big_core_size) {
      LOG(ERROR) << "Request thread num: " << thread_num
                 << ", exceed the big cores size: " << big_core_size
                 << ", truncate thread num to " << big_core_size;
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[i]);
      }
    }
  } else {
    mode_ = LITE_POWER_LOW;
    LOG(ERROR) << "HIGH POWER MODE is not support, switch to little cores.";
    if (thread_num > little_core_size) {
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerLowMode(const int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (little_core_size > 0) {
    mode_ = LITE_POWER_LOW;
    if (thread_num > little_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the little cores size: " << little_core_size
                   << ", truncate thread num to " << little_core_size;
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; i++) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  } else {
    mode_ = LITE_POWER_HIGH;
    LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores";
    if (thread_num > big_core_size) {
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; i++) {
        active_ids_.push_back(big_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerNoBindMode(const int thread_num) {
  active_ids_.clear();
  for (int i = 0; i < thread_num; i++) {
    active_ids_.push_back(0);
  }
  mode_ = LITE_POWER_NO_BIND;
}

void DeviceInfo::RequestPowerRandHighMode(const int shift_num,
                                          const int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (big_core_size > 0) {
    mode_ = LITE_POWER_RAND_HIGH;
    if (thread_num > big_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the big cores size: " << big_core_size
                   << ", truncate thread num to " << big_core_size;
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[(i + shift_num) % big_core_size]);
      }
    }
  } else {
    mode_ = LITE_POWER_LOW;
    LOG(WARNING) << "HIGH POWER MODE is not support, switch to little cores.";
    if (thread_num > little_core_size) {
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(little_core_ids_[i]);
      }
    }
  }
}

void DeviceInfo::RequestPowerRandLowMode(const int shift_num,
                                         const int thread_num) {
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  active_ids_.clear();
  if (little_core_size > 0) {
    mode_ = LITE_POWER_RAND_LOW;
    if (thread_num > little_core_size) {
      LOG(WARNING) << "Request thread num: " << thread_num
                   << ", exceed the little cores size: " << little_core_size
                   << ", truncate thread num to " << little_core_size;
      active_ids_ = little_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(
            little_core_ids_[(i + shift_num) % little_core_size]);
      }
    }
  } else {
    mode_ = LITE_POWER_HIGH;
    LOG(WARNING) << "LOW POWER MODE is not support, switch to big cores.";
    if (thread_num > big_core_size) {
      active_ids_ = big_core_ids_;
    } else {
      for (int i = 0; i < thread_num; ++i) {
        active_ids_.push_back(big_core_ids_[i]);
      }
    }
  }
}

int DeviceInfo::Setup() {
  core_num_ = get_cpu_num();
  mem_size_ = get_mem_size();
  get_cpu_arch(&archs_, core_num_);
  // set defalut CPU info
  SetCacheInfo(0, 1, DEFAULT_L1_CACHE_SIZE);
  SetCacheInfo(1, 1, DEFAULT_L2_CACHE_SIZE);
  SetCacheInfo(2, 1, DEFAULT_L3_CACHE_SIZE);
#ifdef LITE_WITH_LINUX
  // get max&min freq
  max_freqs_.resize(core_num_);
  min_freqs_.resize(core_num_);
  for (int i = 0; i < core_num_; ++i) {
    int max_freq, min_freq;
    get_cpu_max_min_freq(i, &max_freq, &min_freq);
    max_freqs_[i] = max_freq / 1000;
    min_freqs_[i] = min_freq / 1000;
  }
  // get cache size and big.LITTLE core ids
  dev_name_ = get_cpu_name();
  if (!SetCPUInfoByName()) {
    SetCPUInfoByProb();
  }
  // output info
  LOG(INFO) << "ARM multiprocessors name: " << dev_name_;
  LOG(INFO) << "ARM multiprocessors number: " << core_num_;
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << "ARM multiprocessors ID: " << core_ids_[i]
              << ", max freq: " << max_freqs_[i]
              << ", min freq: " << min_freqs_[i]
              << ", cluster ID: " << cluster_ids_[core_ids_[i]]
              << ", CPU ARCH: A" << archs_[i];
  }
  LOG(INFO) << "L1 DataCache size is: ";
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << L1_cache_[i] / 1024 << " KB";
  }
  LOG(INFO) << "L2 Cache size is: ";
  for (int i = 0; i < core_num_; ++i) {
    LOG(INFO) << L2_cache_[i] / 1024 << " KB";
  }
  LOG(INFO) << "Total memory: " << mem_size_ << "KB";
#endif
  // set default run mode
  SetRunMode(LITE_POWER_NO_BIND, 1);  // use single thread by default
  return 0;
}

void DeviceInfo::SetRunMode(PowerMode mode, int thread_num) {
#ifdef ARM_WITH_OMP
  thread_num = std::min(thread_num, core_num_);
#else
  thread_num = 1;  // force thread_num to 1 if OpenMP is disabled
#endif
#ifdef LITE_WITH_LINUX
  int big_core_size = big_core_ids_.size();
  int little_core_size = little_core_ids_.size();
  int big_little_core_size = big_core_size + little_core_size;
  thread_num = std::min(thread_num, big_little_core_size);
  count_++;
  int shift_num = (count_ / 10) % big_core_size;
  switch (mode) {
    case LITE_POWER_FULL:
      RequestPowerFullMode(thread_num);
      break;
    case LITE_POWER_HIGH:
      RequestPowerHighMode(thread_num);
      break;
    case LITE_POWER_LOW:
      RequestPowerLowMode(thread_num);
      break;
    case LITE_POWER_NO_BIND:
      RequestPowerNoBindMode(thread_num);
      break;
    case LITE_POWER_RAND_HIGH:
      RequestPowerRandHighMode(shift_num, thread_num);
      break;
    case LITE_POWER_RAND_LOW:
      RequestPowerRandLowMode(shift_num, thread_num);
      break;
    default:
      LOG(FATAL) << "Unsupported power mode: " << mode;
      break;
  }
  if (active_ids_.size() == 0) {
    active_ids_.push_back(0);
  }
#ifdef ARM_WITH_OMP
  omp_set_num_threads(active_ids_.size());
#endif
  if (mode_ != LITE_POWER_NO_BIND) {
    if (check_cpu_online(active_ids_)) {
      bind_threads(active_ids_);
    } else {
      LOG(WARNING) << "Some cores are offline, switch to NO BIND MODE";
      mode_ = LITE_POWER_NO_BIND;
    }
  }
#else  // LITE_WITH_LINUX
  // only LITE_POWER_NO_BIND is supported in other OS
  RequestPowerNoBindMode(thread_num);
#ifdef ARM_WITH_OMP
  omp_set_num_threads(active_ids_.size());
#endif
#endif  // LITE_WITH_LINUX
  //! alloc memory for sgemm in this context
  workspace_.Resize(
      {static_cast<int64_t>(L2_cache_[active_ids_[0]] / sizeof(float))});
  arch_ = archs_[active_ids_[0]];
}

void DeviceInfo::SetCache(int l1size, int l2size, int l3size) {
  SetCacheInfo(0, 1, l1size);
  SetCacheInfo(1, 1, l2size);
  SetCacheInfo(2, 1, l3size);
  workspace_.Resize({2 * (l1size + l2size)});
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

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
