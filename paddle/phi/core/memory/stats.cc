/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/memory/stats.h"

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

PHI_DEFINE_EXPORTED_bool(
    log_memory_stats,
    false,
    "Log memory stats after each op runs, just used for debug.");
namespace paddle::memory {

class StatRegistry {
 public:
  static StatRegistry* GetInstance() {
    static StatRegistry instance;
    return &instance;
  }

  StatBase* GetStat(const std::string& stat_type, int dev_id) {
    auto it = stat_map_.find(GetStatKey(stat_type, dev_id));
    if (it == stat_map_.end()) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The STAT type \"%s\" for device %d has not been registered.",
          stat_type.c_str(),
          dev_id));
    }
    return it->second;
  }

  std::string GetStatKey(const std::string& stat_type, int dev_id) {
    return stat_type + std::to_string(dev_id);
  }

  int64_t GetCurrentValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetCurrentValue();
  }

  int64_t GetPeakValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetPeakValue();
  }

  void Update(const std::string& stat_type, int dev_id, int64_t increment) {
    GetStat(stat_type, dev_id)->Update(increment);
  }

  void ResetPeakValue(const std::string& stat_type, int dev_id) {
    GetStat(stat_type, dev_id)->ResetPeakValue();
  }

  void Register(const std::string& stat_type, int dev_id, StatBase* stat) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_[GetStatKey(stat_type, dev_id)] = stat;
  }

  void Unregister(const std::string& stat_type, int dev_id) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_.erase(GetStatKey(stat_type, dev_id));
  }

 private:
  StatRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(StatRegistry);

  std::unordered_map<std::string, StatBase*> stat_map_;
  SpinLock stat_map_lock_;
};

int64_t DeviceMemoryStatCurrentValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetCurrentValue("Device" + stat_type,
                                                      dev_id);
}

int64_t DeviceMemoryStatPeakValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetPeakValue("Device" + stat_type,
                                                   dev_id);
}

void DeviceMemoryStatUpdate(const std::string& stat_type,
                            int dev_id,
                            int64_t increment) {
  StatRegistry::GetInstance()->Update("Device" + stat_type, dev_id, increment);
}

void DeviceMemoryStatResetPeakValue(const std::string& stat_type, int dev_id) {
  StatRegistry::GetInstance()->ResetPeakValue("Device" + stat_type, dev_id);
}

int64_t HostMemoryStatCurrentValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetCurrentValue("Host" + stat_type,
                                                      dev_id);
}

int64_t HostMemoryStatPeakValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetPeakValue("Host" + stat_type, dev_id);
}

void HostMemoryStatUpdate(const std::string& stat_type,
                          int dev_id,
                          int64_t increment) {
  StatRegistry::GetInstance()->Update("Host" + stat_type, dev_id, increment);
}

void HostMemoryStatResetPeakValue(const std::string& stat_type, int dev_id) {
  StatRegistry::GetInstance()->ResetPeakValue("Host" + stat_type, dev_id);
}

void LogDeviceMemoryStats(const phi::Place& place, const std::string& op_name) {
  if (FLAGS_log_memory_stats && phi::is_gpu_place(place)) {
    VLOG(0) << "After launching op_name: " << op_name << ", "
            << "memory_allocated: "
            << static_cast<double>(memory::DeviceMemoryStatCurrentValue(
                   "Allocated", place.device)) /
                   1024 / 1024
            << " MB, "
            << "memory_reserved: "
            << static_cast<double>(memory::DeviceMemoryStatCurrentValue(
                   "Reserved", place.device)) /
                   1024 / 1024
            << " MB, "
            << "max_memory_allocated: "
            << static_cast<double>(memory::DeviceMemoryStatPeakValue(
                   "Allocated", place.device)) /
                   1024 / 1024
            << " MB, "
            << "max_memory_reserved: "
            << static_cast<double>(memory::DeviceMemoryStatPeakValue(
                   "Reserved", place.device)) /
                   1024 / 1024
            << " MB";
  }
}

#define DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, id) \
  StatRegistry::GetInstance()->Register(              \
      "Device" #item, id, Stat<DeviceMemoryStat##item##id>::GetInstance());

#define DEVICE_MEMORY_STAT_REGISTER(item)        \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 0);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 1);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 2);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 3);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 4);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 5);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 6);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 7);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 8);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 9);  \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 10); \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 11); \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 12); \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 13); \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 14); \
  DEVICE_MEMORY_STAT_REGISTER_WITH_ID(item, 15)

#define HOST_MEMORY_STAT_REGISTER(item)  \
  StatRegistry::GetInstance()->Register( \
      "Host" #item, 0, Stat<HostMemoryStat##item##0>::GetInstance());

int RegisterAllStats() {
  DEVICE_MEMORY_STAT_REGISTER(Allocated);
  DEVICE_MEMORY_STAT_REGISTER(Reserved);

  HOST_MEMORY_STAT_REGISTER(Allocated);
  HOST_MEMORY_STAT_REGISTER(Reserved);
  return 0;
}

UNUSED static int register_all_stats = RegisterAllStats();

}  // namespace paddle::memory
