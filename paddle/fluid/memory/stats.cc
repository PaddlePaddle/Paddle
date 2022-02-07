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

#include "paddle/fluid/memory/stats.h"
#include <atomic>
#include <map>
#include <string>
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/variant.h"

// "thread_data", "atomic", "monitor"
PADDLE_DEFINE_EXPORTED_string(memory_stats_opt, "thread_data", "None");

namespace paddle {
namespace memory {

using platform::internal::ThreadDataRegistry;

struct ThreadLocalStatBase {
  int64_t current{0};
  int64_t peak{0};
};

class StatBase {
 public:
  StatBase() = default;
  virtual ~StatBase() = default;

  // int Touch() { return 0; }

  virtual int64_t GetCurrentValue() = 0;
  virtual int64_t GetPeakValue() = 0;
  virtual void Update(int64_t) = 0;

 private:
  DISABLE_COPY_AND_ASSIGN(StatBase);
};

template <typename ThreadLocalStatType>
class Stat : public StatBase {
 public:
  static Stat& GetInstance() {
    static Stat instance;
    return instance;
  }

  int64_t GetCurrentValue() override {
    std::unordered_map<uint64_t, ThreadLocalStatType> thread_local_stats =
        ThreadDataRegistry<ThreadLocalStatType>::GetInstance()
            .GetAllThreadDataByValue();
    int64_t current_value = 0;
    for (auto pair : thread_local_stats) {
      current_value += pair.second.current;
    }
    return current_value;
  }

  int64_t GetPeakValue() override { return peak_value_; }

  void Update(int64_t increment) override {
    if (FLAGS_memory_stats_opt == "atomic") {
      int64_t current = current_.fetch_add(increment);
      int64_t prev_value = peak_.load();
      while (prev_value < current &&
             !peak_.compare_exchange_weak(prev_value, current)) {
      }
    }

    ThreadLocalStatType thread_local_stat =
        ThreadDataRegistry<ThreadLocalStatType>::GetInstance()
            .GetCurrentThreadData();
    thread_local_stat.current += increment;

    if (thread_local_stat.current > thread_local_stat.peak) {
      thread_local_stat.peak = thread_local_stat.current;
      int64_t current_value = GetCurrentValue() + increment;
      /* lock_guard */ {
        std::lock_guard<SpinLock> lock_guard(peak_value_lock_);
        peak_value_ = std::max(current_value, peak_value_);
      }
    }
    ThreadDataRegistry<ThreadLocalStatType>::GetInstance().SetCurrentThreadData(
        thread_local_stat);
  }

 private:
  Stat() {}
  ~Stat() {}

  int64_t peak_value_{0};
  SpinLock peak_value_lock_;

  // fot test
  std::atomic<int64_t> current_;
  std::atomic<int64_t> peak_;
};

class StatRegistry {
 public:
  static StatRegistry& GetInstance() {
    static StatRegistry instance;
    return instance;
  }

  StatBase* GetStat(const std::string& stat_type, int dev_id) {
    auto it = stat_map_.find(GetStatKey(stat_type, dev_id));
    if (it == stat_map_.end()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The STAT type \"%s\" for device %d has not been regeistered.",
          stat_type.c_str(), dev_id));
    }
    return it->second;
  }

  std::string GetStatKey(const std::string& stat_type, int dev_id) {
    return "STAT_Device" + std::to_string(dev_id) + "_" + stat_type;
  }

  int64_t GetCurrentValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetCurrentValue();
  }

  int64_t GetPeakValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetPeakValue();
  }

  void Register(const std::string& stat_type, int dev_id, StatBase* stat) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_[GetStatKey(stat_type, dev_id)] = stat;
  }

  void Unregister(const std::string& stat_type, int dev_id) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_.erase(GetStatKey(stat_type, dev_id));
  }

  void Update(const std::string& stat_type, int dev_id, int64_t increment) {
    VLOG(1) << "Update: type = " << stat_type << "  id = " << dev_id
            << "  increment = " << increment;
    stat_map_[GetStatKey(stat_type, dev_id)]->Update(increment);
  }

 private:
  StatRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(StatRegistry);

  std::unordered_map<std::string, StatBase*> stat_map_;  // not owned
  SpinLock stat_map_lock_;
};

int64_t StatGetCurrentValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance().GetCurrentValue(stat_type, dev_id);
}

int64_t StatGetPeakValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance().GetPeakValue(stat_type, dev_id);
}

void StatUpdate(const std::string& stat_type, int dev_id, int64_t increment) {
  if (FLAGS_memory_stats_opt == "thread" ||
      FLAGS_memory_stats_opt == "atomic") {
    StatRegistry::GetInstance().Update(stat_type, dev_id, increment);
  } else if (FLAGS_memory_stats_opt == "monitor") {
    if (stat_type == "Allocated") {
      if (increment > 0) {
        int64_t alloc_size = STAT_INT_ADD(
            "STAT_gpu" + std::to_string(dev_id) + "_alloc_size", increment);
        STAT_INT_UPDATE_MAXIMUM(
            "STAT_gpu" + std::to_string(dev_id) + "_max_alloc_size",
            alloc_size);
      } else {
        STAT_INT_SUB("STAT_gpu" + std::to_string(dev_id) + "_alloc_size",
                     -increment);
      }
    } else {  // Reserved
      if (increment > 0) {
        int64_t mem_size = STAT_INT_ADD(
            "STAT_gpu" + std::to_string(dev_id) + "_mem_size", increment);
        STAT_INT_UPDATE_MAXIMUM(
            "STAT_gpu" + std::to_string(dev_id) + "_max_mem_size", mem_size);
      } else {
        STAT_INT_SUB("STAT_gpu" + std::to_string(dev_id) + "_mem_size",
                     -increment);
      }
    }
  } else if (FLAGS_memory_stats_opt != "disable") {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not support FLAGS_memory_stats_opt = %s", FLAGS_memory_stats_opt));
  }
}

#define MEMORY_STAT_REGISTER_WITH_ID(item, id)                              \
  do {                                                                      \
    struct ThreadLocalStatDevice##id##item : public ThreadLocalStatBase {}; \
    StatRegistry::GetInstance().Register(                                   \
        #item, id, &Stat<ThreadLocalStatDevice##id##item>::GetInstance());  \
  } while (0)

#define MEMORY_STAT_REGISTER(item)          \
  do {                                      \
    MEMORY_STAT_REGISTER_WITH_ID(item, 0);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 1);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 2);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 3);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 4);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 5);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 6);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 7);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 8);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 9);  \
    MEMORY_STAT_REGISTER_WITH_ID(item, 10); \
    MEMORY_STAT_REGISTER_WITH_ID(item, 11); \
    MEMORY_STAT_REGISTER_WITH_ID(item, 12); \
    MEMORY_STAT_REGISTER_WITH_ID(item, 13); \
    MEMORY_STAT_REGISTER_WITH_ID(item, 14); \
    MEMORY_STAT_REGISTER_WITH_ID(item, 15); \
  } while (0)

// To add a new STAT type, regiester it in this function
int RegisterAllStats() {
  MEMORY_STAT_REGISTER(Allocated);
  MEMORY_STAT_REGISTER(Reserved);
  return 0;
}

UNUSED static int regiester_all_stats = RegisterAllStats();

/*

MEMORY_STAT_DECLARE(allocated);
MEMORY_STAT_DECLARE(reserved);


#define MEMORY_STAT_DEFINE_WITH_ID(item, id) \
StatRegistry::GetInstance().Register(item, id,
&Stat<ThreadLocalStat_device##id##_##item>::GetInstance());
//Stat<ThreadLocalStat_device##id##_##item> STAT_device##id##_##item; \
//UNUSED static int use_stat_device##id##_##item =
STAT_device##id##_##item.Touch()

#define MEMORY_STAT_DEFINE(item)     \
MEMORY_STAT_DEFINE_WITH_ID(item, 0); \
MEMORY_STAT_DEFINE_WITH_ID(item, 1); \
MEMORY_STAT_DEFINE_WITH_ID(item, 2)


#define MEMORY_STAT_FUNC(item, id, func, args...) \
do {                                              \
  memory::StatBase* stat;                                 \
  if(id == 0) {                                   \
    stat = &STAT_device0_##item;                  \
  }                                               \
  else {                                          \
    stat = &STAT_device1_##item;                  \
  }                                               \
  stat->func(args);                               \
} while(0)

#define MEMORY_STAT_UPDATE(item, id, increment) MEMORY_STAT_FUNC(item, id,
Update, increment)
#define MEMORY_STAT_CURRENT_VALUE(item, id) MEMORY_STAT_FUNC(item, id,
GetCurrentValue)
#define MEMORY_STAT_PEAK_VALUE(item, id) MEMORY_STAT_FUNC(item, id,
GetPeakValue)
*/

}  // namespace memory
}  // namespace paddle
