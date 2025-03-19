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

#pragma once

#include <atomic>
#include <map>
#include <string>

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/thread_data_registry.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace memory {

using phi::ThreadDataRegistry;

struct ThreadLocalStatBase {
  int64_t current{0};
  int64_t peak{0};

  ThreadLocalStatBase operator+=(const ThreadLocalStatBase& other) {
    current += other.current;
    peak = std::max({current, peak, other.peak});
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ThreadLocalStatBase& stat) {
    os << "{current : " << stat.current << ", peak : " << stat.peak << "}";
    return os;
  }
};

class StatBase {
 public:
  StatBase() = default;
  virtual ~StatBase() = default;

  virtual int64_t GetCurrentValue() = 0;
  virtual int64_t GetPeakValue() = 0;
  virtual void Update(int64_t) = 0;
  virtual void ResetPeakValue() = 0;

 private:
  DISABLE_COPY_AND_ASSIGN(StatBase);
};

template <typename ThreadLocalStatType>
class Stat : public StatBase {
 public:
  static Stat* GetInstance() {
    static Stat instance;
    return &instance;
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
    auto& thread_data_registry =
        ThreadDataRegistry<ThreadLocalStatType>::GetInstance();
    ThreadLocalStatType* thread_local_stat =
        thread_data_registry.GetMutableCurrentThreadData();
    thread_local_stat->current += increment;

    VLOG(8) << string::split_string(
                   common::demangle(typeid(*thread_local_stat).name()), "::")
                   .back()
            << ": Update current_value with " << increment
            << ", after update, current value = " << GetCurrentValue();

    if (thread_local_stat->current > thread_local_stat->peak) {
      thread_local_stat->peak = thread_local_stat->current;
      int64_t current_value = GetCurrentValue();
      int64_t prev_value = peak_value_;
      while (prev_value < current_value &&
             !peak_value_.compare_exchange_weak(prev_value, current_value)) {
      }
      VLOG(8) << string::split_string(
                     common::demangle(typeid(*thread_local_stat).name()), "::")
                     .back()
              << ": Update current_value with " << increment
              << ", after update, peak_value = " << peak_value_.load()
              << " , current value = " << current_value;
    }
  }

  void ResetPeakValue() override {
    int64_t current_value = GetCurrentValue();
    peak_value_.store(current_value, std::memory_order_relaxed);

    std::unordered_map<uint64_t, std::reference_wrapper<ThreadLocalStatType>>
        thread_local_stats =
            ThreadDataRegistry<ThreadLocalStatType>::GetInstance()
                .GetAllThreadDataByRef();

    for (auto pair : thread_local_stats) {
      pair.second.get().peak = pair.second.get().current;
    }

    VLOG(8) << "Reset peak_value to current_value = " << current_value;
  }

 private:
  Stat() {}
  ~Stat() {}
  std::atomic<int64_t> peak_value_{0};
};

// xxxMemoryStatCurrentValue, xxxMemoryStatPeakValue and xxxMemoryStatUpdate
// support to operate STAT values by a string, however, they has worse
// performance than the macro function xxx_MEMORY_STAT_CURRENT_VALUE,
// xxx_MEMORY_STAT_PEAK_VALUE, and xxx_MEMORY_STAT_UPDATE. Try to use the macro
// functions where ultra-low performance overhead is required.
int64_t DeviceMemoryStatCurrentValue(const std::string& stat_type, int dev_id);
int64_t DeviceMemoryStatPeakValue(const std::string& stat_type, int dev_id);
void DeviceMemoryStatUpdate(const std::string& stat_type,
                            int dev_id,
                            int64_t increment);
void DeviceMemoryStatResetPeakValue(const std::string& stat_type, int dev_id);

int64_t HostMemoryStatCurrentValue(const std::string& stat_type, int dev_id);
int64_t HostMemoryStatPeakValue(const std::string& stat_type, int dev_id);
void HostMemoryStatUpdate(const std::string& stat_type,
                          int dev_id,
                          int64_t increment);
void HostMemoryStatResetPeakValue(const std::string& stat_type, int dev_id);

void LogDeviceMemoryStats(const phi::Place& place, const std::string& op_name);

#define DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, id)               \
  case id:                                                          \
    stat = paddle::memory::Stat<                                    \
        paddle::memory::DeviceMemoryStat##item##id>::GetInstance(); \
    break

#define DEVICE_MEMORY_STAT_FUNC(item, id, func, ...)                          \
  [&] {                                                                       \
    paddle::memory::StatBase* stat = nullptr;                                 \
    switch (id) {                                                             \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 0);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 1);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 2);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 3);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 4);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 5);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 6);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 7);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 8);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 9);                           \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 10);                          \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 11);                          \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 12);                          \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 13);                          \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 14);                          \
      DEVICE_MEMORY_STAT_FUNC_SWITCH_CASE(item, 15);                          \
      default:                                                                \
        PADDLE_THROW(common::errors::OutOfRange(                              \
            "Only support device id between [0, 15] for device memory stats," \
            "not support device id: %d",                                      \
            id));                                                             \
        break;                                                                \
    }                                                                         \
    return stat->func(__VA_ARGS__);                                           \
  }()

#define DEVICE_MEMORY_STAT_CURRENT_VALUE(item, id) \
  DEVICE_MEMORY_STAT_FUNC(item, id, GetCurrentValue)
#define DEVICE_MEMORY_STAT_PEAK_VALUE(item, id) \
  DEVICE_MEMORY_STAT_FUNC(item, id, GetPeakValue)
#define DEVICE_MEMORY_STAT_UPDATE(item, id, increment) \
  DEVICE_MEMORY_STAT_FUNC(item, id, Update, increment)
#define DEVICE_MEMORY_STAT_RESET_PEAK_VALUE(item, id) \
  DEVICE_MEMORY_STAT_FUNC(item, id, ResetPeakValue)

#define HOST_MEMORY_STAT_FUNC(item, id, func, ...)                             \
  [&] {                                                                        \
    PADDLE_ENFORCE_EQ(                                                         \
        id,                                                                    \
        0,                                                                     \
        common::errors::OutOfRange("Only support device id 0 for host memory " \
                                   "stats, not support device id: %d",         \
                                   id));                                       \
    return paddle::memory::Stat<                                               \
               paddle::memory::HostMemoryStat##item##0>::GetInstance()         \
        ->func(__VA_ARGS__);                                                   \
  }()

#define HOST_MEMORY_STAT_CURRENT_VALUE(item, id) \
  HOST_MEMORY_STAT_FUNC(item, id, GetCurrentValue)
#define HOST_MEMORY_STAT_PEAK_VALUE(item, id) \
  HOST_MEMORY_STAT_FUNC(item, id, GetPeakValue)
#define HOST_MEMORY_STAT_UPDATE(item, id, increment) \
  HOST_MEMORY_STAT_FUNC(item, id, Update, increment)
#define HOST_MEMORY_STAT_RESET_PEAK_VALUE(item, id) \
  HOST_MEMORY_STAT_FUNC(item, id, ResetPeakValue)

#define DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, id) \
  struct DeviceMemoryStat##item##id : public ThreadLocalStatBase {}

#define DEVICE_MEMORY_STAT_DECLARE(item)        \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 0);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 1);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 2);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 3);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 4);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 5);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 6);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 7);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 8);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 9);  \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 10); \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 11); \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 12); \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 13); \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 14); \
  DEVICE_MEMORY_STAT_DECLARE_WITH_ID(item, 15)

// Only support id 0 for host memory stat
#define HOST_MEMORY_STAT_DECLARE(item) \
  struct HostMemoryStat##item##0 : public ThreadLocalStatBase{};

// To add a new STAT type, declare here and register in stats.cc
DEVICE_MEMORY_STAT_DECLARE(Allocated);
DEVICE_MEMORY_STAT_DECLARE(Reserved);

HOST_MEMORY_STAT_DECLARE(Allocated);
HOST_MEMORY_STAT_DECLARE(Reserved);

}  // namespace memory
}  // namespace paddle
