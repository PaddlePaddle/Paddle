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
#include "paddle/fluid/framework/new_executor/workqueue/thread_data_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace memory {

using framework::ThreadDataRegistry;

struct ThreadLocalStatBase {
  int64_t current{0};
  int64_t peak{0};
};

class StatBase {
 public:
  StatBase() = default;
  virtual ~StatBase() = default;

  virtual int64_t GetCurrentValue() = 0;
  virtual int64_t GetPeakValue() = 0;
  virtual void Update(int64_t) = 0;

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

    if (thread_local_stat->current > thread_local_stat->peak) {
      thread_local_stat->peak = thread_local_stat->current;
      int64_t current_value = GetCurrentValue();
      int64_t prev_value = peak_value_;
      while (prev_value < current_value &&
             !peak_value_.compare_exchange_weak(prev_value, current_value)) {
      }
      VLOG(8) << "Update peak_value, after update, peak_value = " << peak_value_
              << " , current value = " << current_value;
    }
  }

 private:
  Stat() {}
  ~Stat() {}
  std::atomic<int64_t> peak_value_{0};
};

// StatGetCurrentValue, StatGetPeakValue and StatUpdate support to operate STAT
// values by a string, however, they has worse performance than the macro
// function MEMORY_STAT_CURRENT_VALUE, MEMORY_STAT_PEAK_VALUE, and
// MEMORY_STAT_UPDATE. Try to use the macro functions where ultra-low
// performance overhead is required.
int64_t StatGetCurrentValue(const std::string& stat_type, int dev_id);
int64_t StatGetPeakValue(const std::string& stat_type, int dev_id);
void StatUpdate(const std::string& stat_type, int dev_id, int64_t increment);

#define MEMORY_STAT_FUNC_SWITHCH_CASE(item, id)                          \
  case id:                                                               \
    stat = paddle::memory::Stat<                                         \
        paddle::memory::ThreadLocalStatDevice##id##item>::GetInstance(); \
    break

#define MEMORY_STAT_FUNC(item, id, func, ...)                         \
  do {                                                                \
    paddle::memory::StatBase* stat = nullptr;                         \
    switch (id) {                                                     \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 0);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 1);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 2);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 3);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 4);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 5);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 6);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 7);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 8);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 9);                         \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 10);                        \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 11);                        \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 12);                        \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 13);                        \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 14);                        \
      MEMORY_STAT_FUNC_SWITHCH_CASE(item, 15);                        \
      default:                                                        \
        PADDLE_THROW(paddle::platform::errors::OutOfRange(            \
            "Only support device id between [0, 15] in memory stats," \
            "not support device id: %d",                              \
            id));                                                     \
        break;                                                        \
    }                                                                 \
    stat->func(__VA_ARGS__);                                          \
  } while (0)

#define MEMORY_STAT_CURRENT_VALUE(item, id) \
  MEMORY_STAT_FUNC(item, id, GetCurrentValue)
#define MEMORY_STAT_PEAK_VALUE(item, id) \
  MEMORY_STAT_FUNC(item, id, GetPeakValue)
#define MEMORY_STAT_UPDATE(item, id, increment) \
  MEMORY_STAT_FUNC(item, id, Update, increment)

#define MEMORY_STAT_DECLARE_WITH_ID(item, id) \
  struct ThreadLocalStatDevice##id##item : public ThreadLocalStatBase {};

#define MEMORY_STAT_DECLARE(item)        \
  MEMORY_STAT_DECLARE_WITH_ID(item, 0);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 1);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 2);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 3);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 4);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 5);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 6);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 7);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 8);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 9);  \
  MEMORY_STAT_DECLARE_WITH_ID(item, 10); \
  MEMORY_STAT_DECLARE_WITH_ID(item, 11); \
  MEMORY_STAT_DECLARE_WITH_ID(item, 12); \
  MEMORY_STAT_DECLARE_WITH_ID(item, 13); \
  MEMORY_STAT_DECLARE_WITH_ID(item, 14); \
  MEMORY_STAT_DECLARE_WITH_ID(item, 15)

// To add a new STAT type, declare here and register in stats.cc
MEMORY_STAT_DECLARE(Allocated);
MEMORY_STAT_DECLARE(Reserved);

}  // namespace memory
}  // namespace paddle
