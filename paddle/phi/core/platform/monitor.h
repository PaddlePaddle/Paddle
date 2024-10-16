//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>

#include <atomic>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/common/macros.h"

namespace paddle {
namespace platform {

template <typename T>
class StatRegistry;

class MonitorRegistrar {
 public:
  // The design is followed by OperatorRegistrar: To avoid the removal of global
  // name by the linkerr, we add Touch to all StatValue classes and make
  // USE_STAT macros to call this method. So, as long as the callee code calls
  // USE_STAT, the global registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename T>
class StatValue : public MonitorRegistrar {
  T v_{0};
  std::mutex mu_;
  // We use lock rather than atomic for generic values
 public:
  explicit StatValue(const std::string& n) {
    StatRegistry<T>::Instance().add(n, this);
  }
  T increase(T inc) {
    std::lock_guard<std::mutex> lock(mu_);
    return v_ += inc;
  }
  T decrease(T inc) {
    std::lock_guard<std::mutex> lock(mu_);
    return v_ -= inc;
  }
  T reset(T value = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    return v_ = value;
  }
  T get() {
    std::lock_guard<std::mutex> lock(mu_);
    return v_;
  }
};

template <typename T>
struct ExportedStatValue {
  std::string key;
  T value;
};

template <typename T>
class StatRegistry {
 public:
  ~StatRegistry<T>() {}

  static StatRegistry<T>& Instance() {
    static StatRegistry<T> r;
    return r;
  }
  StatValue<T>* get(const std::string& name) {
    std::lock_guard<std::mutex> lg(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }
  int add(const std::string& name, StatValue<T>* stat) {
    std::lock_guard<std::mutex> lg(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) {
      return -1;
    }
    stats_.insert(std::make_pair(name, stat));
    return 0;
  }

  void publish(std::vector<ExportedStatValue<T>>& exported,  // NOLINT
               bool reset = false) {
    std::lock_guard<std::mutex> lg(mutex_);
    exported.resize(stats_.size());
    int i = 0;
    for (const auto& kv : stats_) {
      auto& out = exported.at(i++);
      out.key = kv.first;
      out.value = reset ? kv.second->reset() : kv.second->get();
    }
  }

  std::vector<ExportedStatValue<T>> publish(bool reset = false) {
    std::vector<ExportedStatValue<T>> stats;
    publish(stats, reset);
    return stats;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, StatValue<T>*> stats_;
};

}  // namespace platform
}  // namespace paddle

#define STAT_ADD(item, t) _##item.increase(t)
#define STAT_SUB(item, t) _##item.decrease(t)

// Support add stat value by string
#define STAT_INT_ADD(item, t) \
  paddle::platform::StatRegistry<int64_t>::Instance().get(item)->increase(t)
#define STAT_INT_SUB(item, t) \
  paddle::platform::StatRegistry<int64_t>::Instance().get(item)->decrease(t)

#define STAT_FLOAT_ADD(item, t) \
  paddle::platform::StatRegistry<float>::Instance().get(item)->increase(t)
#define STAT_FLOAT_SUB(item, t) \
  paddle::platform::StatRegistry<float>::Instance().get(item)->decrease(t)

#define STAT_RESET(item, t) _##item.reset(t)
#define STAT_GET(item) _##item.get()

#define DEFINE_FLOAT_STATUS(item)                    \
  paddle::platform::StatValue<float> _##item(#item); \
  int TouchStatRegistrar_##item() {                  \
    _##item.Touch();                                 \
    return 0;                                        \
  }

#define DEFINE_INT_STATUS(item)                        \
  paddle::platform::StatValue<int64_t> _##item(#item); \
  int TouchStatRegistrar_##item() {                    \
    _##item.Touch();                                   \
    return 0;                                          \
  }

#define USE_STAT(item)                    \
  extern int TouchStatRegistrar_##item(); \
  UNUSED static int use_stat_##item = TouchStatRegistrar_##item()

#define USE_INT_STAT(item)                             \
  extern paddle::platform::StatValue<int64_t> _##item; \
  USE_STAT(item)

#define USE_FLOAT_STAT(item)                         \
  extern paddle::platform::StatValue<float> _##item; \
  USE_STAT(item)

#define USE_GPU_MEM_STAT             \
  USE_INT_STAT(STAT_gpu0_mem_size);  \
  USE_INT_STAT(STAT_gpu1_mem_size);  \
  USE_INT_STAT(STAT_gpu2_mem_size);  \
  USE_INT_STAT(STAT_gpu3_mem_size);  \
  USE_INT_STAT(STAT_gpu4_mem_size);  \
  USE_INT_STAT(STAT_gpu5_mem_size);  \
  USE_INT_STAT(STAT_gpu6_mem_size);  \
  USE_INT_STAT(STAT_gpu7_mem_size);  \
  USE_INT_STAT(STAT_gpu8_mem_size);  \
  USE_INT_STAT(STAT_gpu9_mem_size);  \
  USE_INT_STAT(STAT_gpu10_mem_size); \
  USE_INT_STAT(STAT_gpu11_mem_size); \
  USE_INT_STAT(STAT_gpu12_mem_size); \
  USE_INT_STAT(STAT_gpu13_mem_size); \
  USE_INT_STAT(STAT_gpu14_mem_size); \
  USE_INT_STAT(STAT_gpu15_mem_size)
