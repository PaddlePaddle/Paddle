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
#include "glog/logging.h"

namespace paddle {
namespace platform {

template <typename T>
class StatRegistry;

template <typename T>
class StatValue {
  T v_{0};
  std::mutex mu_;
  // We use lock rather than atomic for generic values
 public:
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

  static StatRegistry<T>& get() {
    static StatRegistry<T> r;
    return r;
  }

  StatValue<T>* add(const std::string& name) {
    std::lock_guard<std::mutex> lg(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) {
      return it->second.get();
    }
    auto v = std::unique_ptr<StatValue<T>>(new StatValue<T>);
    VLOG(0) << "Register Stats: " << name;
    auto value = v.get();
    stats_.insert(std::make_pair(name, std::move(v)));
    return value;
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
  std::unordered_map<std::string, std::unique_ptr<StatValue<T>>> stats_;
};

template <typename T>
class Stat {
 public:
  explicit Stat(const std::string& n)
      : name(n), value_(StatRegistry<T>::get().add(n)) {}

  T increase(T inc) { return value_->increase(inc); }
  T decrease(T inc) { return value_->decrease(inc); }
  T reset(T value) { return value_->reset(value); }
  T get() const { return value_->get(); }

 private:
  std::string name;
  StatValue<T>* value_;
};

// Because we only support these two types in pybind
#define REGISTER_FLOAT_STATUS(item) static Stat<float> _##item(#item);

#define REGISTER_INT_STATUS(item) static Stat<int64_t> _##item(#item);

#define STAT_ADD(item, t) paddle::platform::_##item.increase(t)
#define STAT_SUB(item, t) paddle::platform::_##item.decrease(t)

// Support add stat value by string
#define STAT_INT_ADD(item, t) \
  paddle::platform::StatRegistry<int64_t>::get().add(item)->increase(t)
#define STAT_INT_SUB(item, t) \
  paddle::platform::StatRegistry<int64_t>::get().add(item)->decrease(t)

#define STAT_FLOAT_ADD(item, t) \
  paddle::platform::StatRegistry<float>::get().add(item)->increase(t)
#define STAT_FLOAT_SUB(item, t) \
  paddle::platform::StatRegistry<float>::get().add(item)->decrease(t)

#define STAT_RESET(item, t) paddle::platform::_##item.reset(t)

#define STAT_GET(item) paddle::platform::_##item.get()

// Register your own monitor stats here

REGISTER_INT_STATUS(STAT_total_feasign_num_in_mem)
REGISTER_INT_STATUS(STAT_gpu0_mem_size)
REGISTER_INT_STATUS(STAT_gpu1_mem_size)
REGISTER_INT_STATUS(STAT_gpu2_mem_size)
REGISTER_INT_STATUS(STAT_gpu3_mem_size)
REGISTER_INT_STATUS(STAT_gpu4_mem_size)
REGISTER_INT_STATUS(STAT_gpu5_mem_size)
REGISTER_INT_STATUS(STAT_gpu6_mem_size)
REGISTER_INT_STATUS(STAT_gpu7_mem_size)

}  // namespace platform
}  // namespace paddle
