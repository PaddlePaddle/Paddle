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

/*
 * This file implements BasicProfile, a profiler that helps to profile the basic
 * CPU execution. It can display the min, max, average lantency of the execution
 * of each kernel.
 */
#pragma once
#include <glog/logging.h>
#include <time.h>
#include <algorithm>
#include <chrono>  // NOLINT
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace profile {

/* Base class of all the profile records */
template <typename ChildT>
class TimerBase {
 public:
  void Start() { self()->Start(); }
  void Stop() { self()->Stop(); }
  void Log(uint32_t x) { return self()->Log(x); }
  std::string basic_repr() const { return const_self()->basic_repr(); }

  void SetId(int id) { self()->SetId(id); }
  void SetKey(const std::string &key) { self()->SetKey(key); }

  int id() const { return const_self()->id(); }

 protected:
  ChildT *self() { return reinterpret_cast<ChildT *>(this); }
  const ChildT *const_self() const {
    return reinterpret_cast<const ChildT *>(this);
  }
};

class BasicTimer : TimerBase<BasicTimer> {
  uint64_t total_{};
  uint64_t count_{};
  uint32_t max_{std::numeric_limits<uint32_t>::min()};
  uint32_t min_{std::numeric_limits<uint32_t>::max()};
  int id_{-1};
  std::string key_;
  std::chrono::time_point<std::chrono::high_resolution_clock> timer_{};

  // TODO(Superjomn) make static
  static const int name_w;
  static const int data_w;

 public:
  BasicTimer() = default;
  BasicTimer(int id, const std::string &key) : id_(id), key_(key) {}

  void SetId(int id) { id_ = id; }
  void SetKey(const std::string &key) { key_ = key; }
  void Start() { timer_ = std::chrono::high_resolution_clock::now(); }
  void Stop() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - timer_);
    Log(duration.count());
  }

  int count() const { return count_; }

  void Log(uint32_t timespan) {
    total_ += timespan;
    max_ = std::max(max_, timespan);
    min_ = std::min(min_, timespan);
    count_++;
  }

  static std::string basic_repr_header() {
    std::stringstream ss;
    ss << std::setw(name_w) << "kernel"   //
       << std::setw(data_w) << "average"  //
       << std::setw(data_w) << "min"      //
       << std::setw(data_w) << "max"      //
       << std::setw(data_w) << "count";
    return ss.str();
  }

  std::string basic_repr() const {
    std::stringstream ss;
    ss << std::setw(name_w) << key()  //
       << std::setw(data_w) << ave()  //
       << std::setw(data_w) << min()  //
       << std::setw(data_w) << max()  //
       << std::setw(data_w) << count_;
    return ss.str();
  }

  const std::string &key() const { return key_; }

  int id() const {
    CHECK_GE(id_, 0) << "id is not inited";
    return id_;
  }

  double ave() const { return total_ * 1. / count_; }
  double max() const { return max_; }
  double min() const { return min_; }

  // BasicRecord(const BasicRecord &) = delete;
  void operator=(const BasicTimer &) = delete;
};

/*
 * A basic profiler, with each record logs the total latency.
 */
template <typename TimerT>
class BasicProfiler {
 public:
  explicit BasicProfiler(const std::string &name) : name_(name) {}
  using record_t = TimerT;

  static BasicProfiler &Global() {
    static std::unique_ptr<BasicProfiler> x(new BasicProfiler("[global]"));
    return *x;
  }

  record_t &NewRcd(const std::string &key) {
    records_.emplace_back();
    records_.back().SetId(records_.size() - 1);
    records_.back().SetKey(key);
    return records_.back();
  }

  const record_t &record(int id) {
    CHECK_LT(id, records_.size());
    CHECK_GE(id, 0);
    return records_[id];
  }

  record_t *mutable_record(int id) {
    CHECK_GE(id, 0);
    CHECK_LT(static_cast<size_t>(id), records_.size());
    return &records_[id];
  }

  std::string basic_repr() const {
    std::stringstream ss;
    for (const auto &rcd : records_) {
      ss << rcd.basic_repr() << "\n";
    }
    return ss.str();
  }

  ~BasicProfiler() {
    LOG(INFO) << "Profile dumps:";
    LOG(INFO) << "\n" + BasicTimer::basic_repr_header() + "\n" + basic_repr();
  }

 private:
  std::string name_;
  std::vector<record_t> records_;
};

struct ProfileBlock {
  explicit ProfileBlock(int id) : id_(id) {
    BasicProfiler<BasicTimer>::Global().mutable_record(id_)->Start();
  }

  ~ProfileBlock() {
    BasicProfiler<BasicTimer>::Global().mutable_record(id_)->Stop();
  }

 private:
  int id_{};
};

#define LITE_PROFILE_ONE(key__)                          \
  static int key__##__profiler_id =                      \
      ::paddle::lite::profile::BasicProfiler<            \
          ::paddle::lite::profile::BasicTimer>::Global() \
          .NewRcd(#key__)                                \
          .id();                                         \
  ::paddle::lite::profile::ProfileBlock key__##profiler__(key__##__profiler_id);

}  // namespace profile
}  // namespace lite
}  // namespace paddle
