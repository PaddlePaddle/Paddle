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

#pragma once
#include <glog/logging.h>
#include <time.h>
#include <algorithm>
#include <chrono>  // NOLINT
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace profile {

template <typename ChildT>
class ProfileRecordBase {
 public:
  void Start() { self()->Start(); }
  void Stop() { self()->Stope(); }

 protected:
  ChildT *self() { return reinterpret_cast<ChildT *>(this); }
};

class BasicRecord : ProfileRecordBase<BasicRecord> {
  uint64_t total_{};
  uint64_t count_{};
  uint32_t max_{};
  uint32_t min_{};
  int id_{-1};
  std::string key_;
  std::chrono::time_point<std::chrono::high_resolution_clock> timer_{};

 public:
  BasicRecord(int id, const std::string &key) : id_(id), key_(key) {}

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

  const std::string &key() const { return key_; }

  int id() const {
    CHECK_GE(id_, 0) << "id is not inited";
    return id_;
  }

  double ave() const { return total_ * 1. / count_; }
  double max() const { return max_; }
  double min() const { return min_; }

  // BasicRecord(const BasicRecord &) = delete;
  void operator=(const BasicRecord &) = delete;
};

/*
 * A basic profiler, with each record logs the total latency.
 */
class BasicProfiler {
 public:
  explicit BasicProfiler(const std::string &name) : name_(name) {}
  using record_t = BasicRecord;

  static BasicProfiler &Global() {
    static std::unique_ptr<BasicProfiler> x(new BasicProfiler("[global]"));
    return *x;
  }

  record_t &NewRcd(const std::string &key) {
    records_.emplace_back(records_.size(), key);
    return records_.back();
  }

  const record_t &record(int id) {
    CHECK_LT(id, records_.size());
    CHECK_GE(id, 0);
    return records_[id];
  }

  record_t *mutable_record(int id) {
    CHECK_LT(id, records_.size());
    CHECK_GE(id, 0);
    return &records_[id];
  }

  std::string basic_repr() const {
    std::stringstream ss;

    for (const auto &rcd : records_) {
      ss << rcd.key() << "\t" << rcd.ave() << "\t" << rcd.max() << "\t"
         << rcd.min() << "\t" << rcd.count() << "\n";
    }
    return ss.str();
  }

 private:
  std::string name_;
  std::vector<record_t> records_;
};

template <typename RecordT>
struct Profile {
  explicit Profile(int id) : id_(id) {
    BasicProfiler::Global().mutable_record(id_)->Start();
  }

  ~Profile() { BasicProfiler::Global().mutable_record(id_)->Stop(); }

 private:
  int id_{};
};

#define LITE_PROFILE_ONE(key__)                                  \
  static int key__##__profiler_id =                              \
      BasicProfiler::Global().NewRcd(#key__).id();               \
  Profile<paddle::lite::profile::BasicRecord> key__##profiler__( \
      key__##__profiler_id);

}  // namespace profile
}  // namespace lite
}  // namespace paddle
