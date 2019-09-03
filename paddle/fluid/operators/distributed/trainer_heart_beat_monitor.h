// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum TrainerStatus { UNINITED = 0, RUNNING, COMPLETED };

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

struct Trainer {
  int id;
  TrainerStatus status;
  double timestamp;

  Trainer() {}

  explicit Trainer(int trainer_id) {
    this->id = trainer_id;
    this->status = UNINITED;
    this->timestamp = 0;
  }
};

class TrainerHeartBeatMonitor {
 public:
  explicit TrainerHeartBeatMonitor(int trainers, bool is_chief,
                                   std::string varname)
      : trainers_(trainers),
        is_chief_(is_chief),
        varname_(varname),
        running_(true) {
    PADDLE_ENFORCE_GT(trainers, 0, "trainers must have one or more");

    for (auto trainer_id = 0; trainer_id < trainers; trainer_id++) {
      Trainer trainer(trainer_id);
      trainer_status_map_[trainer_id] = std::move(trainer);
    }

    // we define the No.0 pserver is the first parameter server
    // only No.0 will check the heartbeat of all trainers
    if (is_chief_) {
      monitor_thread_.reset(new std::thread(
          std::bind(&TrainerHeartBeatMonitor::LostTrainerMonitor, this)));
    }
  }

  ~TrainerHeartBeatMonitor() {
    running_ = false;
    if (monitor_thread_) monitor_thread_->join();
  }

  static void Init(int trainers, bool is_chief, std::string varname) {
    std::call_once(init_flag_, &TrainerHeartBeatMonitor::InitImpl, trainers,
                   is_chief, varname);
  }

  static TrainerHeartBeatMonitor* GetInstance() {
    if (monitor_ == nullptr) {
      PADDLE_THROW(
          "TrainerHeartBeatMonitor is not inited, call "
          "TrainerHeartBeatMonitor::Init first");
    }
    return monitor_.get();
  }

  void Update(const int trainer_id, std::string varname, TrainerStatus status);

  void LostTrainerMonitor();

 private:
  // Init is called by GetInstance.
  static void InitImpl(int trainers, bool is_chief, std::string varname) {
    if (monitor_ == nullptr) {
      monitor_.reset(new TrainerHeartBeatMonitor(trainers, is_chief, varname));
    }
  }

  static std::once_flag init_flag_;
  static std::unique_ptr<TrainerHeartBeatMonitor> monitor_;

  int trainers_;
  std::string varname_;
  bool is_chief_;
  std::unordered_map<int, Trainer> trainer_status_map_;
  std::unique_ptr<std::thread> monitor_thread_{nullptr};
  std::mutex mutex_;
  bool running_ = false;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
