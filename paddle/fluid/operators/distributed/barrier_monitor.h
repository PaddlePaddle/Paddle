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

#include <chrono>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum BarrierType { kSendBarrier, kRecvBarrier };

constexpr int64_t kMaxWaitMS = 1000;

class BarrierMonitor {
 public:
  explicit BarrierMonitor(int workers) : workers_(workers) {
    PADDLE_ENFORCE_GT(workers, 0, "trainers must have one or more");

    barrier_type = kRecvBarrier;
    running_ = true;
    monitor_thread_.reset(
        new std::thread(std::bind(&BarrierMonitor::Monitor, this)));
  }

  ~BarrierMonitor() {
    running_ = false;
    if (monitor_thread_) monitor_thread_->join();
  }

  static void Init(int workers) {
    std::call_once(init_flag_, &BarrierMonitor::InitImpl, workers);
  }

  static BarrierMonitor *GetInstance() { return monitor_.get(); }

  bool IncreaseBarrier(const int worker_id, const std::string &barrier);

  void Monitor();

  void Swap();

  bool IsReady();

  void Invalid();

  bool Wait() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [this] { return (valid); });
  }

 private:
  // Init is called by GetInstance.
  static void InitImpl(int workers) {
    if (monitor_ == nullptr) {
      monitor_.reset(new BarrierMonitor(workers));
    }
  }

  static std::once_flag init_flag_;
  static std::unique_ptr<BarrierMonitor> monitor_;
  int workers_;
  bool working_ = false;
  bool running_ = false;
  bool valid = false;

  std::condition_variable cv_;
  std::mutex mutex_;
  BarrierType barrier_type;
  std::unique_ptr<std::thread> monitor_thread_{nullptr};
  std::shared_ptr<framework::BlockingQueue<int>> send_barrier_queue;
  std::shared_ptr<framework::BlockingQueue<int>> recv_barrier_queue;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
