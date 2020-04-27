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
#include <deque>
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

#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum BarrierType { kSendBarrier, kRecvBarrier };

constexpr int64_t kMaxWaitMS = 1000;

template <typename T>
class BlockingQueueForBarrier {
 public:
  explicit BlockingQueueForBarrier(size_t capacity) : capacity_(capacity) {
    PADDLE_ENFORCE_GT(capacity_, 0, "The capacity must be greater than 0.");
  }

  bool Push(const T &elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.push_back(elem);
    }
    cv_.notify_one();
    return true;
  }

  bool Push(T &&elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.emplace_back(std::move(elem));
    }
    cv_.notify_one();
    return true;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !queue_.empty(); });
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    cv_.notify_one();
    return rc;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::deque<T>().swap(queue_);
  }

 private:
  const size_t capacity_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

class BarrierMonitor {
 public:
  explicit BarrierMonitor(int workers) : workers_(workers) {
    PADDLE_ENFORCE_GT(workers, 0, "trainers must have one or more");

    barrier_type = BarrierType::kRecvBarrier;
    send_barrier_queue = std::make_shared<BlockingQueueForBarrier<int>>(
        new BlockingQueueForBarrier<int>(workers));
    recv_barrier_queue = std::make_shared<BlockingQueueForBarrier<int>>(
        new BlockingQueueForBarrier<int>(workers));

    running_ = true;
    monitor_thread_.reset(
        new std::thread(std::bind(&BarrierMonitor::Monitor, this)));
  }

  ~BarrierMonitor() {
    running_ = false;
    if (monitor_thread_) monitor_thread_->join();
  }

  static BarrierMonitor *Init(int workers) {
    std::call_once(init_flag_, &BarrierMonitor::InitImpl, workers);
    return GetInstance();
  }

  static BarrierMonitor *GetInstance() { return monitor_.get(); }

  bool IncreaseBarrier(const int worker_id, const std::string &barrier);

  void Monitor();

  void Swap();
  void Swap(BarrierType barrier);

  bool IsReady();

  void Invalid();

  bool Wait();

  void WaitBarrierDone(BarrierType barrier);

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
  bool valid_ = false;
  bool release_ = false;

  std::condition_variable cv_;
  std::mutex mutex_;
  BarrierType barrier_type;
  std::unique_ptr<std::thread> monitor_thread_{nullptr};
  std::shared_ptr<BlockingQueueForBarrier<int>> send_barrier_queue;
  std::shared_ptr<BlockingQueueForBarrier<int>> recv_barrier_queue;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
