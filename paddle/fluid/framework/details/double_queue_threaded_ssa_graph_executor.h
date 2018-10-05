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

#include <algorithm>
#include <atomic>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ThreadPool.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

namespace paddle {
namespace framework {
namespace details {

class OpHandleBase;
class DoubleQueueThreadedSSAGraphExecutor;

namespace internal {

class ControlFlag {
 public:
  inline bool Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return flag_ != kPause; });
    return flag_ != kStop;
  }

  inline void Start() {
    flag_ = kStart;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_all();
    }
  }

  inline bool IsStarted() const { return flag_ == kStart; }

  inline void Pause() { flag_ = kPause; }

  inline bool IsPaused() const { return flag_ == kPause; }

  inline void Stop() {
    flag_ = kStop;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_all();
    }
  }

  inline bool IsStopped() const { return flag_ == kStop; }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;

  using Status = uint8_t;
  std::atomic<Status> flag_{kPause};

  static constexpr Status kPause = 0;
  static constexpr Status kStart = 1;
  static constexpr Status kStop = 2;
};

template <typename T>
struct SimpleQueue {
  inline void MarkAsInit() { init_size_ = tail_ - head_; }

  inline void ResetAndReserve(size_t new_size) {
    new_size = std::max(init_size_, new_size);
    if (new_size > queue_.size()) {
      queue_.resize(init_size_);  // avoid unnecessary copy
      queue_.resize(new_size);
    }
    head_ = 0;
    tail_ = init_size_;
  }

  inline void Push(const T &obj) { queue_[tail_++] = obj; }

  // Ensure [from, to) is valid
  inline void Push(const SimpleQueue<T> &other, size_t from, size_t to) {
    while (from < to) {
      queue_[tail_++] = other.queue_[from++];
    }
  }

  template <typename Container>
  inline void PushAll(const Container &other) {
    for (auto &obj : other) {
      queue_[tail_++] = obj;
    }
  }

  inline T Pop() { return queue_[head_++]; }

  inline size_t PopN(size_t n, size_t *head) {
    size_t pop_num = std::min(n, tail_ - head_);
    *head = head_;
    head_ += pop_num;
    return pop_num;
  }

  inline bool IsEmpty() const { return head_ == tail_; }

  inline size_t size() const { return tail_ - head_; }

  std::vector<T> queue_;
  size_t init_size_{0};
  size_t head_{0};
  size_t tail_{0};
};

}  // namespace internal

class DoubleQueueThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  DoubleQueueThreadedSSAGraphExecutor(
      const ExecutionStrategy &strategy,
      const std::vector<Scope *> &local_scopes,
      const std::vector<platform::Place> &places,
      std::unique_ptr<ir::Graph> &&graph);

  ~DoubleQueueThreadedSSAGraphExecutor();

  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override;

  const ir::Graph &Graph() const override;

 private:
  void PrepareNext();

  const ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  std::unique_ptr<ir::Graph> graph_;
  platform::DeviceContextPool fetch_ctxs_;

  ::ThreadPool prepare_next_pool_;
  std::future<void> prepare_next_future_;

  // members of queues
  std::vector<internal::SimpleQueue<OpHandleBase *>> local_queues_;
  internal::SimpleQueue<OpHandleBase *> global_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  size_t init_queue_idx_{0};

  // members for thread control
  std::vector<std::thread> thread_pool_;
  internal::ControlFlag control_flag_;
  std::mutex exit_mutex_;
  std::condition_variable exit_cv_;
  std::atomic<size_t> run_op_num_{0};
  std::atomic<size_t> running_threads_{0};
  std::exception_ptr exception_ptr_;

  std::unordered_map<OpHandleBase *, size_t> op_deps_;
  std::unordered_map<OpHandleBase *, std::unordered_set<OpHandleBase *>>
      pending_ops_;
  std::unique_ptr<
      std::unordered_map<OpHandleBase *, std::unordered_set<OpHandleBase *>>>
      rt_pending_ops_;
  std::unique_ptr<std::unordered_map<OpHandleBase *, std::atomic<size_t>>>
      rt_op_deps_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
