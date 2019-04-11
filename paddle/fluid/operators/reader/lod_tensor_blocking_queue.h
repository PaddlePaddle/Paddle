//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace reader {

inline size_t GetHashThreadId() {
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

typedef std::vector<framework::LoDTensor> BATCH;

class LoDTensorBlockingQueueHolder;
class LoDTensorBlockingQueuesHolder;

class LoDTensorBlockingQueue {
  friend class LoDTensorBlockingQueueHolder;

 private:
  explicit LoDTensorBlockingQueue(size_t capacity, bool speed_test_mode = false)
      : queue_(capacity, speed_test_mode) {}

 public:
  bool Push(const BATCH& lod_tensor_vec) { return queue_.Send(lod_tensor_vec); }

  bool Push(BATCH&& lod_tensor_vec) {
    return queue_.Send(std::move(lod_tensor_vec));
  }

  BATCH Pop(bool* ok = nullptr) {
    BATCH lod_tensor_vec;
    bool success = queue_.Receive(&lod_tensor_vec);
    if (ok != nullptr) *ok = success;
    return lod_tensor_vec;
  }

  inline size_t Cap() const { return queue_.Cap(); }

  inline size_t Size() const { return queue_.Size(); }

  inline void ReOpen() { queue_.ReOpen(); }

  inline void Close() {
    VLOG(1) << "LoDTensorBlockingQueue close";
    queue_.Close();
  }

  inline bool IsClosed() const { return queue_.IsClosed(); }

 private:
  BlockingQueue<BATCH> queue_;
};

class LoDTensorBlockingQueues {
  friend class LoDTensorBlockingQueuesHolder;

 private:
  explicit LoDTensorBlockingQueues(size_t parallelism_num, size_t capacity,
                                   bool speed_test_mode = false) {
    VLOG(3) << "Init LoDTensorBlockingQueues parallelism_num="
            << parallelism_num << ", capacity=" << capacity
            << ", speed_test_mode" << speed_test_mode;

    for (size_t x = 0; x < parallelism_num; x++) {
      auto q = std::shared_ptr<BlockingQueue<BATCH>>(
          new BlockingQueue<BATCH>(capacity, speed_test_mode));
      queues_.push_back(q);
    }
    current_queue_ = std::shared_ptr<BlockingQueue<BATCH>>(
        new BlockingQueue<BATCH>(capacity, speed_test_mode));
    current_idx_ = 0;
  }

 public:
  bool Push(const int queue_id, const BATCH& lod_tensor_vec) {
    return queues_[queue_id]->Send(lod_tensor_vec);
  }

  bool Push(const int queue_id, BATCH&& lod_tensor_vec) {
    return queues_[queue_id]->Send(std::move(lod_tensor_vec));
  }

  BATCH Pop(bool* ok = nullptr) {
    size_t q_size = current_queue_->Size();
    BATCH lod_tensor_vec;

    while (current_queue_->Size() == 0) {
      if (IsClosed()) {
        *ok = false;
        return lod_tensor_vec;
      }
      Swap();
    }

    VLOG(1) << "swap current queue size, before=" << q_size
            << " after=" << current_queue_->Size();

    bool success = current_queue_->Receive(&lod_tensor_vec);
    if (ok != nullptr) *ok = success;
    return lod_tensor_vec;
  }

  inline size_t Cap() {
    size_t cap = 0;

    for (auto& q : queues_) {
      cap += q->Cap();
    }
    return cap;
  }

  inline size_t Size() {
    size_t size = 0;

    for (auto& q : queues_) {
      size += q->Size();
    }
    return size;
  }

  inline void ReOpen() {
    for (auto& q : queues_) {
      q->ReOpen();
    }
  }

  inline void Close() {
    VLOG(1) << "LoDTensorBlockingQueue close";
    for (auto& q : queues_) {
      q->Close();
    }
  }

  inline bool IsClosed() {
    for (auto& q : queues_) {
      auto ok = q->IsClosed() && q->Size() == 0;

      if (!ok) {
        return false;
      }
    }
    return true;
  }

  inline size_t Queues() const { return queues_.size(); }

  inline std::shared_ptr<BlockingQueue<BATCH>>& Get(size_t idx) {
    return queues_[idx];
  }

 private:
  inline void Swap() {
    size_t q_size = Queues();
    for (auto x = 0; x < q_size; ++x) {
      current_idx_ = (current_idx_ + 1) % q_size;
      auto& q_ = queues_[current_idx_];

      if (q_->Size() == 0) {
        continue;
      }
      current_queue_->Swap(q_.get());
      break;
    }
  }

 private:
  std::vector<std::shared_ptr<BlockingQueue<BATCH>>> queues_;
  std::shared_ptr<BlockingQueue<BATCH>> current_queue_;
  int current_idx_;
};

class LoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE(
        queue_ == nullptr,
        "LoDTensorBlockingQueueHolder::InitOnce() can only be called once");
    queue_.reset(new LoDTensorBlockingQueue(capacity, speed_test_mode));
  }

  inline const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

class LoDTensorBlockingQueuesHolder {
 public:
  void InitOnce(size_t thread_num, size_t parallelism_num, size_t capacity,
                bool speed_test_mode = false) {
    PADDLE_ENFORCE(
        queue_.empty(),
        "LoDTensorBlockingQueuesHolder::InitOnce() can only be called once");

    for (size_t x = 0; x < thread_num; x++) {
      auto q =
          std::shared_ptr<LoDTensorBlockingQueues>(new LoDTensorBlockingQueues(
              parallelism_num, capacity, speed_test_mode));
      queue_.push_back(q);
    }
  }

  inline const std::vector<std::shared_ptr<LoDTensorBlockingQueues>>& GetQueue()
      const {
    return queue_;
  }

 private:
  std::vector<std::shared_ptr<LoDTensorBlockingQueues>> queue_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
