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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace reader {

class LoDTensorBlockingQueueHolder;

class LoDTensorBlockingQueue {
  friend class LoDTensorBlockingQueueHolder;

 private:
  explicit LoDTensorBlockingQueue(size_t capacity, bool speed_test_mode = false)
      : queue_(capacity, speed_test_mode) {}

 public:
  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    return queue_.Send(lod_tensor_vec);
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    return queue_.Send(std::move(lod_tensor_vec));
  }

  std::vector<framework::LoDTensor> Pop(bool* ok = nullptr) {
    std::vector<framework::LoDTensor> lod_tensor_vec;
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
  BlockingQueue<std::vector<framework::LoDTensor>> queue_;
};

class LoDTensorBlockingQueues {
  friend class LoDTensorBlockingQueueHolder;

 private:
  explicit LoDTensorBlockingQueues(size_t cpu_num, size_t capacity,
                                   bool speed_test_mode = false) {
    for (size_t x = 0; x < cpu_num; x++) {
      auto q =
          std::make_shared<BlockingQueue<std::vector<framework::LoDTensor>>>(
              BlockingQueue<std::vector<framework::LoDTensor>>(
                  capacity, speed_test_mode));
      queues_.push_back(q);
    }
  }

 public:
  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    return queues_[0]->Send(lod_tensor_vec);
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    return queues_[0]->Send(std::move(lod_tensor_vec));
  }

  std::vector<framework::LoDTensor> Pop(bool* ok = nullptr) {
    std::vector<framework::LoDTensor> lod_tensor_vec;
    bool success = queues_[0]->Receive(&lod_tensor_vec);
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
      if (!q->IsClosed()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<BlockingQueue<std::vector<framework::LoDTensor>>>>
      queues_;
};

class LoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE(
        queue_ == nullptr,
        "LoDTensorBlockingQueueHolder::InitOnce() can only be called once");
    size_t cpu_num = 1;
    queue_.reset(
        new LoDTensorBlockingQueues(cpu_num, capacity, speed_test_mode));
  }

  inline const std::shared_ptr<LoDTensorBlockingQueues>& GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<LoDTensorBlockingQueues> queue_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
