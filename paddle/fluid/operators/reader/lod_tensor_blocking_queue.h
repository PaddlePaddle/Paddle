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
#include <vector>

#include "paddle/fluid/framework/blockingconcurrentqueue.h"
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
      : queue_(capacity), capacity_(capacity) {}

 public:
  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    return queue_.enqueue(lod_tensor_vec);
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    return queue_.enqueue(std::move(lod_tensor_vec));
  }

  std::vector<framework::LoDTensor> Pop(bool* ok = nullptr) {
    std::vector<framework::LoDTensor> lod_tensor_vec;
    while (!queue_.wait_dequeue_timed(lod_tensor_vec, 1000)) {
      if (queue_.is_closed()) {
        break;
      }
    }
    bool success = true;
    if (ok != nullptr) *ok = success;
    return lod_tensor_vec;
  }

  inline size_t Cap() const { return capacity_; }

  inline size_t Size() const { return queue_.size_approx(); }

  inline void ReOpen() { queue_.open(); }

  inline void Close() { queue_.close(); }

  inline bool IsClosed() const { return queue_.is_closed(); }

 private:
  // BlockingQueue<std::vector<framework::LoDTensor>> queue_;
  moodycamel::BlockingConcurrentQueue<std::vector<framework::LoDTensor>> queue_;
  size_t capacity_;
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

}  // namespace reader
}  // namespace operators
}  // namespace paddle
