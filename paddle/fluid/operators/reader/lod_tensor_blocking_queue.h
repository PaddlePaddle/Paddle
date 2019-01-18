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

#include <functional>
#include <memory>
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
  using TensorVec = std::vector<framework::LoDTensor>;

 private:
  LoDTensorBlockingQueue(size_t capacity,
                         const std::vector<framework::DDim>& dims,
                         bool speed_test_mode = false)
      : capacity_(capacity), dims_(dims), speed_test_mode_(speed_test_mode) {
    queues_.reserve(1);
    queues_[0].reset(new BlockingQueue<TensorVec>(capacity, speed_test_mode));
  }

 public:
  bool BufferedPush(const std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    buffer_.emplace_back(std::move(lod_tensor_vec));
    if (buffer_.size() == queues_.size()) {
      for (size_t i = 0; i < buffer_.size(); ++i) {
        queues_[i]->Send(std::move(buffer_[i]));
      }
      buffer_.clear();
      buffer_.reserve(queues_.size());
    }
    return true;
  }

  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    for (auto q : queues_) {
      if (q->IsClosed()) return false;
    }
    if (queues_.size() == 1UL) return queues_[0]->Send(lod_tensor_vec);
    return BufferedPush(std::move(lod_tensor_vec));
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    for (auto q : queues_) {
      if (q->IsClosed()) return false;
    }
    if (queues_.size() == 1UL)
      return queues_[0]->Send(std::move(lod_tensor_vec));
    return BufferedPush(std::move(lod_tensor_vec));
  }

  std::vector<framework::LoDTensor> Pop(bool* ok = nullptr, int dev_id = 0) {
    std::vector<framework::LoDTensor> lod_tensor_vec;
    PADDLE_ENFORCE_LT(static_cast<size_t>(dev_id), queues_.size(),
                      "Can not find queue for dev id: %d", dev_id);
    bool success = queues_[dev_id]->Receive(&lod_tensor_vec);
    if (ok != nullptr) *ok = success;
    return lod_tensor_vec;
  }

  inline size_t Cap() const { return queues_[0]->Cap(); }

  inline size_t Size() const { return queues_[0]->Size(); }

  inline void ReOpen() {
    for (auto q : queues_) {
      q->ReOpen();
    }
  }

  inline void Close() {
    for (auto q : queues_) {
      q->Close();
    }
  }

  inline bool IsClosed() const {
    bool closed = false;
    for (auto q : queues_) {
      closed |= q->IsClosed();
    }
    return closed;
  }

  inline void ReInitWithMultiDev(size_t place_num) {
    queues_.clear();
    queues_.reserve(place_num);
    for (size_t i = 0; i < place_num; ++i) {
      queues_.emplace_back(
          new BlockingQueue<TensorVec>(capacity_, speed_test_mode_));
    }
    buffer_.reserve(place_num);
  }

 private:
  std::vector<std::shared_ptr<BlockingQueue<TensorVec>>> queues_{nullptr};
  std::vector<TensorVec> buffer_;

  size_t capacity_;
  std::vector<framework::DDim> dims_;
  bool speed_test_mode_;
};

class LoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, const std::vector<framework::DDim>& dims,
                bool speed_test_mode = false) {
    PADDLE_ENFORCE(
        queue_ == nullptr,
        "LoDTensorBlockingQueueHolder::InitOnce() can only be called once");
    queue_.reset(new LoDTensorBlockingQueue(capacity, dims, speed_test_mode));
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
