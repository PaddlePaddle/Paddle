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

#include <atomic>
#include <vector>
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace reader {

class QueueBasedReader final : public framework::FileReader {
 public:
  explicit QueueBasedReader(
      const std::shared_ptr<LoDTensorBlockingQueue>& queue)
      : framework::FileReader() {
    PADDLE_ENFORCE_NOT_NULL(queue, "LoDTensorBlockingQueue must not be null");
    queue_ = queue;
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;
    *out = queue_->Pop(&success);
    if (!success) out->clear();
  }

  ~QueueBasedReader() { queue_->Close(); }

  void Shutdown() override { queue_->Close(); }

  void Start() override { queue_->ReOpen(); }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

class MultiDeviceLoDTensorBlockingQueueHolder {
 public:
  MultiDeviceLoDTensorBlockingQueueHolder(size_t dev_cnt, size_t capacity,
                                          bool speed_test_mode = false) {
    PADDLE_ENFORCE(
        queues_.empty(),
        "MultiDeviceQueueHolder::InitOnce() can only be called once");
    readers_.resize(dev_cnt);
    queues_.reserve(dev_cnt);
    for (size_t i = 0; i < dev_cnt; ++i) {
      queues_.emplace_back(
          new LoDTensorBlockingQueue(capacity, speed_test_mode));
    }
  }

  size_t DeviceNum() const { return queues_.size(); }

  std::shared_ptr<QueueBasedReader> CreateNextReader() {
    auto idx = cur_idx_.fetch_add(1);
    PADDLE_ENFORCE_LT(idx, queues_.size(), "No next queue found");
    auto ret = std::make_shared<QueueBasedReader>(queues_[idx]);
    readers_[idx] = ret;
    return ret;
  }

  const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue(size_t i) {
    PADDLE_ENFORCE_LT(i, queues_.size(), "Index out of bounds");
    return queues_[i];
  }

  void Close() {
    for (auto& q : queues_) {
      q->Close();
    }
  }

  void ResetReaders() {
    for (auto& r : readers_) {
      if (auto p = r.lock()) {
        ResetReader(p.get());
      }
    }
  }

 private:
  static void ResetReader(framework::ReaderBase* reader) {
    auto end_readers = reader->GetEndPoints();
    for (auto* reader : end_readers) {
      reader->Shutdown();
    }

    for (auto* reader : end_readers) {
      reader->Start();
    }
  }

  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> queues_;
  std::vector<std::weak_ptr<QueueBasedReader>> readers_;
  std::atomic<size_t> cur_idx_{0};
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
