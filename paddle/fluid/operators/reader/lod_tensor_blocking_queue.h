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
  LoDTensorBlockingQueue(size_t capacity,
                         const std::vector<framework::DDim>& dims)
      : queue_(capacity), dims_(dims) {}

 public:
  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    CheckDims(lod_tensor_vec);
    return queue_.Send(lod_tensor_vec);
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    CheckDims(lod_tensor_vec);
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

  inline void Close() { return queue_.Close(); }

  inline bool IsClosed() const { return queue_.IsClosed(); }

 private:
  void CheckDims(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    PADDLE_ENFORCE(dims_.size() == lod_tensor_vec.size(),
                   "Expect input size is %d but found %s", dims_.size(),
                   lod_tensor_vec.size());
    for (size_t i = 0; i < dims_.size(); ++i) {
      const auto& in_dims = framework::slice_ddim(
          lod_tensor_vec[i].dims(), 1, lod_tensor_vec[i].dims().size());
      const auto& expect_dims =
          framework::slice_ddim(dims_[i], 1, dims_[i].size());
      PADDLE_ENFORCE(in_dims == expect_dims,
                     "Dims of the %d-th input tensor do not match", i);
    }
  }

  BlockingQueue<std::vector<framework::LoDTensor>> queue_;
  std::vector<framework::DDim> dims_;
};

class LoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, const std::vector<framework::DDim>& dims) {
    PADDLE_ENFORCE(
        queue_ == nullptr,
        "LoDTensorBlockingQueueHolder::InitOnce() can only be called once");
    queue_.reset(new LoDTensorBlockingQueue(capacity, dims));
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
