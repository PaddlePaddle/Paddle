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
#include <deque>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

// T should have memory_size() and clear() method
template <typename T>
class GarbageCollector {
 public:
  GarbageCollector(const platform::Place &place, size_t max_memory_size)
      : max_memory_size_((std::max)(max_memory_size, static_cast<size_t>(1))) {
    garbages_.reset(new std::deque<T *>());
    dev_ctx_ = platform::DeviceContextPool::Instance().Get(place);
  }

  virtual ~GarbageCollector() {}

  void Reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    garbages_.reset(new std::deque<T *>());
    cur_memory_size_ = 0;
  }

  template <typename Container>
  void Add(const Container &objs) {
    Add(objs, []() {});
  }

  template <typename Container, typename Callback>
  void Add(const Container &objs, Callback &&callback) {
    std::shared_ptr<std::deque<T *>> clear_deque;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      for (auto *obj : objs) {
        garbages_->push_back(obj);
        cur_memory_size_ += obj->memory_size();
      }
      if (cur_memory_size_ >= max_memory_size_) {
        cur_memory_size_ = 0;
        clear_deque = garbages_;
        garbages_.reset(new std::deque<T *>());
      }
    }

    if (clear_deque != nullptr) {
      callback();
      ClearCallback([=]() {
        for (auto *obj : *clear_deque) obj->clear();
      });
    }
  }

  virtual void Wait() const {}

 protected:
  virtual void ClearCallback(const std::function<void()> &callback) = 0;

  platform::DeviceContext *dev_ctx_;
  std::shared_ptr<std::deque<T *>> garbages_;
  mutable std::mutex mutex_;
  const size_t max_memory_size_;
  size_t cur_memory_size_ = 0;
};

template <typename T>
class CPUGarbageCollector : public GarbageCollector<T> {
 public:
  CPUGarbageCollector(const platform::CPUPlace &place, size_t max_memory_size)
      : GarbageCollector<T>(place, max_memory_size) {}

 protected:
  void ClearCallback(const std::function<void()> &callback) override {
    callback();
  }
};

#ifdef PADDLE_WITH_CUDA
template <typename T>
class DefaultStreamGarbageCollector : public GarbageCollector<T> {
 public:
  DefaultStreamGarbageCollector(const platform::CUDAPlace &place,
                                size_t max_memory_size)
      : GarbageCollector<T>(place, max_memory_size) {}

  cudaStream_t stream() const {
    return static_cast<const platform::CUDADeviceContext *>(this->dev_ctx_)
        ->stream();
  }

  void Wait() const override {
    this->dev_ctx_->Wait();
    static_cast<const platform::CUDADeviceContext *>(this->dev_ctx_)
        ->WaitStreamCallback();
  }

 protected:
  void ClearCallback(const std::function<void()> &callback) override {
    static_cast<platform::CUDADeviceContext *>(this->dev_ctx_)
        ->AddStreamCallback(callback);
  }
};

template <typename T>
class StreamGarbageCollector : public GarbageCollector<T> {
 public:
  StreamGarbageCollector(const platform::CUDAPlace &place,
                         size_t max_memory_size)
      : GarbageCollector<T>(place, max_memory_size) {
    PADDLE_ENFORCE(cudaSetDevice(place.device));
    PADDLE_ENFORCE(cudaStreamCreate(&stream_));
    callback_manager_.reset(new platform::StreamCallbackManager(stream_));
  }

  ~StreamGarbageCollector() {
    auto place = boost::get<platform::CUDAPlace>(this->dev_ctx_->GetPlace());
    PADDLE_ENFORCE(cudaSetDevice(place.device));
    PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
    PADDLE_ENFORCE(cudaStreamDestroy(stream_));
  }

  void Wait() const override {
    PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
    std::lock_guard<std::mutex> guard(this->mutex_);
    callback_manager_->Wait();
  }

  cudaStream_t stream() const { return stream_; }

 protected:
  void ClearCallback(const std::function<void()> &callback) override {
    std::lock_guard<std::mutex> guard(this->mutex_);
    callback_manager_->AddCallback(callback);
  }

 private:
  cudaStream_t stream_;
  std::unique_ptr<platform::StreamCallbackManager> callback_manager_;
};
#endif

}  // namespace framework
}  // namespace paddle
