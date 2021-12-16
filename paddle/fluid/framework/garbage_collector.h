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

#include <deque>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

class GarbageCollector {
 public:
  using GarbageQueue = std::deque<std::shared_ptr<memory::Allocation>>;

  GarbageCollector(const platform::Place &place, size_t max_memory_size);

  virtual ~GarbageCollector() PADDLE_MAY_THROW {}

  virtual void Wait() const {}

  template <typename Container>
  void Add(Container &&objs);

  template <typename Container, typename Callback>
  void Add(Container &&objs, Callback &&callback);

  void DirectClearCallback(const std::function<void()> &callback) {
    ClearCallback(callback);
  }

 protected:
  virtual void ClearCallback(const std::function<void()> &callback) = 0;

  platform::DeviceContext *dev_ctx_;
  std::unique_ptr<GarbageQueue> garbages_;
  mutable std::unique_ptr<std::mutex> mutex_;
  const size_t max_memory_size_;
  size_t cur_memory_size_{0};
};

class CPUGarbageCollector : public GarbageCollector {
 public:
  CPUGarbageCollector(const platform::CPUPlace &place, size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};

#ifdef PADDLE_WITH_XPU
class XPUGarbageCollector : public GarbageCollector {
 public:
  XPUGarbageCollector(const platform::XPUPlace &place, size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};
#endif

#ifdef PADDLE_WITH_IPU
class IPUGarbageCollector : public GarbageCollector {
 public:
  IPUGarbageCollector(const platform::IPUPlace &place, size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class UnsafeFastGPUGarbageCollector : public GarbageCollector {
 public:
  UnsafeFastGPUGarbageCollector(const platform::CUDAPlace &place,
                                size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};

class DefaultStreamGarbageCollector : public GarbageCollector {
 public:
  DefaultStreamGarbageCollector(const platform::CUDAPlace &place,
                                size_t max_memory_size);

  void Wait() const override;

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};

class StreamGarbageCollector : public GarbageCollector {
 public:
  StreamGarbageCollector(const platform::CUDAPlace &place,
                         size_t max_memory_size);

  ~StreamGarbageCollector();

  void Wait() const override;

  gpuStream_t stream() const;

 protected:
  void ClearCallback(const std::function<void()> &callback) override;

 private:
  gpuStream_t stream_;
  std::unique_ptr<platform::StreamCallbackManager<gpuStream_t>>
      callback_manager_;
};

class CUDAPinnedGarbageCollector : public GarbageCollector {
 public:
  CUDAPinnedGarbageCollector(const platform::CUDAPinnedPlace &place,
                             size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};
#endif

#ifdef PADDLE_WITH_ASCEND_CL
class NPUDefaultStreamGarbageCollector : public GarbageCollector {
 public:
  NPUDefaultStreamGarbageCollector(const platform::NPUPlace &place,
                                   size_t max_memory_size);

  void Wait() const override;

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};

class NPUUnsafeFastGarbageCollector : public GarbageCollector {
 public:
  NPUUnsafeFastGarbageCollector(const platform::NPUPlace &place,
                                size_t max_memory_size);

 protected:
  void ClearCallback(const std::function<void()> &callback) override;
};
#endif

template <typename Container>
void GarbageCollector::Add(Container &&objs) {
  Add(std::forward<Container>(objs), []() {});
}

template <typename Container, typename Callback>
void GarbageCollector::Add(Container &&objs, Callback &&callback) {
  // Special case when FLAGS_eager_delete_tensor_gb=0.0
  // It speeds up GC about 2~3%.
  if (max_memory_size_ <= 1) {
    callback();
    auto *container = new Container(std::move(objs));
    ClearCallback([container] { delete container; });
    return;
  }

  GarbageQueue *garbage_queue = nullptr;
  {
    std::lock_guard<std::mutex> guard(*mutex_);
    for (auto &obj : objs) {
      if (!obj) continue;
      cur_memory_size_ += obj->size();
      garbages_->push_back(std::move(obj));
    }
    if (cur_memory_size_ >= max_memory_size_) {
      cur_memory_size_ = 0;
      garbage_queue = garbages_.release();
      garbages_.reset(new GarbageQueue());
    }
  }

  if (garbage_queue) {
    callback();
    ClearCallback([garbage_queue]() { delete garbage_queue; });
  }
}

int64_t GetEagerDeletionThreshold();
bool IsFastEagerDeletionModeEnabled();

void SetEagerDeletionMode(double threshold, double fraction, bool fast_mode);

double GetEagerDeletionMemoryFraction();

}  // namespace framework
}  // namespace paddle
