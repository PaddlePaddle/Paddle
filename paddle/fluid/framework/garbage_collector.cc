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

#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/garbage_collector.h"

DECLARE_double(eager_delete_tensor_gb);
DECLARE_double(memory_fraction_of_eager_deletion);
DECLARE_bool(fast_eager_deletion_mode);

namespace paddle {
namespace framework {

GarbageCollector::GarbageCollector(const platform::Place &place,
                                   size_t max_memory_size)
    : max_memory_size_((std::max)(max_memory_size, static_cast<size_t>(1))) {
  garbages_.reset(new GarbageQueue());
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place);
  if (max_memory_size_ > 1) {
    mutex_.reset(new std::mutex());
  }
}

CPUGarbageCollector::CPUGarbageCollector(const platform::CPUPlace &place,
                                         size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void CPUGarbageCollector::ClearCallback(const std::function<void()> &callback) {
  callback();
}

#ifdef PADDLE_WITH_CUDA
UnsafeFastGPUGarbageCollector::UnsafeFastGPUGarbageCollector(
    const platform::CUDAPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void UnsafeFastGPUGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback();
}

DefaultStreamGarbageCollector::DefaultStreamGarbageCollector(
    const platform::CUDAPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void DefaultStreamGarbageCollector::Wait() const {
  static_cast<platform::CUDADeviceContext *>(this->dev_ctx_)
      ->WaitStreamCallback();
}

void DefaultStreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  static_cast<platform::CUDADeviceContext *>(this->dev_ctx_)
      ->AddStreamCallback(callback);
}

StreamGarbageCollector::StreamGarbageCollector(const platform::CUDAPlace &place,
                                               size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {
  platform::CUDADeviceGuard guard(place.device);
  PADDLE_ENFORCE(cudaStreamCreate(&stream_));
  callback_manager_.reset(new platform::StreamCallbackManager(stream_));
}

StreamGarbageCollector::~StreamGarbageCollector() {
  auto place = BOOST_GET_CONST(platform::CUDAPlace, this->dev_ctx_->GetPlace());
  platform::CUDADeviceGuard guard(place.device);
  PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
  PADDLE_ENFORCE(cudaStreamDestroy(stream_));
}

cudaStream_t StreamGarbageCollector::stream() const { return stream_; }

void StreamGarbageCollector::Wait() const { callback_manager_->Wait(); }

void StreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback_manager_->AddCallback(callback);
}
#endif

int64_t GetEagerDeletionThreshold() {
  return FLAGS_eager_delete_tensor_gb < 0
             ? -1
             : static_cast<int64_t>(FLAGS_eager_delete_tensor_gb *
                                    (static_cast<int64_t>(1) << 30));
}

bool IsFastEagerDeletionModeEnabled() { return FLAGS_fast_eager_deletion_mode; }

void SetEagerDeletionMode(double threshold, double fraction, bool fast_mode) {
  FLAGS_eager_delete_tensor_gb = threshold;
  FLAGS_memory_fraction_of_eager_deletion = fraction;
  FLAGS_fast_eager_deletion_mode = fast_mode;
}

double GetEagerDeletionMemoryFraction() {
  return FLAGS_memory_fraction_of_eager_deletion;
}

}  // namespace framework
}  // namespace paddle
