/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream_callback_manager.h"

namespace paddle {
namespace platform {
namespace stream {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

enum class Priority : uint8_t {
  kNull = 0x0,
  kHigh = 0x1,
  kNormal = 0x2,
};

enum class StreamFlag : uint8_t {
  kDefaultFlag = 0x0,
  kStreamNonBlocking = 0x1,
};

#endif
class CUDAStream final {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

 public:
  CUDAStream() = default;
  explicit CUDAStream(const Place& place,
                      const Priority& priority = Priority::kNormal,
                      const StreamFlag& flag = StreamFlag::kDefaultFlag) {
    Init(place, priority, flag);
  }
  explicit CUDAStream(gpuStream_t stream, const Place& place)
      : place_(place), stream_(stream) {
    owned_stream_ = false;
    callback_manager_.reset(new StreamCallbackManager<gpuStream_t>(stream_));
  }
  virtual ~CUDAStream() { Destroy(); }

  bool Init(const Place& place, const Priority& priority = Priority::kNormal,
            const StreamFlag& flag = StreamFlag::kDefaultFlag);

  void AddCallback(std::function<void()> callback) const {
    callback_manager_->AddCallback(callback);
  }

#ifdef PADDLE_WITH_HIP
  void RecordEvent(hipEvent_t ev, const std::function<void()>& callback) const {
    callback();
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(ev, stream_));
  }
#else
  void RecordEvent(cudaEvent_t ev,
                   const std::function<void()>& callback) const {
    callback();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ev, stream_));
  }
#endif

#ifdef PADDLE_WITH_HIP
  void RecordEvent(hipEvent_t ev) const {
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(ev, stream_));
  }
#else
  void RecordEvent(cudaEvent_t ev) const {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ev, stream_));
  }
#endif

#ifdef PADDLE_WITH_HIP
  void WaitEvent(hipEvent_t ev) const {
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(stream_, ev, 0));
  }
#else
  void WaitEvent(cudaEvent_t ev) const {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream_, ev, 0));
  }
#endif

  void Wait() const;
  void WaitCallback() const { callback_manager_->Wait(); }

#ifdef PADDLE_WITH_HIP
  const hipStream_t& raw_stream() const { return stream_; }
#else
  const cudaStream_t& raw_stream() const { return stream_; }
#endif
  void Destroy();

  bool Query() const {
#ifdef PADDLE_WITH_HIP
    hipError_t err = hipStreamQuery(stream_);
    if (err == hipSuccess) {
      return true;
    }
    if (err == hipErrorNotReady) {
      return false;
    }
#else
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) {
      return true;
    }
    if (err == cudaErrorNotReady) {
      return false;
    }
#endif

    PADDLE_ENFORCE_GPU_SUCCESS(err);
    return false;
  }

  void Synchronize() const { platform::GpuStreamSync(stream_); }

  const Place& GetPlace() const { return place_; }

  // Note: Can only be used under thread_local semantics.
  void SetStream(gpuStream_t stream);

 private:
  Place place_;
  bool owned_stream_{true};
#ifdef PADDLE_WITH_HIP
  hipStream_t stream_{nullptr};
#else
  cudaStream_t stream_{nullptr};
#endif
  Priority priority_{Priority::kNormal};
  std::unique_ptr<StreamCallbackManager<gpuStream_t>> callback_manager_;
#endif
  DISABLE_COPY_AND_ASSIGN(CUDAStream);
};

CUDAStream* get_current_stream(int deviceId);
// NOTE: There is a problem with the interface and needs to be fixed
CUDAStream* set_current_stream(CUDAStream* stream);

}  // namespace stream
}  // namespace platform
}  // namespace paddle
