/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/stream.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"

namespace phi {

// Currently, CudaStream is used in python-side API only
class CUDAStream {
 public:
  enum class StreamFlag : uint8_t {
    kDefaultFlag = 0x0,
    kStreamNonBlocking = 0x1,
  };

 public:
  CUDAStream(const Place& place, const Stream& stream)
      : place_(place), stream_(stream) {}
  CUDAStream(const Place& place,
             const int priority = 0,
             const StreamFlag& flag = StreamFlag::kDefaultFlag) {
    place_ = place;
    gpuStream_t stream = nullptr;
    backends::gpu::GPUDeviceGuard guard(place_.device);

    // Stream priorities follow a convention where lower numbers imply greater
    // priorities
    auto priority_range = backends::gpu::GetGpuStreamPriorityRange();
    int least_priority = priority_range.first;      // 0 in V100
    int greatest_priority = priority_range.second;  // -5 in V100

    // NOTE(Ruibiao): Replacing the following `PADDLE_ENFORCE_EQ` with
    // `PADDLE_ENFORCE` leads to a nvcc compile error. This is probably a bug.
    PADDLE_ENFORCE_EQ(
        priority <= least_priority && priority >= greatest_priority,
        true,
        phi::errors::InvalidArgument(
            "Cannot create a stream with priority = %d because stream priority "
            "must be inside the meaningful range [%d, %d].",
            priority,
            least_priority,
            greatest_priority));

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreateWithPriority(
        &stream, static_cast<unsigned int>(flag), priority));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(
        &stream, static_cast<unsigned int>(flag), priority));
#endif

    VLOG(10) << "Create CUDAStream " << stream
             << " with priority = " << priority
             << ", flag = " << static_cast<unsigned int>(flag);
    stream_ = Stream(reinterpret_cast<StreamId>(stream));
    owned_ = true;
  }

  gpuStream_t raw_stream() const { return reinterpret_cast<gpuStream_t>(id()); }

  void set_raw_stream(gpuStream_t stream) {
    if (owned_ && stream_.id() != 0) {
      backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(raw_stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(raw_stream()));
#endif
    }
    stream_ = Stream(reinterpret_cast<StreamId>(stream));
  }

  StreamId id() const { return stream_.id(); }

  Place place() const { return place_; }

  bool Query() const {
#ifdef PADDLE_WITH_HIP
    hipError_t err = hipStreamQuery(raw_stream());
    if (err == hipSuccess) {
      return true;
    }
    if (err == hipErrorNotReady) {
      return false;
    }
#else
    cudaError_t err = cudaStreamQuery(raw_stream());
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

  void Synchronize() const {
    VLOG(10) << "Synchronize " << raw_stream();
    backends::gpu::GpuStreamSync(raw_stream());
  }

  void WaitEvent(gpuEvent_t ev) const {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(raw_stream(), ev, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(raw_stream(), ev, 0));
#endif
  }

  ~CUDAStream() {
    VLOG(10) << "~CUDAStream " << raw_stream();
    Synchronize();
    if (owned_ && stream_.id() != 0) {
      backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP
      hipStreamDestroy(raw_stream());
#else
      cudaStreamDestroy(raw_stream());
#endif
    }
  }

 private:
  Place place_;
  Stream stream_;
  bool owned_{false};  // whether the stream is created and onwed by self
};

}  // namespace phi
