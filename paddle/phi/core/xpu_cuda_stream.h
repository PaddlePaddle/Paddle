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

#ifdef PADDLE_WITH_XPU

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/stream.h"

namespace phi {

// Currently, XpuCudaStream is used in python-side API only
class XPUCUDAStream {
 public:
  enum class StreamFlag : uint8_t {
    kDefaultFlag = 0x0,
    kStreamNonBlocking = 0x1,
  };

 public:
  XPUCUDAStream(const Place& place, const Stream& stream)
      : place_(place), stream_(stream) {}
  XPUCUDAStream(const Place& place,
                const int priority = 0,
                const StreamFlag& flag = StreamFlag::kDefaultFlag) {
    place_ = place;
    cudaStream_t stream = nullptr;
    backends::xpu::XPUDeviceGuard guard(place_.device);

    // Stream priorities follow a convention where lower numbers imply greater
    // priorities
    auto priority_range = backends::xpu::GetXpuStreamPriorityRange();
    int least_priority = priority_range.first;
    int greatest_priority = priority_range.second;

    PADDLE_ENFORCE_EQ(
        priority <= least_priority && priority >= greatest_priority,
        true,
        common::errors::InvalidArgument(
            "Cannot create a stream with priority = %d because stream priority "
            "must be inside the meaningful range [%d, %d].",
            priority,
            least_priority,
            greatest_priority));

    PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamCreateWithPriority(
        &stream, static_cast<unsigned int>(flag), priority));

    // VLOG(10) << "Create XPUCUDAStream " << stream
    //          << " with priority = " << priority
    //          << ", flag = " << static_cast<unsigned int>(flag);
    stream_ = Stream(reinterpret_cast<StreamId>(stream));
    owned_ = true;
  }

  cudaStream_t raw_stream() const {
    return reinterpret_cast<cudaStream_t>(id());
  }

  void set_raw_stream(cudaStream_t stream) {
    if (owned_ && stream_.id() != 0) {
      backends::xpu::XPUDeviceGuard guard(place_.device);

      PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamDestroy(raw_stream()));
    }
    stream_ = Stream(reinterpret_cast<StreamId>(stream));
  }

  StreamId id() const { return stream_.id(); }

  Place place() const { return place_; }

  bool Query() const {
    cudaError_t err = cudaStreamQuery(raw_stream());
    if (err == cudaSuccess) {
      return true;
    }
    if (err == cudaErrorNotReady) {
      return false;
    }
    PADDLE_ENFORCE_XPU_SUCCESS(err);

    return false;
  }

  void Synchronize() const {
    // VLOG(10) << "Synchronize " << raw_stream();
    backends::xpu::XpuStreamSync(raw_stream());
  }

  void WaitEvent(cudaEvent_t ev) const {
    PADDLE_ENFORCE_XPU_SUCCESS(cudaStreamWaitEvent(raw_stream(), ev, 0));
  }

  ~XPUCUDAStream() {
    // VLOG(10) << "~XPUCUDAStream " << raw_stream();
    Synchronize();
    if (owned_ && stream_.id() != 0) {
      backends::xpu::XPUDeviceGuard guard(place_.device);
      cudaStreamDestroy(raw_stream());
    }
  }

 private:
  Place place_;
  Stream stream_;
  bool owned_{false};  // whether the stream is created and owned by self
};

}  // namespace phi

#endif
