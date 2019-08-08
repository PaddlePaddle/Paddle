/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/memcpy.h"

#include <cstring>  // for memcpy
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace memory {

template <>
void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace, void* dst,
                                                  platform::CPUPlace,
                                                  const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_CUDA
static constexpr size_t kMaxGpuAsyncCopyBytes = 64 * 1024;  // 64K

// NOTE(zcd): Do not use GpuMemcpySync as much as possible.
// because GpuMemcpySync issues the copying command to the default stream,
// which will make two commands from different streams cannot run concurrently.
// Reference:
// https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

template <>
void Copy<platform::CPUPlace, platform::CUDAPlace>(
    platform::CPUPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);

  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CPU");
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CPU");
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      cudaStreamSynchronize(0);
    }
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CPUPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CPU->GPU");
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CPU->GPU");
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
    // FIXME(zjl): do we really need it?
    if (num <= kMaxGpuAsyncCopyBytes) {
      cudaStreamSynchronize(0);
    }
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyAsync(same_gpu):GPU->GPU");
      platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
    } else {
      platform::RecordEvent record_event("GpuMemcpySync(same_gpu):GPU->GPU");
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
    }
  } else {
    if (stream) {
      platform::RecordEvent record_event("GpuMemcpyPeerAsync:GPU->GPU");
      platform::GpuMemcpyPeerAsync(dst, dst_place.device, src, src_place.device,
                                   num, stream);
    } else {
      platform::RecordEvent record_event("GpuMemcpyPeerSync:GPU->GPU");
      platform::GpuMemcpyPeerSync(dst, dst_place.device, src, src_place.device,
                                  num);
    }
  }
}

template <>
void Copy<platform::CPUPlace, platform::CUDAPinnedPlace>(
    platform::CPUPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CPUPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CPUPlace src_place, const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

template <>
void Copy<platform::CUDAPinnedPlace, platform::CUDAPlace>(
    platform::CUDAPinnedPlace dst_place, void* dst,
    platform::CUDAPlace src_place, const void* src, size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:GPU->CUDAPinned");
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:GPU->CUDAPinned");
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
  }
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPinnedPlace>(
    platform::CUDAPlace dst_place, void* dst,
    platform::CUDAPinnedPlace src_place, const void* src, size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  if (stream) {
    platform::RecordEvent record_event("GpuMemcpyAsync:CUDAPinned->GPU");
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
  } else {
    platform::RecordEvent record_event("GpuMemcpySync:CUDAPinned->GPU");
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
  }
}

#endif

}  // namespace memory
}  // namespace paddle
