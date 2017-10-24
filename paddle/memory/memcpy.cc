/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/memory/memcpy.h"

#include <cstring>  // for memcpy

namespace paddle {
namespace memory {

template <>
void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace, void* dst,
                                                  platform::CPUPlace,
                                                  const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_CUDA
template <>
void Copy<platform::CPUPlace, platform::GPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::GPUPlace src_place,
                                                  const void* src, size_t num,
                                                  cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);
  platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
}

template <>
void Copy<platform::GPUPlace, platform::CPUPlace>(platform::GPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num,
                                                  cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
}

template <>
void Copy<platform::GPUPlace, platform::GPUPlace>(platform::GPUPlace dst_place,
                                                  void* dst,
                                                  platform::GPUPlace src_place,
                                                  const void* src, size_t num,
                                                  cudaStream_t stream) {
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
  } else {
    platform::GpuMemcpyPeer(dst, dst_place.device, src, src_place.device, num,
                            stream);
  }
}

template <>
void Copy<platform::CPUPlace, platform::GPUPlace>(platform::CPUPlace dst_place,
                                                  void* dst,
                                                  platform::GPUPlace src_place,
                                                  const void* src, size_t num) {
  platform::SetDeviceId(src_place.device);
  platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
}

template <>
void Copy<platform::GPUPlace, platform::CPUPlace>(platform::GPUPlace dst_place,
                                                  void* dst,
                                                  platform::CPUPlace src_place,
                                                  const void* src, size_t num) {
  platform::SetDeviceId(dst_place.device);
  platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
}

template <>
void Copy<platform::GPUPlace, platform::GPUPlace>(platform::GPUPlace dst_place,
                                                  void* dst,
                                                  platform::GPUPlace src_place,
                                                  const void* src, size_t num) {
  platform::SetDeviceId(dst_place.device);
  platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
}

#endif

}  // namespace memory
}  // namespace paddle
