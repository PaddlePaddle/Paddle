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
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/string_tensor.h"

namespace phi {
namespace strings {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
__global__ void SerializeStringsData(const phi::dtype::pstring* src_str,
                                     uint8_t* strings_data,
                                     int32_t* strings_offset,
                                     int64_t numel,
                                     int32_t start_offset) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    strings_offset[0] = start_offset;
    for (int64_t i = 1; i <= numel; ++i) {
      strings_offset[i] = strings_offset[i - 1] + src_str[i - 1].length() + 1;
    }
  }
  __syncthreads();
  CUDA_KERNEL_LOOP(i, numel) {
    memcpy(strings_data + strings_offset[i],
           src_str[i].data(),
           src_str[i].length() + 1);
  }
}

__global__ void SumStringsLen(const phi::dtype::pstring* src_ptr,
                              int64_t numel,
                              int* num) {
  extern __shared__ int counter[];
  int thread_counter = 0;
  CUDA_KERNEL_LOOP(i, numel) { thread_counter += src_ptr[i].length() + 1; }
  counter[threadIdx.x] = thread_counter;
  __syncthreads();
  if (threadIdx.x == 0) {
    int block_counter = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      block_counter += counter[i];
    }
    atomicAdd(num, block_counter);
  }
}

template <typename Context>
int GetAllStringsSize(const Context& dev_ctx,
                      const phi::dtype::pstring* src_ptr,
                      size_t numel) {
  auto nums_meta =
      phi::DenseTensorMeta(DataType::INT32, {1}, phi::DataLayout::NCHW);
  DenseTensor nums_tensor = phi::Empty(dev_ctx, std::move(nums_meta));

  int* nums_ptr = dev_ctx.template Alloc<int>(&nums_tensor);
  phi::backends::gpu::GpuMemsetAsync(
      nums_ptr, 0, sizeof(int), dev_ctx.stream());

  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  dim3 grid_size =
      dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  SumStringsLen<<<grid_size,
                  block_size,
                  PREDEFINED_BLOCK_SIZE * sizeof(int),
                  dev_ctx.stream()>>>(src_ptr, numel, nums_ptr);
  int num = -1;
#ifdef PADDLE_WITH_HIP
  phi::backends::gpu::GpuMemcpyAsync(
      &num, nums_ptr, sizeof(int), hipMemcpyDeviceToHost, dev_ctx.stream());
#else
  phi::backends::gpu::GpuMemcpyAsync(
      &num, nums_ptr, sizeof(int), cudaMemcpyDeviceToHost, dev_ctx.stream());
#endif
  return num;
}

__global__ void DeserializeCUDAKernel(const char* strings_data,
                                      const int* strings_offset,
                                      phi::dtype::pstring* dst_str,
                                      int numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    // -1 not include '\0'
    auto len = strings_offset[i + 1] - strings_offset[i] - 1;
    dst_str[i] = phi::dtype::pstring(strings_data + strings_offset[i], len);
  }
}
#endif

template <typename Context>
void SerializeOnCPU(const Context& dev_ctx,
                    const StringTensor& src,
                    DenseTensor* dst) {
  int64_t numel = src.numel();
  int64_t num = sizeof(int) * (numel + 1);
  auto* src_str = src.data();
  for (int64_t i = 0; i < numel; ++i) {
    num += src_str[i].length() + 1;
  }
  dst->Resize(common::make_ddim({num}));
  uint8_t* strings_data = dev_ctx.template HostAlloc<uint8_t>(dst);
  auto* strings_offset = reinterpret_cast<int*>(strings_data);
  int start_offset = sizeof(int) * (numel + 1);
  for (int64_t i = 0; i <= numel; ++i) {
    if (i == 0) {
      strings_offset[i] = start_offset;
    } else {
      strings_offset[i] = strings_offset[i - 1] + src_str[i - 1].length() + 1;
    }
  }
  for (int64_t i = 0; i < numel; ++i) {
    memcpy(strings_data + strings_offset[i],
           src_str[i].data(),
           src_str[i].length() + 1);
  }
}

template <typename Context>
void DeserializeOnCPU(const Context& dev_ctx,
                      const DenseTensor& src,
                      StringTensor* dst) {
  auto* strings_data = reinterpret_cast<const char*>(src.data<uint8_t>());
  auto* strings_offset = reinterpret_cast<const int*>(strings_data);
  int numel = strings_offset[0] / sizeof(int) - 1;
  dst->Resize(common::make_ddim({numel}));
  dtype::pstring* dst_str = dev_ctx.template HostAlloc<dtype::pstring>(dst);
  for (int i = 0; i < numel; ++i) {
    // -1 not include '\0'
    auto len = strings_offset[i + 1] - strings_offset[i] - 1;
    dst_str[i] = phi::dtype::pstring(strings_data + strings_offset[i], len);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void SerializeOnGPU(const phi::GPUContext& dev_ctx,
                    const StringTensor& src,
                    DenseTensor* dst) {
  int64_t numel = src.numel();
  auto* src_str = src.data();
  // 1.get the number of bytes of all strings in string tensor
  auto strings_size = GetAllStringsSize(dev_ctx, src_str, numel);
  strings_size += sizeof(int32_t) * (numel + 1);

  dst->Resize(common::make_ddim({strings_size}));
  uint8_t* strings_data = dev_ctx.template Alloc<uint8_t>(dst);
  auto* strings_offset = reinterpret_cast<int*>(strings_data);

  int32_t start_offset = sizeof(int32_t) * (numel + 1);
  // 2. serialize strings data to dense tensor
  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  dim3 grid_size =
      dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);

  SerializeStringsData<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      src_str, strings_data, strings_offset, numel, start_offset);
}

void DeserializeOnGPU(const phi::GPUContext& dev_ctx,
                      const DenseTensor& src,
                      StringTensor* dst) {
  auto* strings_data = reinterpret_cast<const char*>(src.data<uint8_t>());
  auto* strings_offset = reinterpret_cast<const int*>(strings_data);
  int numel = 0;
#ifdef PADDLE_WITH_HIP
  phi::backends::gpu::GpuMemcpySync(
      &numel, strings_data, sizeof(numel), hipMemcpyDeviceToHost);
#else
  phi::backends::gpu::GpuMemcpySync(
      &numel, strings_data, sizeof(numel), cudaMemcpyDeviceToHost);
#endif
  numel = numel / sizeof(int) - 1;
  dst->Resize(common::make_ddim({numel}));
  dtype::pstring* dst_str = dev_ctx.template Alloc<dtype::pstring>(dst);

  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  dim3 grid_size =
      dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  DeserializeCUDAKernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      strings_data, strings_offset, dst_str, numel);
}
#endif

}  // namespace strings
}  // namespace phi
