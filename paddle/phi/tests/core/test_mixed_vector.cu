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

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/mixed_vector.h"

template <typename T>
using vec = phi::MixVector<T>;
using gpuStream_t = phi::gpuStream_t;

static __global__ void multiply_10(int* ptr) {
  for (int i = 0; i < 10; ++i) {
    ptr[i] *= 10;
  }
}

gpuStream_t GetCUDAStream(phi::GPUPlace place) {
  return reinterpret_cast<const phi::GPUContext*>(
             phi::DeviceContextPool::Instance().Get(place))
      ->stream();
}

TEST(mixed_vector, GPU_VECTOR) {
  std::vector<int> x;
  for (int i = 0; i < 10; ++i) {
    x.push_back(i);
  }
  vec<int> tmp(&x);
  ASSERT_EQ(tmp.size(), 10UL);
  phi::GPUPlace gpu(0);

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(multiply_10,
                     dim3(1),
                     dim3(1),
                     0,
                     GetCUDAStream(gpu),
                     tmp.MutableData(gpu));
#else
  multiply_10<<<1, 1, 0, GetCUDAStream(gpu)>>>(tmp.MutableData(gpu));
#endif

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp[i], i * 10);
  }
}

TEST(mixed_vector, MultiGPU) {
  if (phi::backends::gpu::GetGPUDeviceCount() < 2) {
    LOG(WARNING) << "Skip mixed_vector.MultiGPU since there are not multiple "
                    "GPUs in your machine.";
    return;
  }

  std::vector<int> x;
  for (int i = 0; i < 10; ++i) {
    x.push_back(i);
  }
  vec<int> tmp(&x);
  ASSERT_EQ(tmp.size(), 10UL);
  phi::GPUPlace gpu0(0);
  phi::backends::gpu::SetDeviceId(0);

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(multiply_10,
                     dim3(1),
                     dim3(1),
                     0,
                     GetCUDAStream(gpu0),
                     tmp.MutableData(gpu0));
#else
  multiply_10<<<1, 1, 0, GetCUDAStream(gpu0)>>>(tmp.MutableData(gpu0));
#endif
  phi::GPUPlace gpu1(1);
  auto* gpu1_ptr = tmp.MutableData(gpu1);
  phi::backends::gpu::SetDeviceId(1);

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(
      multiply_10, dim3(1), dim3(1), 0, GetCUDAStream(gpu1), gpu1_ptr);
#else
  multiply_10<<<1, 1, 0, GetCUDAStream(gpu1)>>>(gpu1_ptr);
#endif
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp[i], i * 100);
  }
}
