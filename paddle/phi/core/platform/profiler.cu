/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "paddle/phi/core/platform/profiler.h"

namespace paddle {
namespace platform {

__global__ void DummyKernel(int *a) { a[0] = 0; }

static void ForEachDevice(std::function<void(int)> func) {
  auto original_device = platform::GetCurrentDeviceId();
  int count = platform::GetGPUDeviceCount();
  for (int i = 0; i < count; i++) {
    platform::SetDeviceId(i);
    func(i);
  }
  platform::SetDeviceId(original_device);
}

void DummyKernelAndEvent() {
#ifdef PADDLE_WITH_HIP
  for (int i = 0; i < 5; i++) {
    ForEachDevice([](int d) {
      platform::SetDeviceId(d);
      hipStream_t stream;
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream));
      Mark("_cuda_startup_");
      int *ptr;
      PADDLE_ENFORCE_GPU_SUCCESS(hipMalloc(&ptr, sizeof(int)));
      hipLaunchKernelGGL(DummyKernel, dim3(1), dim3(1), 0, stream, ptr);
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
      PADDLE_ENFORCE_GPU_SUCCESS(hipFree(ptr));
    });
  }
#else
  for (int i = 0; i < 5; i++) {
    ForEachDevice([](int d) {
      platform::SetDeviceId(d);
      cudaStream_t stream;
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
      Mark("_cuda_startup_");
      int *ptr;
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, sizeof(int)));
      DummyKernel<<<1, 1, 0, stream>>>(ptr);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(ptr));
    });
  }
#endif
}

}  // namespace platform
}  // namespace paddle
