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

#include <cuda.h>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace platform {

__global__ void DummyKernel(int *a) { a[0] = 0; }

static void ForEachDevice(std::function<void(int)> func) {
  auto original_device = platform::GetCurrentDeviceId();
  int count = platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    platform::SetDeviceId(i);
    func(i);
  }
  platform::SetDeviceId(original_device);
}

void DummyKernelAndEvent() {
  for (int i = 0; i < 5; i++) {
    ForEachDevice([](int d) {
      platform::SetDeviceId(d);
      cudaStream_t stream;
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream));
      Mark("_cuda_startup_");
      int *ptr;
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&ptr, sizeof(int)));
      DummyKernel<<<1, 1, 0, stream>>>(ptr);
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(ptr));
    });
  }
}

}  // namespace platform
}  // namespace paddle
