// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

namespace phi {

class GpuTimer {
 public:
  GpuTimer() {
#ifdef PADDLE_WITH_HIP
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
#else
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
#endif
    PADDLE_ENFORCE_NOT_NULL(
        start_, phi::errors::PreconditionNotMet("Start Event is not ready."));
    PADDLE_ENFORCE_NOT_NULL(
        stop_, phi::errors::PreconditionNotMet("Stop Event is not ready."));
  }

  ~GpuTimer() {
#ifdef PADDLE_WITH_HIP
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
#else
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
#endif
  }

  void Start(gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    hipEventRecord(start_, stream);
#else
    cudaEventRecord(start_, stream);
#endif
  }

  void Stop(gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    hipEventRecord(stop_, stream);
#else
    cudaEventRecord(stop_, stream);
#endif
  }

  float ElapsedTime() {
    float milliseconds = 0;
#ifdef PADDLE_WITH_HIP
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&milliseconds, start_, stop_);
#else
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds, start_, stop_);
#endif
    return milliseconds;
  }

 private:
  gpuEvent_t start_;
  gpuEvent_t stop_;
};

}  // namespace phi
