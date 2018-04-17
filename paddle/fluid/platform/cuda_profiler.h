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
#pragma once

#ifdef PADDLE_WITH_HIP
#include "hip/hip_runtime_api.h"
#else
#include <cuda_profiler_api.h>
#endif

#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_HIP
void CudaProfilerInit(std::string output_file, std::string output_mode,
                      std::string config_file) {
}

void CudaProfilerStart() { PADDLE_ENFORCE(hipProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE(hipProfilerStop()); }

#else

void CudaProfilerInit(std::string output_file, std::string output_mode,
                      std::string config_file) {
  PADDLE_ENFORCE(output_mode == "kvp" || output_mode == "csv");
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE(
      cudaProfilerInitialize(config_file.c_str(), output_file.c_str(), mode));
}

void CudaProfilerStart() { PADDLE_ENFORCE(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE(cudaProfilerStop()); }

#endif
}  // namespace platform
}  // namespace paddle
