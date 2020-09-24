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

#include <cuda_profiler_api.h>

#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

void CudaProfilerInit(std::string output_file, std::string output_mode,
                      std::string config_file) {
  PADDLE_ENFORCE(output_mode == "kvp" || output_mode == "csv",
                 platform::errors::InvalidArgument(
                     "Unsupported cuda profiler output mode, expect `kvp` or "
                     "`csv`, but received `%s`.",
                     output_mode));
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaProfilerInitialize(config_file.c_str(), output_file.c_str(), mode));
}

void CudaProfilerStart() { PADDLE_ENFORCE_CUDA_SUCCESS(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE_CUDA_SUCCESS(cudaProfilerStop()); }

}  // namespace platform
}  // namespace paddle
