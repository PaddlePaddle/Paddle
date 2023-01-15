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

#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(GemmAllParams)>>
        &gemm_funcs,
    GemmAllParams params) {
  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;
  float min_time = 100000.f;
  int min_time_index = -1;
  for (int i = 0; i < gemm_funcs.size(); i++) {
    cutlass::Status status;
    auto func = gemm_funcs[i];
    // When func has large diff, we will make it nullptr.
    if (!func) continue;

    for (int ii = 0; ii < WARMUP; ii++) {
      status = func(params);
    }

    cudaEvent_t beg, end;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&beg));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(beg));
    for (int ii = 0; ii < REPEAT; ii++) {
      status = func(params);
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(end));
    float elapsed_time;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(&elapsed_time, beg, end));
    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
      min_time = elapsed_time;
      min_time_index = i;
    }
  }

  if (min_time_index < 0) {
    PADDLE_THROW(
        phi::errors::NotFound("Can't find any cutlass config for this op."));
  }
  return min_time_index;
}

}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
