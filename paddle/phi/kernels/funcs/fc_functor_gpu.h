/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <cstdint>

#include "paddle/phi/backends/gpu/gpu_decls.h"

namespace phi {

class GPUContext;

namespace funcs {

template <typename T>
void AddReluKernel(
    gpuStream_t stream, int M, int N, T* Y, const T* B, bool relu);

template <typename T>
void LaunchFcQuantKernel(const T* input,
                         int8_t* output,
                         float scale,
                         int m,
                         int n,
                         int round_type,
                         float max_bound,
                         float min_bound,
                         gpuStream_t stream);

template <typename T>
void LaunchFcDequantKernel(const GPUContext& dev_ctx,
                           const int32_t* input,
                           T* output,
                           int m,
                           int n,
                           float quant_in_scale,
                           const float* quant_weight_scale,
                           float quant_max_bound);

}  // namespace funcs
}  // namespace phi

#endif
