// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
typedef struct {
  const cutlass::int4b_t *input;
  const cutlass::int4b_t *weight;
  const int32_t *bias;
  int32_t *output;
  int batch;
  int m;
  int n;
  int k;
  const phi::GPUContext *ctx;
} GemmAllParams;

// Below functions are provided by cutlass, they are called bt phi
void Int4Gemm(GemmAllParams params, int sm);
void Int4GemmBias(GemmAllParams params, int sm);
void Int4GemmRelu(GemmAllParams params, int sm);
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
