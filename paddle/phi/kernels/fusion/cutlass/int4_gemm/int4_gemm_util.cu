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

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"
#include "paddle/phi/kernels/transpose_kernel.h"

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

template <typename Destination, typename Source>
__global__ void DynamicConvert(Source const *s, Destination *t, int N) {
  cutlass::NumericConverter<Destination, Source> converter;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    t[i] = converter(s[i]);
  }
  return;
}

template __global__ void DynamicConvert<int8_t, int32_t>(int32_t const *s,
                                                         int8_t *t,
                                                         int N);

template <>
__global__ void DynamicConvert<int32_t, int32_t>(int32_t const *s,
                                                 int32_t *t,
                                                 int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    t[i] = s[i];
  }
  return;
}

template <>
__global__ void DynamicConvert<cutlass::int4b_t, int32_t>(int32_t const *s,
                                                          cutlass::int4b_t *t,
                                                          int N) {
  cutlass::NumericArrayConverter<cutlass::int4b_t, int, 8> converter;

  cutlass::Array<cutlass::int4b_t, 8> *result_ptr =
      reinterpret_cast<cutlass::Array<cutlass::int4b_t, 8> *>(t);
  cutlass::Array<int, 8> const *source_ptr =
      reinterpret_cast<cutlass::Array<int, 8> const *>(s);

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  CUTLASS_PRAGMA_UNROLL
  for (int i = idx; i < N / 8; i += gridDim.x * blockDim.x) {
    result_ptr[i] = converter(source_ptr[i]);
  }
  return;
}

template <typename T>
__global__ void ExpendKernel(
    const T *vector, T *matrix, const int n, const int m, const int col_major) {
  if (col_major) {
    int idx = threadIdx.x + blockIdx.x * m;
    T myval = vector[blockIdx.x % n];
    while (idx < ((blockIdx.x + 1) * m)) {
      matrix[idx] = myval;
      idx += blockDim.x;
    }
  } else {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    T myval = vector[idx % n];
    while (idx < m * n) {
      matrix[idx] = myval;
      idx += gridDim.x * blockDim.x;
    }
  }
}

template __global__ void ExpendKernel<int32_t>(const int32_t *vector,
                                               int32_t *matrix,
                                               const int n,
                                               const int m,
                                               const int col_major);

template <typename T, typename Context>
void ConvertDataToInt4(const Context &ctx,
                       const DenseTensor &source,
                       cutlass::int4b_t *output,
                       const size_t source_size,
                       const bool transpose) {
  auto stream = ctx.stream();
  DenseTensor source_prepare;
  if (transpose) {
    source_prepare = TransposeLast2Dim<T, Context>(ctx, source);
  } else {
    source_prepare = source;
  }

  constexpr int block_ = 256;
  dim3 grid((source_size + block_ - 1) / block_);
  dim3 block(block_);
  DynamicConvert<cutlass::int4b_t, T>
      <<<grid, block>>>(reinterpret_cast<const T *>(source_prepare.data()),
                        reinterpret_cast<cutlass::int4b_t *>(output),
                        source_size);
  return;
}

template void ConvertDataToInt4<int32_t, phi::GPUContext>(
    const phi::GPUContext &ctx,
    const DenseTensor &source,
    cutlass::int4b_t *output,
    const size_t source_size,
    const bool transpose);

}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
