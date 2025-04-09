// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/gpu/cuda_gemm_kernel.h"
#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Type, int CtaM, int CtaN, int Threads>
__global__ void int8_gemm(int8_t const* act,
                          int8_t const* weight,
                          Type* output,
                          int m,
                          int n,
                          int k) {
#if defined(PADDLE_WITH_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
  return;
#else
  using VecType = int4;
  static constexpr int kStepK = 128 / (8 * sizeof(int8_t));
  static constexpr int CtaK = kStepK * Threads;
  int tile_id_m = blockIdx.x * CtaM;
  int tile_id_n = blockIdx.y * CtaN;
  int tid = threadIdx.x;
  int8_t tile_a[kStepK], tile_w[CtaN * kStepK];
  int acc[CtaM * CtaN];
#pragma unroll
  for (int i = 0; i < CtaM * CtaN; ++i) {
    acc[i] = 0;
  }
  act += tile_id_m * k;
  weight += tile_id_n * k;
  output += tile_id_m * n + tile_id_n;
  for (int idx_k = tid * kStepK; idx_k < k; idx_k += CtaK) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      reinterpret_cast<VecType*>(tile_w)[i] =
          reinterpret_cast<VecType const*>(weight + i * k + idx_k)[0];
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      reinterpret_cast<VecType*>(tile_a)[0] =
          reinterpret_cast<VecType const*>(act + i * k + idx_k)[0];
#pragma unroll
      for (int j = 0; j < CtaN; ++j) {
#pragma unroll
        for (int l = 0; l < kStepK; l += 4) {
          acc[i * CtaN + j] =
              __dp4a(reinterpret_cast<int*>(tile_a + l)[0],
                     reinterpret_cast<int*>(tile_w + j * kStepK + l)[0],
                     acc[i * CtaN + j]);
        }
      }
    }
  }

  static constexpr int kWarpSize = 32;
  static constexpr int kWarpNum = Threads / kWarpSize;
  __shared__ int shmem[CtaM * CtaN * kWarpNum];
  int warp_id = tid / kWarpSize, lane_id = tid % kWarpSize;
#pragma unroll
  for (int i = 0; i < CtaM; ++i) {
#pragma unroll
    for (int j = 0; j < CtaN; ++j) {
      int val = acc[i * CtaN + j];
      val += __shfl_xor_sync(~0, val, 16);
      val += __shfl_xor_sync(~0, val, 8);
      val += __shfl_xor_sync(~0, val, 4);
      val += __shfl_xor_sync(~0, val, 2);
      val += __shfl_xor_sync(~0, val, 1);
      if (lane_id == 0) {
        shmem[i * CtaN + j + warp_id * CtaM * CtaN] = val;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int ii = tid; ii < CtaM * CtaN; ii += Threads) {
    int mid = ii / CtaN, nid = ii % CtaN;
    int val = 0;
#pragma unroll
    for (int jj = 0; jj < kWarpNum; ++jj) {
      val += shmem[jj * CtaM * CtaN + ii];
    }
    output[mid * n + nid] = static_cast<Type>(static_cast<float>(val));
  }
#endif
}

template <typename InputType,
          typename OutputType,
          int32_t TILE_M,
          int32_t TILE_N,
          int32_t BLOCK_SIZE>
void cudaCoreGemmKernel(GemmParams const& params) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(params.m / TILE_M, params.n / TILE_N);
  int8_gemm<OutputType, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, params.stream>>>(
          reinterpret_cast<InputType const*>(params.act),
          reinterpret_cast<InputType const*>(params.weight),
          reinterpret_cast<OutputType*>(params.output),
          params.m,
          params.n,
          params.k);
}

template <typename InputType,
          typename OutputType,
          int TILE_M,
          int TILE_N,
          int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(GemmParams const& params) {
  constexpr int cudaCoreGemmTemplateMaxM = 16;
  if (params.m == TILE_M) {
    cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(
        params);
    return true;
  }
  if constexpr (TILE_M < cudaCoreGemmTemplateMaxM) {
    return cudaCoreGemmTemplateCaller<InputType,
                                      OutputType,
                                      TILE_M + 1,
                                      TILE_N,
                                      BLOCK_SIZE>(params);
  }
  return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(GemmParams const& params) {
  return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 256>(params);
}

template <typename T, typename Context>
void CudaGemm(const Context& ctx,
              const DenseTensor& input,
              const DenseTensor& w,
              DenseTensor* output) {
  ctx.template Alloc<int32_t>(output);
  auto input_dims = input.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The input tensor dimensions should be 2, but got %d.",
                        input_dims.size()));
  auto weight_dims = w.dims();
  PADDLE_ENFORCE_EQ(weight_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The weight tensor dimensions should be 2, but got %d.",
                        weight_dims.size()));

  auto out_dims = output->dims();

  const int m = input_dims[0];
  const int n = weight_dims[0];

  PADDLE_ENFORCE_EQ(
      input_dims[1],
      weight_dims[1],
      common::errors::InvalidArgument(
          "The input dims[1] %d should be equal to weight dims[1] %d.",
          input_dims[1],
          weight_dims[1]));
  const int k = weight_dims[1];

  GemmParams params = {
      reinterpret_cast<const void*>(input.data<T>()),
      reinterpret_cast<const void*>(w.data<T>()),
      reinterpret_cast<void*>(output->data<int32_t>()),
      m,
      n,
      k,
      ctx.stream(),
  };

  if (!cudaCoreGemmLauncher<int8_t, int32_t>(params)) {
    PADDLE_THROW(common::errors::Fatal("cuda gemm kernel run error"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(cuda_gemm, GPU, ALL_LAYOUT, phi::CudaGemm, int8_t) {}
