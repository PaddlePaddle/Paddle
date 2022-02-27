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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_helper.h"

namespace phi {

#ifdef __HIPCC__
static constexpr int kNumCUDAThreads = 256;
#else
static constexpr int kNumCUDAThreads = 512;
#endif
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, int BlockDim>
__global__ void Sum(const T *counts, int num, const T eps, T *sum) {
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T in = 0;
  for (int i = threadIdx.x; i < num; i += BlockDim) {
    in += counts[i];
  }
  __syncthreads();
  auto out =
      BlockReduce(temp_storage).Reduce(static_cast<double>(in), cub::Sum());
  __syncthreads();
  if (threadIdx.x == 0) {
    T a = out > eps ? out : eps;
    sum[0] = a;
  }
}

template <typename T>
__global__ void Div(T *loss, const int num, const T *norm) {
  CUDA_KERNEL_LOOP(i, num) { loss[i] /= norm[0]; }
}

}  // namespace phi
