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
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include "paddle/fluid/operators/clip_by_norm_op.h"

namespace paddle {
namespace operators {

template <typename T>
__device__ __forceinline__ T sqrt(const T& v) {
  return sqrt(v);
}

template <>
__device__ __forceinline__ float sqrt(const float& v) {
  return sqrtf(v);
}

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          cub::BlockReduceAlgorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__ void clip_by_norm_kernel(const T* x, T* out, const size_t& numel,
                                    const T& max_norm) {
  typedef cub::BlockReduce<T, BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += blockDim.x * gridDim.x) {
    out[i] = x[i] * x[i];
  }
  T data[ITEMS_PER_THREAD];
  cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, out, data);
  T x_norm = BlockReduceT(temp_storage).Sum(data);

  __shared__ T norm;
  if (threadIdx.x == 0) {
    norm = sqrt(x_norm);
  }
  __syncthreads();
  x_norm = norm;
  T scaling(0);
  if (x_norm > max_norm) {
    scaling = max_norm / x_norm;
  } else {
    scaling = static_cast<T>(1);
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += blockDim.x * gridDim.x) {
    out[i] = x[i] * scaling;
  }
}

template <typename T>
void ClipByNormFunctor<CUDADeviceContext, T>::operator()(const T* x, T* out,
                                                         const size_t& numel,
                                                         const T& max_norm) {
  const int kBlockThreads = 1024;
  const int kItemsPerThread = 128;
  int block = 1024;
  int grid = (numel + block - 1) / block;
  clip_by_norm_kernel<T, kBlockThreads, kItemsPerThread><<<grid, block, 0, ctx_.stream()>>>(x, out, numel,
                                                            max_norm);
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    clip_by_norm,
    ops::ClipByNormKernel<paddle::platform::CUDADeviceContext, float>);
