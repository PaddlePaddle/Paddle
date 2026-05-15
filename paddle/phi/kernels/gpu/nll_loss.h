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

#include <algorithm>
#include <functional>
#include <string>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math.h"

namespace phi {
static constexpr int kNumCUDAThreads = 512;
// Thread count for the 4D hard-label reduce path, matching PyTorch's
// CUDA_NUM_THREADS = 1024 used in NLLLoss2d.cu.
static constexpr int kNumCUDAThreads4D = 1024;
static constexpr int kNumMaximumNumBlocks = 4096;
static const int NTHREADS = 32;
static inline int64_t NumBlocks(const int64_t N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  static_cast<int64_t>(kNumMaximumNumBlocks));
}

template <typename T>
__global__ void GPUNLLLossForward1D_no_reduce(T* out_data,
                                              const T* x_data,
                                              const int64_t* label_data,
                                              const T* weight_data,
                                              const int64_t batch_size,
                                              const int64_t n_classes,
                                              const int64_t ignore_index) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    const int64_t cur_label = label_data[i];
    if (cur_label == ignore_index) {
      out_data[i] = 0;
      continue;
    }
    PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                   "label should not be out of bounds.");
    const T cur_weight = weight_data ? weight_data[cur_label] : (T)1;
    out_data[i] = -x_data[i * n_classes + cur_label] * cur_weight;
  }
}

template <typename T, typename AccT>
__global__ void GPUNLLLossForward1D_with_reduce(T* out_data,
                                                T* total_weight_data,
                                                const T* x_data,
                                                const int64_t* label_data,
                                                const T* weight_data,
                                                const int64_t batch_size,
                                                const int64_t n_classes,
                                                const int64_t size_average,
                                                const int64_t ignore_index) {
  __shared__ T sharedInputs[NTHREADS], sharedWeights[NTHREADS];
  sharedInputs[threadIdx.x] = 0;
  sharedWeights[threadIdx.x] = 0;
  int64_t i;
  for (i = threadIdx.x; i < batch_size; i += NTHREADS) {
    const auto cur_label = label_data[i];
    if (cur_label != ignore_index) {
      PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                     "label should not be out of bounds.");
      const auto cur_weight = weight_data ? weight_data[cur_label] : (T)1;
      sharedInputs[threadIdx.x] -=
          x_data[i * n_classes + cur_label] * cur_weight;
      sharedWeights[threadIdx.x] += cur_weight;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *out_data = *total_weight_data = 0;
    AccT output_val = 0;
    AccT total_weight_val = 0;
    for (i = 0; i < NTHREADS; ++i) {
      output_val += sharedInputs[i];
      total_weight_val += sharedWeights[i];
    }
    *total_weight_data = total_weight_val;
    *out_data = output_val;

    if (size_average && *total_weight_data != 0) {
      *out_data = output_val / total_weight_val;
    }
  }
}

// Compute thread count: clamp(round_pow2(N/16), 32, 1024).
inline int nll_loss_threads(int64_t batch_size) {
  int x = static_cast<int>((batch_size + 15) / 16);
  // Round to nearest power of 2
  x = std::max(x, 1);
  int log2_val = 0;
  int tmp = x;
  while (tmp > 1) {
    tmp >>= 1;
    log2_val++;
  }
  // Round: check if x is closer to (1 << log2_val) or (1 << (log2_val + 1))
  int lower = 1 << log2_val;
  int upper = 1 << (log2_val + 1);
  int rounded = (x - lower <= upper - x) ? lower : upper;
  // Clamp to [32, 1024]
  return std::min(std::max(rounded, 32), 1024);
}

// Accuracy-compatible NLL loss with tree reduction in shared memory.
template <typename T, typename AccT>
__global__ void GPUNLLLossForward1D_with_reduce_compatible(
    T* out_data,
    T* total_weight_data,
    const T* x_data,
    const int64_t* label_data,
    const T* weight_data,
    const int64_t batch_size,
    const int64_t n_classes,
    const int64_t size_average,
    const int64_t ignore_index) {
  // Dynamic shared memory: first nthreads AccT for loss, next nthreads AccT
  // for weight
  extern __shared__ char smem[];
  AccT* sh_loss = reinterpret_cast<AccT*>(smem);
  AccT* sh_weight = sh_loss + blockDim.x;

  // Thread-strided sequential accumulation
  AccT thread_loss = AccT(0);
  AccT thread_weight = AccT(0);
  for (int64_t i = threadIdx.x; i < batch_size; i += blockDim.x) {
    const int64_t cur_label = label_data[i];
    if (cur_label != ignore_index) {
      PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                     "label should not be out of bounds.");
      const AccT cur_weight =
          weight_data ? static_cast<AccT>(weight_data[cur_label]) : AccT(1);
      thread_loss -=
          static_cast<AccT>(x_data[i * n_classes + cur_label]) * cur_weight;
      thread_weight += cur_weight;
    }
  }

  sh_loss[threadIdx.x] = thread_loss;
  sh_weight[threadIdx.x] = thread_weight;
  __syncthreads();

  // Power-of-2 tree reduction
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_loss[threadIdx.x] += sh_loss[threadIdx.x + stride];
      sh_weight[threadIdx.x] += sh_weight[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (size_average && sh_weight[0] != AccT(0)) {
      *out_data = static_cast<T>(sh_loss[0] / sh_weight[0]);
    } else {
      *out_data = static_cast<T>(sh_loss[0]);
    }
    *total_weight_data = static_cast<T>(sh_weight[0]);
  }
}

// Reduce N values concurrently, i.e. suppose N = 2, and there are 4 threads:
// (1, 2), (3, 4), (5, 6), (7, 8), then the return in threadVals for thread 0
// is (1 + 3 + 5 + 7, 2 + 4 + 6 + 8) = (16, 20)
//
// If smem is not used again, there is no need to __syncthreads before this
// call. However, if smem will be used, e.g., this function is called in a loop,
// then __syncthreads is needed either before or afterwards to prevent non-0
// threads overriding smem in the next loop before num-0 thread reads from it.
template <typename T, typename ReduceOp, int64_t N>
__device__ void reduceNValuesInBlock(T* smem,
                                     T threadVals[N],
                                     const unsigned int numVals,
                                     ReduceOp reduceOp,
                                     T init) {
  if (numVals == 0) {
#pragma unroll
    for (int64_t i = 0; i < N; ++i) {
      threadVals[i] = init;
    }
    return;
  }

  // We store each of the N values contiguously, so if N = 2, all values for
  // the first threadVal for each thread in the block are stored followed by
  // all of the values for the second threadVal for each thread in the block
  if (threadIdx.x < numVals) {
#pragma unroll
    for (int64_t i = 0; i < N; ++i) {
      smem[i * numVals + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  // Number of lanes in the final reduction --> this is used to determine
  // where to put the outputs of each of the n things we are reducing. If
  // nLP = 32, then we have the 32 outputs for the first threadVal,
  // followed by the 32 outputs for the second threadVal, etc.
  const unsigned int numLanesParticipating = min(numVals, warpSize);

  if (numVals > warpSize && ((threadIdx.x / warpSize) == 0)) {
#pragma unroll
    for (int64_t i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? threadVals[i] : init;
    }

    for (int64_t i = warpSize + static_cast<int64_t>(threadIdx.x); i < numVals;
         i += warpSize) {
#pragma unroll
      for (int64_t j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[j * numVals + i]);
      }
    }

#pragma unroll
    for (int64_t i = 0; i < N; ++i) {
      smem[i * numLanesParticipating + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (numLanesParticipating == 32) {
#pragma unroll
      for (int64_t i = 0; i < N; ++i) {
#pragma unroll
        for (int64_t j = 1; j < 32; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * 32 + j]);
        }
      }
    } else {
#pragma unroll
      for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 1; j < numLanesParticipating; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * numVals + j]);
        }
      }
    }
  }
}

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
//
// If smem is not used again, there is no need to __syncthreads before this
// call. However, if smem will be used, e.g., this function is called in a loop,
// then __syncthreads is needed either before or afterwards to prevent non-0
// threads overriding smem in the next loop before num-0 thread reads from it.
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         const unsigned int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  reduceNValuesInBlock<T, ReduceOp, 1>(
      smem, &threadVal, numVals, reduceOp, init);
  return threadVal;
}

// ---------------------------------------------------------------------------
// Warp/block reduction helpers matching PyTorch's block_reduce.cuh pattern.
// Used exclusively by GPUNLLLossForward2D_with_reduce (4D hard-label reduce).
// ---------------------------------------------------------------------------

// Per-warp butterfly sum using __shfl_down_sync (matches PyTorch
// WarpReduceSum). All lanes receive the warp-sum, but only lane-0 is
// authoritative after the subsequent two-level block reduce.
template <typename T>
__device__ __forceinline__ T NLLLossWarpReduceSum(T val) {
#pragma unroll
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
#ifdef PADDLE_WITH_CUDA
    val += __shfl_down_sync(0xffffffff, val, offset);
#else
    val += __shfl_down(val, offset);
#endif
  }
  return val;
}

// Two-level block reduction matching PyTorch's BlockReduceSum in
// block_reduce.cuh.
//   1. Each warp does NLLLossWarpReduceSum -> warp leader writes to
//   shared[wid].
//   2. Warp 0 loads shared and does another NLLLossWarpReduceSum.
// Only thread 0 has the correct result on return.
// shared[] must hold at least (blockDim.x / warpSize) elements.
// A __syncthreads() at the start prevents races when called back-to-back.
template <typename T>
__device__ __forceinline__ T NLLLossBlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;
  val = NLLLossWarpReduceSum(val);
  __syncthreads();  // guard against back-to-back calls sharing the same smem
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  const int n_warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < n_warps) ? shared[lid] : T(0);
  if (wid == 0) {
    val = NLLLossWarpReduceSum(val);
  }
  return val;
}
// ---------------------------------------------------------------------------

template <typename T>
__global__ void GPUNLLLossForward2D_no_reduce(T* out_data,
                                              const T* x_data,
                                              const int64_t* label_data,
                                              const T* weight_data,
                                              const int64_t batch_size,
                                              const int64_t n_classes,
                                              const int64_t in_dim2,
                                              const int64_t in_dim3,
                                              const int64_t ignore_index) {
  const int64_t map_size = in_dim2 * in_dim3;
  const int64_t sample_size = n_classes * map_size;
  const int64_t out_numel = batch_size * map_size;
  CUDA_KERNEL_LOOP(i, out_numel) {
    const int64_t b = i % batch_size;
    const int64_t h = (i / batch_size) % in_dim2;
    const int64_t w = (i / (batch_size * in_dim2)) % in_dim3;

    const int64_t index = b * map_size + h * in_dim3 + w;
    const int64_t cur_label = label_data[index];
    if (cur_label == ignore_index) {
      out_data[index] = 0;
      continue;
    }
    PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                   "label should not be out of bounds.");
    const T cur_weight = weight_data ? weight_data[cur_label] : (T)1;
    out_data[index] =
        -x_data[b * sample_size + cur_label * map_size + h * in_dim3 + w] *
        cur_weight;
  }
}

template <typename T, typename AccT>
__global__ void GPUNLLLossForward2D_with_reduce(T* out_data,
                                                T* total_weight_data,
                                                const T* x_data,
                                                const int64_t* label_data,
                                                const T* weight_data,
                                                const int64_t batch_size,
                                                const int64_t n_classes,
                                                const int64_t map_nelem,
                                                const int64_t blocks_per_sample,
                                                const int64_t ignore_index) {
  // Two separate shared-memory arrays for the two-level block reduction.
  // 32 entries is sufficient for up to kNumCUDAThreads4D=1024 threads
  // (1024 / 32 = 32 warps on CUDA; 1024 / 64 = 16 wavefronts on HIP).
  // Layout matches PyTorch's nll_loss2d_forward_kernel in NLLLoss2d.cu.
  __shared__ AccT acc_weight_smem[32];
  __shared__ AccT input_sum_smem[32];

  AccT input_sum = AccT(0);
  AccT acc_weight = AccT(0);

  int64_t sample = static_cast<int64_t>(blockIdx.x) / blocks_per_sample;
  int64_t toffset = sample * map_nelem;
  int64_t ioffset = sample * map_nelem * n_classes;
  int64_t step = static_cast<int64_t>(blockDim.x) * blocks_per_sample;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) % blocks_per_sample) *
                       static_cast<int64_t>(blockDim.x) +
                   static_cast<int64_t>(threadIdx.x);
       i < map_nelem;
       i += step) {
    const int64_t cur_label = label_data[toffset + i];
    if (cur_label != ignore_index) {
      PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                     "label should not be out of bounds.");
      const AccT cur_weight =
          weight_data ? static_cast<AccT>(weight_data[cur_label]) : AccT(1);
      input_sum -=
          static_cast<AccT>(x_data[ioffset + i + map_nelem * cur_label]) *
          cur_weight;
      acc_weight += cur_weight;
    }
  }

  // Two-level warp-shuffle block reduction matching PyTorch's BlockReduceSum.
  // acc_weight_ is reduced first so its __syncthreads guards input_sum's smem.
  AccT acc_weight_ = NLLLossBlockReduceSum(acc_weight, acc_weight_smem);
  AccT input_sum_ = NLLLossBlockReduceSum(input_sum, input_sum_smem);

  if (threadIdx.x == 0) {
    CudaAtomicAdd(total_weight_data, static_cast<T>(acc_weight_));
    CudaAtomicAdd(out_data, static_cast<T>(input_sum_));
  }
}

template <typename T, typename AccT>
__global__ void GPUNLLLossForward2D_with_reduce_compatible(
    T* out_data,
    T* total_weight_data,
    const T* x_data,
    const int64_t* label_data,
    const T* weight_data,
    const int64_t batch_size,
    const int64_t n_classes,
    const int64_t map_nelem,
    const int64_t size_average,
    const int64_t ignore_index) {
  extern __shared__ char smem[];
  AccT* sh_loss = reinterpret_cast<AccT*>(smem);
  AccT* sh_weight = sh_loss + blockDim.x;

  const int64_t total_nelem = batch_size * map_nelem;
  const int64_t sample_size = map_nelem * n_classes;
  AccT thread_loss = AccT(0);
  AccT thread_weight = AccT(0);

  for (int64_t index = threadIdx.x; index < total_nelem; index += blockDim.x) {
    const int64_t cur_label = label_data[index];
    if (cur_label != ignore_index) {
      PADDLE_ENFORCE(cur_label >= 0 && cur_label < n_classes,
                     "label should not be out of bounds.");
      const int64_t sample = index / map_nelem;
      const int64_t map_offset = index % map_nelem;
      const AccT cur_weight =
          weight_data ? static_cast<AccT>(weight_data[cur_label]) : AccT(1);
      thread_loss -=
          static_cast<AccT>(x_data[sample * sample_size + map_offset +
                                   map_nelem * cur_label]) *
          cur_weight;
      thread_weight += cur_weight;
    }
  }

  sh_loss[threadIdx.x] = thread_loss;
  sh_weight[threadIdx.x] = thread_weight;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_loss[threadIdx.x] += sh_loss[threadIdx.x + stride];
      sh_weight[threadIdx.x] += sh_weight[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (size_average && sh_weight[0] != AccT(0)) {
      *out_data = static_cast<T>(sh_loss[0] / sh_weight[0]);
    } else {
      *out_data = static_cast<T>(sh_loss[0]);
    }
    *total_weight_data = static_cast<T>(sh_weight[0]);
  }
}

template <typename T>
__global__ void GPUNLLLossForward2D_size_average(T* out_data,
                                                 T* total_weight_data) {
  if (*total_weight_data != 0) {
    *out_data /= *total_weight_data;
  }
}
template <typename T>
__global__ void GPUNLLLossBackward1D_no_reduce(T* dx_data,
                                               const int64_t* label_data,
                                               const T* weight_data,
                                               const T* dout_data,
                                               const int64_t batch_size,
                                               const int64_t n_classes,
                                               const int64_t ignore_index) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    const int64_t cur_label = label_data[i];
    if (cur_label == ignore_index) {
      continue;
    }
    const T cur_weight = weight_data ? weight_data[cur_label] : (T)1;
    dx_data[i * n_classes + cur_label] = -dout_data[i] * cur_weight;
  }
}

template <typename T>
__global__ void GPUNLLLossBackward1D_with_reduce(T* dx_data,
                                                 const T* total_weight_data,
                                                 const int64_t* label_data,
                                                 const T* weight_data,
                                                 const T* dout_data,
                                                 const int64_t batch_size,
                                                 const int64_t n_classes,
                                                 const int64_t size_average,
                                                 const int64_t ignore_index) {
  if (*total_weight_data <= 0) {
    return;
  }
  int64_t i;
  const T norm = size_average ? (T)(1 / *total_weight_data) : (T)1;
  for (i = threadIdx.x; i < batch_size; i += NTHREADS) {
    const int64_t cur_label = label_data[i];
    if (cur_label != ignore_index) {
      const T cur_weight = weight_data ? weight_data[cur_label] : (T)1;
      dx_data[i * n_classes + cur_label] = -cur_weight * dout_data[0] * norm;
    }
  }
}

template <typename T>
__global__ void GPUNLLLossBackward2D_no_reduce(T* dx_data,
                                               const int64_t* label_data,
                                               const T* weight_data,
                                               const T* dout_data,
                                               const int64_t batch_size,
                                               const int64_t n_classes,
                                               const int64_t in_dim2,
                                               const int64_t in_dim3,
                                               const int64_t ignore_index) {
  const int64_t map_size = in_dim2 * in_dim3;
  const int64_t sample_size = n_classes * map_size;
  const int64_t out_numel = batch_size * map_size;
  CUDA_KERNEL_LOOP(i, out_numel) {
    const int64_t b = i % batch_size;
    const int64_t h = (i / batch_size) % in_dim2;
    const int64_t w = (i / (batch_size * in_dim2)) % in_dim3;
    const int64_t index = b * map_size + h * in_dim3 + w;
    const int64_t cur_label = label_data[index];
    if (cur_label == ignore_index) {
      continue;
    }
    const T cur_weight = weight_data ? weight_data[cur_label] : (T)1;
    dx_data[b * sample_size + cur_label * map_size + h * in_dim3 + w] =
        -dout_data[index] * cur_weight;
  }
}

template <typename T>
__global__ void GPUNLLLossBackward2D_with_reduce(
    T* dx_data,
    const T* total_weight_data,
    const int64_t* label_data,
    const T* weight_data,
    const T* dout_data,
    const int64_t batch_size,
    const int64_t n_classes,
    const int64_t map_nelem,
    const int64_t blocks_per_sample,
    const int64_t size_average,
    const int64_t ignore_index) {
  if (*total_weight_data <= 0) {
    return;
  }
  int64_t i;
  const T norm = size_average ? (T)(1 / *total_weight_data) : (T)1;
  int64_t sample = static_cast<int64_t>(blockIdx.x) / blocks_per_sample;
  int64_t step = static_cast<int64_t>(blockDim.x) * blocks_per_sample;
  int64_t toffset = sample * map_nelem;
  int64_t ioffset = sample * map_nelem * n_classes;
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    const int64_t cur_label = label_data[toffset + i];
    if (cur_label != ignore_index) {
      dx_data[ioffset + i + map_nelem * cur_label] =
          -(weight_data ? weight_data[cur_label] : (T)1) * norm * dout_data[0];
    }
  }
}

}  // namespace phi
