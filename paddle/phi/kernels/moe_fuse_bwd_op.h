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

#pragma once
#ifdef PADDLE_WITH_CUDA
#include "paddle/common/exception.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/moe_kernel_impl.h"

namespace phi {

template <typename T, int64_t vec_size>
__global__ void gather_with_mask_permute_kernel(
    const T* dy,                   // [s*k, d]
    const int* scatter_index,      // [s, k]
    const float* combine_weights,  // [s, k]
    T* dx,                         // [s, d]
    int64_t num_rows,              // s
    int64_t k,                     // k
    int64_t dim,                   // d
    int64_t N,
    int64_t num_active,  // skip > num_active pos is num_active specified
    int64_t s_shared_num,
    int64_t capacity,
    int64_t world_size,
    int64_t num_local_experts) {
  extern __shared__ char shared[];
  int* scatter_index_shared = reinterpret_cast<int*>(shared);
  float* combine_weights_shared =
      reinterpret_cast<float*>(shared + s_shared_num * k * sizeof(int));
  int64_t shared_idx_begin = blockIdx.x * blockDim.x * vec_size;

  for (int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
       idx < N;
       idx += blockDim.x * gridDim.x * vec_size) {
    int64_t si = idx / dim;
    int64_t di_begin = idx % dim;
    int64_t si_shared_begin = shared_idx_begin / dim;
    int64_t shared_stride =
        min(static_cast<int64_t>(blockDim.x), N - shared_idx_begin);

    for (int64_t i = threadIdx.x; i < k * s_shared_num; i += shared_stride) {
      if (si_shared_begin * k + i >= num_rows * k) {
        break;
      }
      scatter_index_shared[i] = scatter_index[si_shared_begin * k + i];
      combine_weights_shared[i] = combine_weights[si_shared_begin * k + i];
    }
    __syncthreads();

    phi::AlignedVector<T, vec_size> in_vec;
    phi::AlignedVector<T, vec_size> out_vec;
    for (int ii = 0; ii < vec_size; ++ii) {
      out_vec[ii] = static_cast<T>(0);
    }

    for (int64_t i = 0; i < k; ++i) {
      int64_t scatter_offset = (si - si_shared_begin) * k + i;
      int id = scatter_index_shared[scatter_offset];
      if (num_active >= 0 && id >= num_active) {
        continue;
      }
      if (combine_weights_shared[scatter_offset] > 0.f) {
        int64_t remaining_after_irank = id % (num_local_experts * capacity);

        int64_t irank = id / (num_local_experts * capacity);
        int64_t local_iexpert = remaining_after_irank / capacity;
        int64_t row_in_expert = remaining_after_irank % capacity;
        int64_t permuted_id = local_iexpert * (world_size * capacity) +
                              irank * capacity + row_in_expert;
        int64_t in_offset = permuted_id * dim + di_begin;
        phi::Load<T, vec_size>(dy + in_offset, &in_vec);
        for (int64_t j = 0; j < vec_size; ++j) {
          out_vec[j] += in_vec[j];
        }
      }
    }
    phi::Store<T, vec_size>(out_vec, dx + idx);
    shared_idx_begin += blockDim.x * gridDim.x * vec_size;
  }
}

template <typename T, int64_t vec_size>
__global__ void gather_with_mask_kernel(
    const T* dy,                   // [s*k, d]
    const int* scatter_index,      // [s, k]
    const float* combine_weights,  // [s, k]
    T* dx,                         // [s, d]
    int64_t num_rows,              // s
    int64_t k,                     // k
    int64_t dim,                   // d
    int64_t N,
    int64_t num_active,  // skip > num_active pos is num_active specified
    int64_t s_shared_num) {
  extern __shared__ char shared[];
  int* scatter_index_shared = reinterpret_cast<int*>(shared);
  float* combine_weights_shared =
      reinterpret_cast<float*>(shared + s_shared_num * k * sizeof(int));
  int64_t shared_idx_begin = blockIdx.x * blockDim.x * vec_size;

  for (int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
       idx < N;
       idx += blockDim.x * gridDim.x * vec_size) {
    int64_t si = idx / dim;
    int64_t di_begin = idx % dim;
    int64_t si_shared_begin = shared_idx_begin / dim;
    int64_t shared_stride =
        min(static_cast<int64_t>(blockDim.x), N - shared_idx_begin);

    for (int64_t i = threadIdx.x; i < k * s_shared_num; i += shared_stride) {
      if (si_shared_begin * k + i >= num_rows * k) {
        break;
      }
      scatter_index_shared[i] = scatter_index[si_shared_begin * k + i];
      combine_weights_shared[i] = combine_weights[si_shared_begin * k + i];
    }
    __syncthreads();

    phi::AlignedVector<T, vec_size> in_vec;
    phi::AlignedVector<T, vec_size> out_vec;
    for (int ii = 0; ii < vec_size; ++ii) {
      out_vec[ii] = static_cast<T>(0);
    }

    for (int64_t i = 0; i < k; ++i) {
      int64_t scatter_offset = (si - si_shared_begin) * k + i;
      int id = scatter_index_shared[scatter_offset];
      if (num_active >= 0 && id >= num_active) {
        continue;
      }
      if (combine_weights_shared[scatter_offset] > 0.f) {
        int64_t in_offset = id * dim + di_begin;
        phi::Load<T, vec_size>(dy + in_offset, &in_vec);
        for (int64_t j = 0; j < vec_size; ++j) {
          out_vec[j] += in_vec[j];
        }
      }
    }
    phi::Store<T, vec_size>(out_vec, dx + idx);
    shared_idx_begin += blockDim.x * gridDim.x * vec_size;
  }
}

template <typename T = int64_t>
inline T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

inline int64_t max_shared_s_num(int64_t num_rows,
                                int64_t dim,
                                int64_t threads,
                                int64_t vec_size) {
  if ((threads * vec_size) % dim == 0) {
    return min(num_rows, threads * vec_size / dim);
  } else {
    int64_t max_res = DivUp<int64_t>(threads * 4, dim);
    for (int64_t idx = 0; idx < num_rows * dim; idx += vec_size * threads) {
      int64_t si_start = idx / dim;
      int64_t si_end = min(num_rows * dim, idx + vec_size * threads - 1) / dim;
      max_res = max(max_res, (si_end - si_start + 1));
    }
    return min(num_rows, max_res);
  }
}

template <typename T>
void gather_with_mask_launcher(const T* dy,                   // [s*k, d]
                               const int* scatter_index,      // [s, k]
                               const float* combine_weights,  // [s, k]
                               T* dx,                         // [s,k,d]
                               int64_t num_rows,              // s
                               int64_t k,                     // k
                               int64_t dim,                   // d
                               int64_t num_active,
                               cudaStream_t stream,
                               bool use_all2all_permute = false,
                               int64_t world_size = -1,
                               int64_t num_local_experts = -1,
                               int64_t capacity = -1) {
  int numel = num_rows * dim;

  int64_t threads = 512;
  if (dim % 4 == 0) {
    int64_t blocks = DivUp<int64_t>(DivUp<int64_t>(numel, 4), threads);
    int64_t s_shared_num = max_shared_s_num(num_rows, dim, threads, 4);
    size_t shared_size = k * s_shared_num * (sizeof(int) + sizeof(float));

    if (!use_all2all_permute) {
      gather_with_mask_kernel<T, 4>
          <<<blocks, threads, shared_size, stream>>>(dy,
                                                     scatter_index,
                                                     combine_weights,
                                                     dx,
                                                     num_rows,
                                                     k,
                                                     dim,
                                                     numel,
                                                     num_active,
                                                     s_shared_num);
    } else {
      PD_CHECK(world_size > 0 && num_local_experts > 0 && capacity > 0);
      gather_with_mask_permute_kernel<T, 4>
          <<<blocks, threads, shared_size, stream>>>(dy,
                                                     scatter_index,
                                                     combine_weights,
                                                     dx,
                                                     num_rows,
                                                     k,
                                                     dim,
                                                     numel,
                                                     num_active,
                                                     s_shared_num,
                                                     capacity,
                                                     world_size,
                                                     num_local_experts);
    }
  } else {
    int64_t blocks = DivUp<int64_t>(DivUp<int64_t>(numel, 1), threads);
    int64_t s_shared_num = max_shared_s_num(num_rows, dim, threads, 1);
    size_t shared_size = k * s_shared_num * (sizeof(int) + sizeof(float));

#ifdef DEBUG_MOE_OP
    std::cerr
        << "[DEBUG-BWD] gather_with_mask without vectorized, s_shared_num="
        << s_shared_num << ", block=" << blocks << std::endl;
#endif

    if (!use_all2all_permute) {
      gather_with_mask_kernel<T, 1>
          <<<blocks, threads, shared_size, stream>>>(dy,
                                                     scatter_index,
                                                     combine_weights,
                                                     dx,
                                                     num_rows,
                                                     k,
                                                     dim,
                                                     numel,
                                                     num_active,
                                                     s_shared_num);
    } else {
      gather_with_mask_permute_kernel<T, 1>
          <<<blocks, threads, shared_size, stream>>>(dy,
                                                     scatter_index,
                                                     combine_weights,
                                                     dx,
                                                     num_rows,
                                                     k,
                                                     dim,
                                                     numel,
                                                     num_active,
                                                     s_shared_num,
                                                     capacity,
                                                     world_size,
                                                     num_local_experts);
    }
  }
}

template <typename T>
__global__ void topk_grad_with_mask(const T* dy,               // [s, k]
                                    const int* topk_idx,       // [s, k]
                                    const T* combine_weights,  // [s, k]
                                    T* dx,                     // [s, e]
                                    int64_t num_rows,          // s
                                    int64_t k,                 // k
                                    int64_t num_experts        // e
) {
  // init dx to zero
  for (int i = blockIdx.x; i < num_rows; i += gridDim.x) {
    int base_grad = i * num_experts;
    for (int j = threadIdx.x; j < num_experts; j += blockDim.x) {
      dx[base_grad + j] = static_cast<T>(0);
    }
    __syncthreads();
    int base_index = i * k;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
      int64_t idx = topk_idx[base_index + j];
      if (combine_weights[base_index + j] > static_cast<T>(0)) {
        dx[base_grad + idx] = dy[base_index + j];
      }
    }
  }
}

// y=zero_part(topk(x)) 的反向过程
// x:  [s,e]
// dy: [s,k]
// X: [s, e] -(topk)-> Y:[s, k] - (越界设置为0)-> combine_weights: [s, k]
template <typename T>
void topk_grad_with_mask_launcher(const T* dy,               // [s, k]
                                  const int* topk_idx,       // [s, k]
                                  const T* combine_weights,  // [s, k]
                                  T* dx,                     // [s, e]
                                  int64_t num_rows,          // s
                                  int64_t k,                 // k
                                  int64_t num_experts,       // e
                                  cudaStream_t stream) {
  int blocks = num_rows;
  int threads = 1024;

  topk_grad_with_mask<T><<<blocks, threads, 0, stream>>>(
      dy, topk_idx, combine_weights, dx, num_rows, k, num_experts);
}

}  // namespace phi
#endif
