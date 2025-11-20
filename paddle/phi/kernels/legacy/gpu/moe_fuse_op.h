// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <thrust/adjacent_difference.h>  // 包含常用的 thrust 算法
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include "paddle/common/enforce.h"
#include "paddle/common/exception.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/legacy/gpu/moe_kernel_impl.h"

namespace phi {

template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void moe_top_k(const T* inputs_after_softmax,
                   const T* bias,  // bias could be nullptr if not used
                   T* output,
                   int* indices,
                   int* source_rows,
                   const int num_experts,
                   const int k) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int num_rows = gridDim.x;
  const int block_row = blockIdx.x;
  const int thread_read_offset = blockIdx.x * num_experts;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = bias ? inputs_after_softmax[idx] + bias[expert]
                           : inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      output[idx] =
          bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]
               : result_kvp.value;
      indices[idx] = result_kvp.key;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input,
                                        const T* bias,
                                        T* output,
                                        T* softmax,  // no use
                                        int* indices,
                                        int* source_row,
                                        const int num_rows,
                                        const int num_experts,
                                        const int k,
                                        cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int TPB = 256;
  moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(
      input, bias, output, indices, source_row, num_experts, k);
}

template <typename T>
__global__ void modify_expert_id(const T* expert_id,
                                 T* expert_id_out,
                                 const int k,
                                 const int num_rows,
                                 const int64_t num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * num_rows) return;
  int ik = idx % k;
  int irow = idx / k;
  // const T mask = (~0) >> (8*sizeof(T)-ik); // 最后 ik 位为 1 其他位为 0
  int mask = ik;  // k => 2(11)
  // printf("before: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id[idx], ik);
  int offset = log2(k) + 1;
  expert_id_out[idx] = (expert_id[idx] << offset) | mask;
  // printf("after: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id_out[idx],
  // ik);
}

template <typename T>
void modify_expert_id_launcher(const T* expert_id,
                               T* expert_id_out,
                               const int k,
                               const int num_rows,
                               const int64_t num_experts,
                               const cudaStream_t& stream) {
  int max = 1024;
  const int threads = std::min(max, num_rows * k);
  const int blocks = (num_rows * k + threads - 1) / threads;

  modify_expert_id<T><<<blocks, threads, 0, stream>>>(
      expert_id, expert_id_out, k, num_rows, num_experts);
}

template <typename T>
__global__ void unmodify_expert_id(const T* expert_id,
                                   T* expert_id_out,
                                   const int k,
                                   const int num_rows,
                                   const int64_t num_experts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * num_rows) return;
  int ik = idx % k;
  int irow = idx / k;
  int offset = log2(k) + 1;
  expert_id_out[idx] = (expert_id[idx] >> offset);
}

template <typename T>
void unmodify_expert_id_launcher(const T* expert_id,
                                 T* expert_id_out,
                                 const int k,
                                 const int num_rows,
                                 const int64_t num_experts,
                                 const cudaStream_t& stream) {
  int max = 1024;
  const int threads = std::min(max, num_rows * k);
  const int blocks = (num_rows * k + threads - 1) / threads;

  unmodify_expert_id<T><<<blocks, threads, 0, stream>>>(
      expert_id, expert_id_out, k, num_rows, num_experts);
}

template <typename T>
__device__ inline int find_total_elts_leq_target(const T* sorted_indices,
                                                 const int arr_length,
                                                 const int target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] > target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <typename T>
__global__ void compute_total_rows_before_expert_kernel(
    const T* sorted_experts,
    const int sorted_experts_len,
    const int64_t num_experts,
    int64_t* total_rows_before_expert) {
  // First, compute the global tid. We only need 1 thread per expert.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;

  // This should construct the last index where each expert occurs.
  total_rows_before_expert[expert] =
      find_total_elts_leq_target<T>(sorted_experts, sorted_experts_len, expert);
  // total_rows_before_expert[0] = 0;
  // total_rows_before_expert[1] = 1;
  // if (sorted_experts_len > 3) {
  //     for (int i=0; i<35;i++){
  //         total_rows_before_expert[i] = i;
  //     }
  // }
}

template <typename T>
void compute_total_rows_before_expert(const T* sorted_indices,
                                      const int total_indices,
                                      const int64_t num_experts,
                                      int64_t* total_rows_before_expert,
                                      const cudaStream_t& stream) {
  const int threads = std::min(static_cast<int64_t>(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

template <typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    bool use_pad) {
  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int expanded_dest_row = blockIdx.x;
  const int expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  const int64_t iexpert = permuted_experts[expanded_dest_row];
  const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
  const int64_t row_in_expert = expanded_dest_row - offset;
  if (row_in_expert >= capacity) {
    if (threadIdx.x == 0) {
      expanded_source_row_to_expanded_dest_row[expanded_source_row] =
          0;  // unset scatter-idx
      auto ik = expanded_source_row / num_rows;
      auto isent = expanded_source_row % num_rows;  // transpose
      combine_weights[isent * k + ik] = 0.f;        // unset combine-weight
    }
    return;
  }
  int64_t num_padded = 0;
  if (threadIdx.x == 0) {
    // printf("going through: capacity=%lld, num_active=%lld, row=[%d->%d],
    // row-in-expert %lld\n",
    //     capacity,
    //     num_active,
    //     expanded_dest_row, expanded_source_row,
    //     row_in_expert
    // );
    if (use_pad) num_padded = iexpert * capacity - offset;
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        expanded_dest_row + num_padded;
  }
  // Duplicate and permute rows
  const int source_row = expanded_source_row % num_rows;

  const T* source_row_ptr = unpermuted_input + source_row * cols;
  T* dest_row_ptr;
  if (use_pad) {
    dest_row_ptr =
        permuted_output + iexpert * capacity * cols + row_in_expert * cols;
  } else {
    dest_row_ptr = permuted_output + expanded_dest_row * cols;
  }

  for (int tid = threadIdx.x * VecSize; tid < cols;
       tid += blockDim.x * VecSize) {
    phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
    phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
  }
}

template <typename T>
void initialize_moe_routing_kernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    bool use_pad,
    cudaStream_t stream) {
  const int blocks = num_rows * k;
  const int threads = std::min(cols, 1024);
  constexpr int max_pack_size = 16 / sizeof(T);
  if (cols % max_pack_size == 0) {
    initialize_moe_routing_kernel<T, max_pack_size>
        <<<blocks, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            permuted_experts,
            expert_offset,
            combine_weights,
            num_rows,
            cols,
            k,
            capacity,
            use_pad);
  } else {
    initialize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(
        unpermuted_input,
        permuted_output,
        expanded_dest_row_to_expanded_source_row,
        expanded_source_row_to_expanded_dest_row,
        permuted_experts,
        expert_offset,
        combine_weights,
        num_rows,
        cols,
        k,
        capacity,
        use_pad);
  }
}

/**
 * 原逻辑的output:
 * R0E0
 * R0E1
 * R1E0
 * R1E1
 *
 * 我们想对all2all和专家gemm做overlap, 所以需要将all2all拆成流水线,
 * 为了便于后续计算, 此kernel的output: R0E0 R1E0 R0E1 R1E1
 */
template <typename T, int VecSize, int LoopSize>
__global__ void initialize_moe_routing_permute_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    const int64_t world_size,
    const int64_t num_local_experts) {
  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
#pragma unroll
  for (int i = 0; i < LoopSize; i++) {
    using LoadT = phi::AlignedVector<T, VecSize>;
    LoadT src_vec;
    const int expanded_dest_row = blockIdx.x + i * gridDim.x;
    const int expanded_source_row =
        expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    const int64_t iexpert = permuted_experts[expanded_dest_row];
    const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
    const int64_t row_in_expert = expanded_dest_row - offset;
    if (row_in_expert >= capacity) {
      if (threadIdx.x == 0) {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] =
            0;  // unset scatter-idx
        auto ik = expanded_source_row / num_rows;
        auto isent = expanded_source_row % num_rows;  // transpose
        combine_weights[isent * k + ik] = 0.f;        // unset combine-weight
      }
      continue;
    }
    int64_t num_padded = 0;
    if (threadIdx.x == 0) {
      num_padded = iexpert * capacity - offset;
      expanded_source_row_to_expanded_dest_row[expanded_source_row] =
          expanded_dest_row + num_padded;
    }
    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr;

    const int64_t irank = iexpert / num_local_experts;
    const int64_t local_iexpert = iexpert % num_local_experts;
    dest_row_ptr = permuted_output +
                   local_iexpert * world_size * capacity * cols +
                   irank * capacity * cols + row_in_expert * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols;
         tid += blockDim.x * VecSize) {
      phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
      phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
  }
}

template <typename T>
void initialize_moe_routing_permute_kernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    const int64_t world_size,
    const int64_t num_local_experts,
    cudaStream_t stream) {
  const int loop_size = 2;
  const int blocks = (num_rows * k) / loop_size;
  assert((num_rows * k) % loop_size == 0);
  const int threads = std::min(cols, 1024);
  constexpr int max_pack_size = 16 / sizeof(T);
  if (cols % max_pack_size == 0) {
    initialize_moe_routing_permute_kernel<T, max_pack_size, loop_size>
        <<<blocks, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            permuted_experts,
            expert_offset,
            combine_weights,
            num_rows,
            cols,
            k,
            capacity,
            world_size,
            num_local_experts);
  } else {
    initialize_moe_routing_permute_kernel<T, 1, loop_size>
        <<<blocks, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            permuted_experts,
            expert_offset,
            combine_weights,
            num_rows,
            cols,
            k,
            capacity,
            world_size,
            num_local_experts);
  }
}

// moe_ops_partial_nosoftmaxtopk utils

template <typename T>
void compute_global_expert_offset(
    const T* expert_id,      // [len]
    T* sort_buffer,          // [len]
    int64_t* expert_offset,  // [num_experts]
    const int64_t len,
    const int64_t num_experts,
    const int64_t capacity,
    const cudaStream_t& stream,
    const phi::memory_utils::ThrustAllocator<cudaStream_t>& allocator) {
  auto ptr = thrust::device_pointer_cast(expert_id);
  auto outptr = thrust::device_pointer_cast(sort_buffer);
  auto offsetptr = thrust::device_pointer_cast(expert_offset);
  const auto& exec_policy = thrust::cuda::par(allocator).on(stream);
  thrust::copy(exec_policy, ptr, ptr + len, outptr);
  thrust::sort(exec_policy, outptr, outptr + len);
  const int threads = std::min(static_cast<int64_t>(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
      sort_buffer, len, num_experts, expert_offset);
  thrust::adjacent_difference(
      exec_policy, offsetptr, offsetptr + num_experts, offsetptr);
  // thrust::transform(offsetptr,
  //     offsetptr + num_experts,
  //     thrust::constant_iterator<int64_t>(capacity),
  //     offsetptr,
  //     thrust::minimum<int64_t>()
  // );
}

template <typename T>
__global__ void modify_and_mask_expert_id(const T* expert_id,
                                          T* expert_id_out,
                                          const int k,
                                          const int num_rows,
                                          const int num_experts,
                                          const int expert_start_index,
                                          const int expert_end_index) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * num_rows) return;
  int ik = idx % k;
  int irow = idx / k;
  // const T mask = (~0) >> (8*sizeof(T)-ik); // 最后 ik 位为 1 其他位为 0
  int mask = ik;  // k => 2(11)
  // printf("before: idx=%d, expert-id:%d, ik=%d, s=%d, e=%d\n", idx,
  // expert_id[idx], ik, expert_start_index, expert_end_index);
  int offset = log2(k) + 1;
  if (expert_id[idx] < expert_start_index ||
      expert_id[idx] >= expert_end_index) {
    expert_id_out[idx] = (num_experts << offset);  // -1 means
  } else {
    expert_id_out[idx] = (expert_id[idx] << offset) | mask;
  }
  // printf("after: idx=%d, expert-id:%d, ik=%d\n", idx, expert_id_out[idx],
  // ik);
}

template <typename T>
void modify_and_mask_expert_id_launcher(const T* expert_id,
                                        T* expert_id_out,
                                        const int k,
                                        const int num_rows,
                                        const int num_experts,
                                        const int expert_start_index,
                                        const int expert_end_index,
                                        const cudaStream_t& stream) {
  int max = 1024;
  const int threads = std::min(max, num_rows * k);
  const int blocks = (num_rows * k + threads - 1) / threads;

  modify_and_mask_expert_id<T>
      <<<blocks, threads, 0, stream>>>(expert_id,
                                       expert_id_out,
                                       k,
                                       num_rows,
                                       num_experts,
                                       expert_start_index,
                                       expert_end_index);
}

template <typename T>
void compute_local_expert_offset(
    const T* sorted_expert_id,  // [len]
    int64_t* expert_offset,     // [num_experts]
    int64_t* expert_num,
    const int64_t len,
    const int64_t num_experts,
    const int64_t capacity,
    const cudaStream_t& stream,
    const phi::memory_utils::ThrustAllocator<cudaStream_t>& allocator) {
  auto offset_ptr = thrust::device_pointer_cast(expert_offset);
  auto expert_num_ptr = thrust::device_pointer_cast(expert_num);
  const auto& exec_policy = thrust::cuda::par(allocator).on(stream);
  thrust::fill(
      exec_policy, offset_ptr, offset_ptr + num_experts, static_cast<T>(0));

  const int threads = std::min(static_cast<int64_t>(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
      sorted_expert_id, len, num_experts, expert_offset);
  // 不考虑 capacity 影响
  thrust::adjacent_difference(
      exec_policy, offset_ptr, offset_ptr + num_experts, expert_num_ptr);
}

template <typename T>
__global__ void cal_expert_size_and_filter(T* expert_id,
                                           const int64_t* expert_offset,
                                           int64_t len,
                                           int64_t num_experts,
                                           int64_t capacity,
                                           int64_t expert_start_index,
                                           int64_t expert_end_index,
                                           bool reverse) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) return;
  int64_t off = reverse ? expert_offset[expert_end_index - 1] : 0;
  if (reverse) {
    for (int64_t i = expert_end_index - 1; i >= expert_start_index; --i) {
      if (idx >= expert_offset[i]) break;
      off = expert_offset[i];
    }
  } else {
    for (int64_t i = expert_start_index; i != expert_end_index; ++i) {
      if (idx < expert_offset[i]) break;
      off = expert_offset[i];
    }
  }
  if (reverse) {
    if (((off - 1) - idx) >= capacity) {
      expert_id[idx] = num_experts;
    }
  } else {
    if ((idx - off) >= capacity) {
      expert_id[idx] = num_experts;
    }
  }
}

template <typename T>
void cal_expert_size_and_filter_launcher(T* expert_id,
                                         const int64_t* expert_offset,
                                         int64_t len,
                                         int64_t num_experts,
                                         int64_t capacity,
                                         int64_t expert_start_index,
                                         int64_t expert_end_index,
                                         bool reverse,
                                         const cudaStream_t& stream) {
  if (len <= 0) return;
  const int64_t threads = std::min(static_cast<int64_t>(1024), len);
  const int64_t blocks = (len + threads - 1) / threads;
  cal_expert_size_and_filter<T>
      <<<blocks, threads, 0, stream>>>(expert_id,
                                       expert_offset,
                                       len,
                                       num_experts,
                                       capacity,
                                       expert_start_index,
                                       expert_end_index,
                                       reverse);
}

template <typename T>
__global__ void build_seqsort_kv_pairs_kernel(
    T* seqsort_key,
    T* seqsort_value,
    const int* expanded_dest_row_to_expanded_source_row,
    // int*       expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int k,
    const int64_t num_active,
    const int64_t capacity,
    int64_t expert_start_index,
    bool use_pad) {
  const int expanded_dest_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (expanded_dest_row >= num_rows * k) {
    return;
  }
  const int expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  const int64_t iexpert = permuted_experts[expanded_dest_row];
  const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
  const int64_t row_in_expert = expanded_dest_row - offset;
  // printf("DEBUG %d=>%d, num_active=%lld, offset=%lld, cap=%lld \n",
  // expanded_dest_row,  expanded_source_row, num_active, row_in_expert,
  // capacity); 从此以后不会发生截断，后续的 seqsort 也不会截断。
  // printf("expanded_dest_row:%d row_in_expert:%lld capacity:%lld
  // num_active:%lld\n", expanded_dest_row, row_in_expert, capacity,
  // num_active);
  if ((use_pad && row_in_expert >= capacity) ||
      expanded_dest_row >= num_active) {
    // expanded_source_row_to_expanded_dest_row[expanded_source_row] = 0; //
    // unset scatter-idx
    auto ik = expanded_source_row / num_rows;
    auto isent = expanded_source_row % num_rows;  // transpose
    combine_weights[isent * k + ik] = 0.f;        // unset combine-weight
    return;
  }

  // auto num_padded = use_pad ? (iexpert - expert_start_index) * capacity -
  // offset : 0; expanded_source_row_to_expanded_dest_row[expanded_source_row] =
  // expanded_dest_row + num_padded;

  // Duplicate and permute rows
  T source_row = expanded_source_row % num_rows;

  if (use_pad) {
    // printf("inner print: k=%d num_row=%d before minus %d\n", k, num_rows,
    // source_row);
    seqsort_key[(iexpert - expert_start_index) * capacity + row_in_expert] =
        source_row;  // 为保证 padding 位置(0)在最后, 所以对 pos-id
                     // 取减去其最大值
    seqsort_value[(iexpert - expert_start_index) * capacity + row_in_expert] =
        expanded_source_row;
  } else {
    seqsort_key[expanded_dest_row] = source_row;
    seqsort_value[expanded_dest_row] = expanded_source_row;
  }
}

template <typename T>
void build_seqsort_kv_pairs_kernel_launcher(
    T* seqsort_key,  // 实现初始化为 num-rows，保证 sort 到最后
    T* seqsort_value,
    const int* expanded_dest_row_to_expanded_source_row,
    //    int*         expanded_source_row_to_expanded_dest_row,
    const int* permuted_experts,
    const int64_t* expert_offset,
    float* combine_weights,  // output
    const int num_rows,
    const int k,
    const int64_t num_active,  // -1 expert pos
    const int64_t capacity,
    const int64_t expert_start_index,
    bool use_pad,
    cudaStream_t stream) {
  int max = 1024;
  const int threads = std::min(max, num_rows * k);
  const int blocks = (num_rows * k + threads - 1) / threads;
  build_seqsort_kv_pairs_kernel<<<blocks, threads, 0, stream>>>(
      seqsort_key,
      seqsort_value,
      expanded_dest_row_to_expanded_source_row,
      // expanded_source_row_to_expanded_dest_row,
      permuted_experts,
      expert_offset,
      combine_weights,
      num_rows,
      k,
      num_active,
      capacity,
      expert_start_index,
      use_pad);
}

template <typename T, int VecSize>
__global__ void copy_unpermuted_to_permuted_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* padded_out_to_unpermuted_input,
    const int* padded_out_to_expanded_input,
    int* expanded_input_to_padded_out,
    const int64_t padded_len,
    const int64_t num_rows,
    const int64_t k,
    const int64_t cols) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int padded_dest_row = blockIdx.x;
  if (padded_out_to_unpermuted_input[padded_dest_row] == num_rows) {
    //     padded_out_to_unpermuted_input[padded_dest_row] = -1;
    return;  // padded place
  }
  const int source_row = padded_out_to_unpermuted_input[padded_dest_row];
  const int source_row_expanded = padded_out_to_expanded_input[padded_dest_row];
  if (threadIdx.x == 0) {
    expanded_input_to_padded_out[source_row_expanded] = padded_dest_row;
  }

  const T* source_row_ptr = unpermuted_input + source_row * cols;
  T* padded_dest_row_ptr = permuted_output + padded_dest_row * cols;

  for (int tid = threadIdx.x * VecSize; tid < cols;
       tid += blockDim.x * VecSize) {
    phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
    phi::Store<T, VecSize>(src_vec, &padded_dest_row_ptr[tid]);
  }
  PADDLE_ENFORCE(
      (padded_dest_row < padded_len) && (source_row_expanded < num_rows * k),
      "The index is out of bounds, "
      "origin_input[%d] -> distributed_input:[%d], should < [%ld],[%ld] \n",
      source_row_expanded,
      padded_dest_row,
      num_rows * k,
      padded_len);

  // for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
  //     padded_dest_row_ptr[tid] = source_row_ptr[tid]; // copy
  // }
}

template <typename T>
void copy_unpermuted_to_permuted_kernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* padded_out_to_unpermuted_input,
    const int* padded_out_to_expanded_input,
    int* expanded_input_to_padded_out,
    const int64_t padded_len,
    const int64_t num_rows,  // unpermuted_input_len
    const int64_t k,
    const int64_t num_cols,
    cudaStream_t stream) {
  auto blocks = padded_len;
  auto threads = std::min(num_cols, static_cast<int64_t>(1024));
  constexpr int64_t max_pack_size = 16 / sizeof(T);
  if (num_cols % max_pack_size == 0) {
    copy_unpermuted_to_permuted_kernel<T, max_pack_size>
        <<<blocks, threads, 0, stream>>>(unpermuted_input,
                                         permuted_output,
                                         padded_out_to_unpermuted_input,
                                         padded_out_to_expanded_input,
                                         expanded_input_to_padded_out,
                                         padded_len,
                                         num_rows,
                                         k,
                                         num_cols);
  } else {
    copy_unpermuted_to_permuted_kernel<T, 1>
        <<<blocks, threads, 0, stream>>>(unpermuted_input,
                                         permuted_output,
                                         padded_out_to_unpermuted_input,
                                         padded_out_to_expanded_input,
                                         expanded_input_to_padded_out,
                                         padded_len,
                                         num_rows,
                                         k,
                                         num_cols);
  }
}
}  // namespace phi
