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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/moe_permute_utils.h"
#include "paddle/utils/optional.h"

#if CUDA_VERSION >= 12080
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>
namespace cg = cooperative_groups;
#endif

namespace phi {

#define CUMSUM_BLOCK_SIZE 40
#define CUMSUM_INVALID_TAG -1
#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 64
#endif

template <typename probs_T>
struct expert_infos {
  int expert_row_idx;
  probs_T expert_probs;

  __device__ __host__ expert_infos()
      : expert_row_idx(-1), expert_probs(probs_T(0)) {}
  __device__ __host__ expert_infos(int idx, probs_T prob)
      : expert_row_idx(idx), expert_probs(prob) {}

  __device__ __host__ expert_infos &operator=(const expert_infos &other) {
    expert_row_idx = other.expert_row_idx;
    expert_probs = other.expert_probs;
    return *this;
  }
};

template <typename X_T,
          typename routemap_T,
          typename probs_T,
          typename scale_T,
          bool has_scale,
          bool do_gather,
          int max_num_experts = 64>
__global__ __launch_bounds__(512) void permute_generic_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const scale_T *__restrict__ XScale,
    const int *__restrict__ expert_base_offset,
    const int *__restrict__ expert_base_offset_end,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    scale_T *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
  using expert_infos_t = expert_infos<probs_T>;
  int local_cumsum = 0;
  int local_expert_offsets;
  int local_expert_end_offsets;
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset = (blockIdx.x != 0) * CUMSUM_INVALID_TAG;
  __shared__ expert_infos_t
      shared_expert_infos[CUMSUM_BLOCK_SIZE][max_num_experts];

  // Init shared memory
  for (int i = threadIdx.x; i < CUMSUM_BLOCK_SIZE * max_num_experts;
       i += blockDim.x) {
    shared_expert_infos[i / max_num_experts][i % max_num_experts] =
        expert_infos_t();
  }
  __syncthreads();

  // ---------------Expertwise deterministic job scheduling ---------------
  if (threadIdx.x < num_experts) {
    const int expert_id = threadIdx.x;
    local_expert_offsets = expert_base_offset[expert_id];
    local_expert_end_offsets = expert_base_offset_end[expert_id];

    for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
         row++) {
      if (row >= total_zipped_tokens_num) break;
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        expert_infos_t proposed = {routemap_topk[row * topk + k],
                                   probs_topk[row * topk + k]};
        if (proposed.expert_row_idx == -1) continue;
        if (threadIdx.x == proposed.expert_row_idx) {
          shared_expert_infos[internal_row][expert_id] = {
              local_cumsum + local_expert_offsets, proposed.expert_probs};
          local_cumsum += 1;
        }
      }
    }
    // Inter-block communication
    const int anticipate_signal_idx = blockIdx.x * num_experts + threadIdx.x;
    const int push_signal_idx = (blockIdx.x + 1) * num_experts + threadIdx.x;
    if (blockIdx.x != 0) {
      // signal receive from previous block, using light-weight
      // atomicAdd(check) this will not change any data, only do fetch in
      // low-cost
      while ((cumsum_offset = atomicAdd(
                  &global_expertwise_block_cumsum[anticipate_signal_idx], 0)) ==
             CUMSUM_INVALID_TAG) {
      }
    }
    // signal send for next block, with current cumsum
    const int proposed_offset = cumsum_offset + local_cumsum;
    global_expertwise_block_cumsum[push_signal_idx] = proposed_offset;
// Intra-block communication;
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
      shared_expert_infos[i][expert_id].expert_row_idx =
          (shared_expert_infos[i][expert_id].expert_row_idx == -1)
              ? -1
              : shared_expert_infos[i][expert_id].expert_row_idx +
                    cumsum_offset;
    }
  }

  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();
  const int block_row_end =
      min(block_row_base + CUMSUM_BLOCK_SIZE, total_zipped_tokens_num);
  for (int row = block_row_base; row < block_row_end; row++) {
    // OOB check
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const expert_infos_t this_expert_token_info =
          shared_expert_infos[internal_row][expert];
      const int proposed_row_idx = this_expert_token_info.expert_row_idx;
      if (threadIdx.x == 0)
        zipped_expertwise_rowmap[row * num_experts + expert] = proposed_row_idx;
      if (proposed_row_idx == -1) continue;  // no memcpy
      if (threadIdx.x == 0)
        probs_unzipped[proposed_row_idx] = this_expert_token_info.expert_probs;

      if constexpr (do_gather) {
        // vec copy
        if constexpr (has_scale) {
          // src or dst may be unaligned with 128bits
          try_vectorized_memcpy(&XScale[(int64_t)row * (int64_t)scale_length],
                                &XScale_unzipped[(int64_t)proposed_row_idx *
                                                 (int64_t)scale_length],
                                scale_length);
        }
        vectorized_memcpy(
            &X[(int64_t)row * (int64_t)token_length],
            &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
            token_length);
      }
    }
  }
}

template <typename X_T,
          typename routemap_T,
          typename probs_T,
          typename scale_T,
          bool has_scale,
          bool do_gather,
          int max_num_experts = 32>
__global__ __launch_bounds__(256) void permute_opt_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const scale_T *__restrict__ XScale,
    const int *__restrict__ expert_base_offset,
    const int *__restrict__ expert_base_offset_end,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    scale_T *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
// This kernel using TMA, so it only be compiled on Hopper or above architecture
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  using expert_infos_t = expert_infos<probs_T>;
  int local_cumsum = 0;
  int local_expert_offsets;
  int local_expert_end_offsets;
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  const int block_row_end =
      min(block_row_base + CUMSUM_BLOCK_SIZE, total_zipped_tokens_num);
  int cumsum_offset = (blockIdx.x != 0) * CUMSUM_INVALID_TAG;
  __shared__ expert_infos_t
      shared_expert_infos[CUMSUM_BLOCK_SIZE][max_num_experts];

  constexpr int stages = 2;
  // Preload the tokens to smem
  extern __shared__ float
      smem_fp32[];  // NVCC do not support extern __shared__ X_T smem[];
  X_T *smem = reinterpret_cast<X_T *>(smem_fp32);
  X_T *A0 = smem + 0 * token_length;
  X_T *A1 = smem + 1 * token_length;

  routemap_T *__restrict__ shared_routemap_topk =
      reinterpret_cast<routemap_T *>(smem + token_length * stages);
  probs_T *__restrict__ shared_probs_topk = reinterpret_cast<probs_T *>(
      shared_routemap_topk + CUMSUM_BLOCK_SIZE * topk);

  cg::thread_block block = cg::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;

// Suppress NVCC warning about dynamic initialization -
// cuda::pipeline_shared_state is trivially initializable and designed for use
// with __shared__ memory
#pragma nv_diag_suppress 20054
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
#pragma nv_diag_default 20054
  auto pipe = cuda::make_pipeline(block, &pstate);

  const int local_rowmap_size = (block_row_end - block_row_base) * topk;
  // Asynchronously read the topk map and probs for current block into smem
  pipe.producer_acquire();
  cuda::memcpy_async(
      block,
      shared_routemap_topk,
      routemap_topk + static_cast<int64_t>(block_row_base) * topk,
      cuda::aligned_size_t<32>(local_rowmap_size * sizeof(routemap_T)),
      pipe);

  cuda::memcpy_async(
      block,
      shared_probs_topk,
      probs_topk + static_cast<int64_t>(block_row_base) * topk,
      cuda::aligned_size_t<32>(local_rowmap_size * sizeof(probs_T)),
      pipe);
  pipe.producer_commit();

  if constexpr (do_gather) {
    // Prime stage 0 for tokens prefetch read
    pipe.producer_acquire();
    cuda::memcpy_async(block,
                       A0,
                       X + static_cast<int64_t>(block_row_base) * token_length,
                       cuda::aligned_size_t<32>(token_length * sizeof(X_T)),
                       pipe);
    pipe.producer_commit();
  }

  // Init shared memory
#pragma unroll
  for (int i = threadIdx.x; i < CUMSUM_BLOCK_SIZE * max_num_experts;
       i += blockDim.x) {
    shared_expert_infos[i / max_num_experts][i % max_num_experts] =
        expert_infos_t();
  }
  // Waiting for the shared_routemap_topk and shared_probs_topk async loading
  pipe.consumer_wait();
  __syncthreads();
  // Finishing  shared_routemap_topk and shared_probs_topk loading, we should
  // release the stage
  pipe.consumer_release();

  // ---------------Expertwise deterministic job scheduling ---------------
  if (threadIdx.x < num_experts) {
    const int expert_id = threadIdx.x;
    local_expert_offsets = expert_base_offset[expert_id];
    local_expert_end_offsets = expert_base_offset_end[expert_id];

    for (int row = block_row_base; row < block_row_end; row++) {
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        expert_infos_t proposed = {
            shared_routemap_topk[internal_row * topk + k],
            shared_probs_topk[internal_row * topk + k]};
        // const expert_infos_t proposed = {routemap_topk[row * topk + k],
        //                     probs_topk[row * topk + k]};
        if (proposed.expert_row_idx == -1) continue;
        if (threadIdx.x == proposed.expert_row_idx) {
          shared_expert_infos[internal_row][expert_id] = {
              local_cumsum + local_expert_offsets, proposed.expert_probs};
          local_cumsum += 1;
        }
      }
    }
    // Inter-block communication
    const int anticipate_signal_idx = blockIdx.x * num_experts + threadIdx.x;
    const int push_signal_idx = (blockIdx.x + 1) * num_experts + threadIdx.x;
    if (blockIdx.x != 0) {
      // signal receive from previous block, using light-weight
      // atomicAdd(check) this will not change any data, only do fetch in
      // low-cost
      while ((cumsum_offset = atomicAdd(
                  &global_expertwise_block_cumsum[anticipate_signal_idx], 0)) ==
             CUMSUM_INVALID_TAG) {
      }
    }
    // signal send for next block, with current cumsum
    const int proposed_offset = cumsum_offset + local_cumsum;
    global_expertwise_block_cumsum[push_signal_idx] = proposed_offset;
// Intra-block communication;
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
      shared_expert_infos[i][expert_id].expert_row_idx =
          (shared_expert_infos[i][expert_id].expert_row_idx == -1)
              ? -1
              : shared_expert_infos[i][expert_id].expert_row_idx +
                    cumsum_offset;
    }
  }
  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();

  for (int row = block_row_base; row < block_row_end; row++) {
    const int internal_row = row - block_row_base;
    X_T *a_stage = (internal_row % 2 == 0) ? A0 : A1;
    X_T *a_next = (internal_row % 2 == 0) ? A1 : A0;
    if constexpr (do_gather) {
      // wait async loading to SMEM(a_stage) done
      pipe.consumer_wait();
      block.sync();  // ensure shared memory is ready

      // start next stage
      if (row + 1 < block_row_end) {
        pipe.producer_acquire();
        // Start loading the next rows token into SMEM(a_next)
        cuda::memcpy_async(block,
                           a_next,
                           X + static_cast<int64_t>(row + 1) * token_length,
                           cuda::aligned_size_t<32>(token_length * sizeof(X_T)),
                           pipe);

        pipe.producer_commit();
      }
    }

    for (int expert = 0; expert < num_experts; expert++) {
      const expert_infos_t this_expert_token_info =
          shared_expert_infos[internal_row][expert];
      const int proposed_row_idx = this_expert_token_info.expert_row_idx;
      if (threadIdx.x == 0)
        zipped_expertwise_rowmap[row * num_experts + expert] = proposed_row_idx;
      if (proposed_row_idx == -1) continue;  // no memcpy
      if (threadIdx.x == 0)
        probs_unzipped[proposed_row_idx] = this_expert_token_info.expert_probs;

      if constexpr (do_gather) {
#if CUDA_VERSION >= 12080
        // Using TMA copy data from SMEM to GMEM
        if (threadIdx.x == 0) {
          cuda::device::experimental::cp_async_bulk_shared_to_global(
              &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
              a_stage,
              token_length * sizeof(X_T));
          cuda::device::experimental::cp_async_bulk_commit_group();
        }
#else
        vectorized_memcpy(
            a_stage,
            &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
            token_length);
#endif
        // vec copy
        if constexpr (has_scale) {
          // src or dst may be unaligned with 128bits
          try_vectorized_memcpy(&XScale[(int64_t)row * (int64_t)scale_length],
                                &XScale_unzipped[(int64_t)proposed_row_idx *
                                                 (int64_t)scale_length],
                                scale_length);
        }
      }
    }
    if constexpr (do_gather) {
#if CUDA_VERSION >= 12080
      // waiting async writing from SMEM to GMEM done
      if (threadIdx.x == 0) {
        cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
      }
#endif
      // release stage for SMEM (a_stage)
      pipe.consumer_release();
    }
  }
#endif
}

template <typename T, typename Context>
void dispatch_permute_kernel(const Context &dev_ctx,
                             const DenseTensor &X,
                             const DenseTensor &expert_routemap_topk,
                             const DenseTensor &expert_prob_topk,
                             const optional<DenseTensor> &XScale,
                             const DenseTensor &expert_offsets,
                             const DenseTensor &expert_offset_end,
                             DenseTensor *X_unzipped,
                             DenseTensor *zipped_expertwise_rowmap,
                             DenseTensor *token_prob_unzipped,
                             DenseTensor *XScale_unzipped,
                             DenseTensor *global_expertwise_block_cumsum,
                             const int total_zipped_tokens_num,
                             const int token_length,
                             const int topk,  // deprecated
                             const int num_experts,
                             const int scale_length,
                             const bool do_gather,
                             const bool using_ue8m0_scale) {
  dim3 grid, block;
  grid.x =
      (total_zipped_tokens_num + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  static int capability = dev_ctx.GetComputeCapability();

#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()
#define GET_PTR_DATA(tensor, type) tensor->data<type>()
#define MAX_NUM_EXPERTS_FOR_OPT_KERNEL 32

#if CUDA_VERSION >= 12080
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER)   \
  if (capability >= 100 && num_experts <= MAX_NUM_EXPERTS_FOR_OPT_KERNEL &&    \
      is_aligned_in_bytes(token_length * sizeof(TOKEN_T)) &&                   \
      is_aligned_in_bytes(sizeof(INT_T) * topk * CUMSUM_BLOCK_SIZE)) {         \
    DISPATCH_OPT_KERNEL(TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER) \
  } else {                                                                     \
    DISPATCH_GENERIC_KERNEL(                                                   \
        TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER)                 \
  }
#else
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER) \
  DISPATCH_GENERIC_KERNEL(TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER)

#endif

#define DISPATCH_OPT_KERNEL(                                        \
    TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER)          \
  auto kernel = permute_opt_kernel<TOKEN_T,                         \
                                   INT_T,                           \
                                   PROB_T,                          \
                                   SCALE_T,                         \
                                   HAS_SCALE,                       \
                                   DO_GATHER,                       \
                                   MAX_NUM_EXPERTS_FOR_OPT_KERNEL>; \
  block.x = 256;                                                    \
  const int smem =                                                  \
      2 * token_length * sizeof(TOKEN_T) +                          \
      (sizeof(INT_T) + sizeof(PROB_T)) * topk * CUMSUM_BLOCK_SIZE;  \
  kernel<<<grid, block, smem, dev_ctx.stream()>>>(                  \
      GET_DATA(X, TOKEN_T),                                         \
      GET_DATA(expert_routemap_topk, INT_T),                        \
      GET_DATA(expert_prob_topk, PROB_T),                           \
      XScale ? GET_PTR_DATA(XScale.get_ptr(), SCALE_T) : nullptr,   \
      GET_DATA(expert_offsets, int),                                \
      GET_DATA(expert_offset_end, int),                             \
      GET_PTR_DATA(X_unzipped, TOKEN_T),                            \
      GET_PTR_DATA(zipped_expertwise_rowmap, INT_T),                \
      GET_PTR_DATA(token_prob_unzipped, PROB_T),                    \
      GET_PTR_DATA(XScale_unzipped, SCALE_T),                       \
      global_expertwise_block_cumsum->data<int>(),                  \
      total_zipped_tokens_num,                                      \
      token_length,                                                 \
      scale_length,                                                 \
      num_experts,                                                  \
      topk);

#define DISPATCH_GENERIC_KERNEL(                                  \
    TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER)        \
  block.x = 512;                                                  \
  auto kernel = permute_generic_kernel<TOKEN_T,                   \
                                       INT_T,                     \
                                       PROB_T,                    \
                                       SCALE_T,                   \
                                       HAS_SCALE,                 \
                                       DO_GATHER,                 \
                                       MAX_NUM_EXPERTS>;          \
  const int smem = 0;                                             \
  kernel<<<grid, block, smem, dev_ctx.stream()>>>(                \
      GET_DATA(X, TOKEN_T),                                       \
      GET_DATA(expert_routemap_topk, INT_T),                      \
      GET_DATA(expert_prob_topk, PROB_T),                         \
      XScale ? GET_PTR_DATA(XScale.get_ptr(), SCALE_T) : nullptr, \
      GET_DATA(expert_offsets, int),                              \
      GET_DATA(expert_offset_end, int),                           \
      GET_PTR_DATA(X_unzipped, TOKEN_T),                          \
      GET_PTR_DATA(zipped_expertwise_rowmap, INT_T),              \
      GET_PTR_DATA(token_prob_unzipped, PROB_T),                  \
      GET_PTR_DATA(XScale_unzipped, SCALE_T),                     \
      global_expertwise_block_cumsum->data<int>(),                \
      total_zipped_tokens_num,                                    \
      token_length,                                               \
      scale_length,                                               \
      num_experts,                                                \
      topk);

#define HANDLE_SCALE_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, DO_GATHER)  \
  if (using_ue8m0_scale) {                                               \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, int32_t, HAS_SCALE, DO_GATHER) \
  } else {                                                               \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, float, HAS_SCALE, DO_GATHER)   \
  }
#define HANDLE_GATHER_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)   \
  if (do_gather) {                                              \
    HANDLE_SCALE_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, true)  \
  } else {                                                      \
    HANDLE_SCALE_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, false) \
  }

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                        \
  if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                        \
    HANDLE_GATHER_CASE(phi::bfloat16, PROB_T, INT_T, false)     \
  } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {            \
    HANDLE_GATHER_CASE(phi::float8_e4m3fn, PROB_T, INT_T, true) \
  }

#define HANDLE_PROB_TYPE(INT_T)                               \
  if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {       \
    HANDLE_TOKEN_TYPE(phi::bfloat16, INT_T)                   \
  } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) { \
    HANDLE_TOKEN_TYPE(float, INT_T)                           \
  }

  if (DTYPE_CASE(zipped_expertwise_rowmap->dtype(), INT32)) {
    HANDLE_PROB_TYPE(int)
  }

#undef DTYPE_CASE
#undef GET_DATA
#undef DISPATCH_CASE
#undef HANDLE_EXPERT_CASE
#undef HANDLE_TOKEN_TYPE
#undef HANDLE_PROB_TYPE
}
template <typename X_T,
          typename SCALE_T,
          bool FILLING_X_UNZIPPED,
          bool FILLING_X_SCALE_UNZIPPED>
__global__ __launch_bounds__(512) void filling_padding_rows_kernel(
    X_T *__restrict__ X_unzipped_ptr,
    SCALE_T *__restrict__ XScale_unzipped_ptr,
    float *__restrict__ token_prob_unzipped_ptr,
    const int cols,
    const int quanted_cols,
    const int *__restrict__ padding_rows) {
  uint32_t rows = padding_rows[blockIdx.x];
  if constexpr (FILLING_X_UNZIPPED) {
    vectorized_memset(&X_unzipped_ptr[rows * cols], static_cast<X_T>(0), cols);
  }
  if constexpr (FILLING_X_SCALE_UNZIPPED) {
    unrolled_memset(&XScale_unzipped_ptr[rows * quanted_cols],
                    static_cast<SCALE_T>(0),
                    quanted_cols);
  }
  if (threadIdx.x == 0) {
    token_prob_unzipped_ptr[rows] = static_cast<float>(0.0);
  }
}
template <typename X_T, typename SCALE_T, typename Context>
void FillingPaddingRows(const Context &dev_ctx,
                        X_T *X_unzipped_ptr,
                        SCALE_T *XScale_unzipped_ptr,
                        float *token_prob_unzipped_ptr,
                        const int cols,
                        const int quanted_cols,
                        const std::vector<int> &padding_rows) {
  if (padding_rows.empty()) return;

  // Allocate GPU memory for padding_rows using DenseTensor
  DenseTensor padding_tokens_tensor;
  padding_tokens_tensor.Resize({static_cast<int64_t>(padding_rows.size())});
  dev_ctx.template Alloc<int>(&padding_tokens_tensor);

  // Copy padding_rows from host to device
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(padding_tokens_tensor.data<int>(),
                                             padding_rows.data(),
                                             sizeof(int) * padding_rows.size(),
                                             cudaMemcpyHostToDevice,
                                             dev_ctx.stream()));

  dim3 grid, block;
  grid.x = padding_rows.size();
  block.x = 512;
// Launch kernel
#define DISPATCH_CASE(FILLING_X_UNZIPPED, FILLING_X_SCALE_UNZIPPED) \
  filling_padding_rows_kernel<X_T,                                  \
                              SCALE_T,                              \
                              FILLING_X_UNZIPPED,                   \
                              FILLING_X_SCALE_UNZIPPED>             \
      <<<grid, block, 0, dev_ctx.stream()>>>(                       \
          X_unzipped_ptr,                                           \
          XScale_unzipped_ptr,                                      \
          token_prob_unzipped_ptr,                                  \
          cols,                                                     \
          quanted_cols,                                             \
          padding_tokens_tensor.data<int>());
#define HANDLE_X_SCALED(X_UNZIPPED)     \
  if (XScale_unzipped_ptr != nullptr) { \
    DISPATCH_CASE(X_UNZIPPED, true)     \
  } else {                              \
    DISPATCH_CASE(X_UNZIPPED, false)    \
  }

  if (X_unzipped_ptr != nullptr) {
    HANDLE_X_SCALED(true)
  } else {
    HANDLE_X_SCALED(false)
  }
#undef DISPATCH_CASE
#undef HANDLE_X_SCALED
}

template <typename T, typename Context>
void MoePermuteKernel(const Context &dev_ctx,
                      const DenseTensor &X,
                      const optional<DenseTensor> &XScale,
                      const DenseTensor &expert_routemap_topk,
                      const DenseTensor &expert_prob_topk,
                      const int num_experts,
                      const std::vector<int> &tokens_per_expert,
                      const int padding_multiplex,
                      const bool do_gather,
                      const bool using_ue8m0_scale,
                      DenseTensor *X_unzipped,
                      DenseTensor *zipped_expertwise_rowmap,
                      DenseTensor *token_prob_unzipped,
                      DenseTensor *XScale_unzipped) {
  const int64_t rows = X.dims()[0];
  const int64_t cols = X.dims()[1];
  PADDLE_ENFORCE_LE(
      rows,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[0] should be less than "
                                      "INT_MAX, received X.dims()[0]: (%ld)",
                                      rows));
  PADDLE_ENFORCE_LE(
      cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[1] should be less than "
                                      "INT_MAX, received X.dims()[1]: (%ld)",
                                      cols));
  PADDLE_ENFORCE_LE(
      num_experts,
      MAX_NUM_EXPERTS,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          MAX_NUM_EXPERTS,
          num_experts));
  const int64_t quanted_cols = (XScale) ? XScale.get_ptr()->dims()[1] : 0;
  PADDLE_ENFORCE_LE(
      quanted_cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("quanted_cols should be less than "
                                      "INT_MAX, received quanted_cols: (%ld)",
                                      quanted_cols));

  // Expert base offset initialization, tensor numeric range [0, max_token_num]
  int expert_offset[MAX_NUM_EXPERTS];
  int expert_offset_end[MAX_NUM_EXPERTS];
  int tokens_cumulated = 0;
  for (int i = 0; i < MAX_NUM_EXPERTS; i++) {
    if (i < num_experts) {
      expert_offset[i] = tokens_cumulated;
      expert_offset_end[i] = expert_offset[i] + tokens_per_expert[i] - 1;
      tokens_cumulated +=
          ((tokens_per_expert[i] + padding_multiplex - 1) / padding_multiplex) *
          padding_multiplex;
    } else {
      expert_offset[i] = 0;
    }
  }
  DenseTensor expert_offset_tensor;
  expert_offset_tensor.Resize({MAX_NUM_EXPERTS});
  dev_ctx.template Alloc<int>(&expert_offset_tensor);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(expert_offset_tensor.data<int>(),
                                             expert_offset,
                                             sizeof(int) * MAX_NUM_EXPERTS,
                                             cudaMemcpyHostToDevice,
                                             dev_ctx.stream()));

  DenseTensor expert_offset_end_tensor;
  expert_offset_end_tensor.Resize({MAX_NUM_EXPERTS});
  dev_ctx.template Alloc<int>(&expert_offset_end_tensor);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(expert_offset_end_tensor.data<int>(),
                      expert_offset_end,
                      sizeof(int) * MAX_NUM_EXPERTS,
                      cudaMemcpyHostToDevice,
                      dev_ctx.stream()));
  // ------------------- resource allocate -------------------------
  const int output_rows = tokens_cumulated;
  const int64_t topk = expert_routemap_topk.dims()[1];
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  token_prob_unzipped->Resize({output_rows});
  if (do_gather) {  // no gather, no resize.
    X_unzipped->Resize({output_rows, cols});
    if (XScale) {
      // TODO(large-tensor): downstream functors may still use int; guard until
      // upgraded.
      int64_t quanted_cols = XScale.get_ptr()->dims()[1];

      XScale_unzipped->Resize({output_rows, quanted_cols});
    }
  }
  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
  dev_ctx.template Alloc<float>(token_prob_unzipped);
  auto X_unzipped_ptr = reinterpret_cast<void *>(X_unzipped->data<T>());
  auto token_prob_unzipped_ptr =
      reinterpret_cast<void *>(token_prob_unzipped->data<float>());
  void *XScale_unzipped_ptr = nullptr;
  if (using_ue8m0_scale) {
    // if using the ue8m0 scale, four ue8m0 scale will be packed into one int32
    dev_ctx.template Alloc<int32_t>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<int32_t>());
  } else {
    dev_ctx.template Alloc<float>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<float>());
  }
  // Handle 0-size input
  if (X.numel() == 0) return;

  std::vector<int> padding_rows;
  for (int i = 0; i < num_experts; i++) {
    int64_t next_expert_offset =
        i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
    int64_t invalid_rows =
        next_expert_offset - expert_offset[i] - tokens_per_expert[i];
    int64_t cur_expert_end = expert_offset[i] + tokens_per_expert[i];
    for (int i = 0; i < invalid_rows; ++i) {
      padding_rows.push_back(cur_expert_end + i);
    }
  }
  if (using_ue8m0_scale) {
    FillingPaddingRows(dev_ctx,
                       do_gather ? X_unzipped->data<T>() : nullptr,
                       XScale ? XScale_unzipped->data<int32_t>() : nullptr,
                       token_prob_unzipped->data<float>(),
                       cols,
                       quanted_cols,
                       padding_rows);
  } else {
    FillingPaddingRows(dev_ctx,
                       do_gather ? X_unzipped->data<T>() : nullptr,
                       XScale ? XScale_unzipped->data<float>() : nullptr,
                       token_prob_unzipped->data<float>(),
                       cols,
                       quanted_cols,
                       padding_rows);
  }

  // -------- Initialize semaphore for cumsum ---------------
  const int cumsum_blocknum =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;

  DenseTensor global_expertwise_block_cumsum;
  global_expertwise_block_cumsum.Resize(
      {static_cast<int64_t>(cumsum_blocknum + 2),
       static_cast<int64_t>(num_experts)});
  dev_ctx.template Alloc<int>(&global_expertwise_block_cumsum);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(global_expertwise_block_cumsum.data<int>(),
                      -1,
                      global_expertwise_block_cumsum.numel() * sizeof(int),
                      dev_ctx.stream()));

  dispatch_permute_kernel<T, Context>(dev_ctx,
                                      X,
                                      expert_routemap_topk,
                                      expert_prob_topk,
                                      XScale,
                                      expert_offset_tensor,
                                      expert_offset_end_tensor,
                                      X_unzipped,
                                      zipped_expertwise_rowmap,
                                      token_prob_unzipped,
                                      XScale_unzipped,
                                      &global_expertwise_block_cumsum,
                                      static_cast<int>(rows),
                                      static_cast<int>(cols),
                                      static_cast<int>(topk),
                                      num_experts,
                                      static_cast<int>(quanted_cols),
                                      do_gather,
                                      using_ue8m0_scale);
}
#undef CUMSUM_BLOCK_SIZE
#undef CUMSUM_INVALID_TAG
#undef MAX_NUM_EXPERTS
}  // namespace phi

PD_REGISTER_KERNEL(moe_permute,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoePermuteKernel,
                   phi::float8_e4m3fn,
                   phi::bfloat16) {}
