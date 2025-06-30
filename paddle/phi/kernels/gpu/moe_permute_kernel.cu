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

namespace phi {

#define CUMSUM_BLOCK_SIZE 48
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

template <typename X_T, typename routemap_T, typename probs_T, bool has_scale>
__global__ __launch_bounds__(512) void tokens_unzip_stable_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const float *__restrict__ XScale,
    const int *__restrict__ expert_base_offset,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    float *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
  using expert_infos_t = expert_infos<probs_T>;
  int local_cumsum = 0;
  int local_expert_offsets;
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset = (blockIdx.x != 0) * CUMSUM_INVALID_TAG;
  __shared__ expert_infos_t
      shared_expert_infos[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS];

  // ---------------Expertwise deterministic job scheduling ---------------
  if (threadIdx.x < num_experts) {
    local_expert_offsets = expert_base_offset[threadIdx.x];
    expert_infos_t local_expert_infos[CUMSUM_BLOCK_SIZE];
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
          local_expert_infos[internal_row] = {
              local_cumsum + local_expert_offsets, proposed.expert_probs};
          local_cumsum += 1;
        }
      }
    }
    // Inter-block communication
    const int anticipate_signal_idx = blockIdx.x * num_experts + threadIdx.x;
    const int push_signal_idx = (blockIdx.x + 1) * num_experts + threadIdx.x;
    if (blockIdx.x != 0) {
      // signal receive from previous block, using light-weight atomicAdd(check)
      // this will not change any data, only do fetch in low-cost
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
      local_expert_infos[i].expert_row_idx =
          (local_expert_infos[i].expert_row_idx == -1)
              ? -1
              : local_expert_infos[i].expert_row_idx + cumsum_offset;
      shared_expert_infos[i][threadIdx.x] = local_expert_infos[i];
    }
  }

  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();
  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
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
      // vec copy
      if constexpr (has_scale) {
        vectorized_memcpy(
            &XScale[(int64_t)row * (int64_t)scale_length],
            &XScale_unzipped[(int64_t)proposed_row_idx * (int64_t)scale_length],
            scale_length);
      }
      vectorized_memcpy(
          &X[(int64_t)row * (int64_t)token_length],
          &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
          token_length);
    }
  }
}
template <typename T, typename Context>
void dispatch_tokens_unzip_stable(const Context &dev_ctx,
                                  const DenseTensor &X,
                                  const DenseTensor &expert_routemap_topk,
                                  const DenseTensor &expert_prob_topk,
                                  const paddle::optional<DenseTensor> &XScale,
                                  const DenseTensor &expert_offsets,
                                  DenseTensor *X_unzipped,
                                  DenseTensor *zipped_expertwise_rowmap,
                                  DenseTensor *token_prob_unzipped,
                                  DenseTensor *XScale_unzipped,
                                  DenseTensor *global_expertwise_block_cumsum,
                                  const int total_zipped_tokens_num,
                                  const int token_length,
                                  const int topk,  // deprecated
                                  const int num_experts,
                                  const int scale_length) {
  dim3 grid, block;
  grid.x =
      (total_zipped_tokens_num + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  block.x = 512;

#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()
#define GET_PTR_DATA(tensor, type) tensor->data<type>()
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)                       \
  auto kernel = tokens_unzip_stable_kernel<TOKEN_T, INT_T, PROB_T, HAS_SCALE>; \
  kernel<<<grid, block, 0, dev_ctx.stream()>>>(                                \
      GET_DATA(X, TOKEN_T),                                                    \
      GET_DATA(expert_routemap_topk, INT_T),                                   \
      GET_DATA(expert_prob_topk, PROB_T),                                      \
      XScale ? XScale.get_ptr()->data<float>() : nullptr,                      \
      GET_DATA(expert_offsets, int),                                           \
      GET_PTR_DATA(X_unzipped, TOKEN_T),                                       \
      GET_PTR_DATA(zipped_expertwise_rowmap, INT_T),                           \
      GET_PTR_DATA(token_prob_unzipped, PROB_T),                               \
      XScale_unzipped->data<float>(),                                          \
      global_expertwise_block_cumsum->data<int>(),                             \
      total_zipped_tokens_num,                                                 \
      token_length,                                                            \
      scale_length,                                                            \
      num_experts,                                                             \
      topk);

#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE) \
  DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                        \
  if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                        \
    HANDLE_EXPERT_CASE(phi::bfloat16, PROB_T, INT_T, false)     \
  } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {            \
    HANDLE_EXPERT_CASE(phi::float8_e4m3fn, PROB_T, INT_T, true) \
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

template <typename T, typename Context>
void MoePermuteKernel(const Context &dev_ctx,
                      const DenseTensor &X,
                      const paddle::optional<DenseTensor> &XScale,
                      const DenseTensor &expert_routemap_topk,
                      const DenseTensor &expert_prob_topk,
                      const int num_experts,
                      const std::vector<int> &tokens_per_expert,
                      const int padding_multiplex,
                      DenseTensor *X_unzipped,
                      DenseTensor *zipped_expertwise_rowmap,
                      DenseTensor *token_prob_unzipped,
                      DenseTensor *XScale_unzipped) {
  const int rows = X.dims()[0];
  const int cols = X.dims()[1];
  PADDLE_ENFORCE_LE(
      num_experts,
      MAX_NUM_EXPERTS,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          MAX_NUM_EXPERTS,
          num_experts));

  const int quanted_cols = (XScale) ? XScale.get_ptr()->dims()[1] : 0;
  int expert_offset[MAX_NUM_EXPERTS];
  int tokens_cumulated = 0;
  for (int i = 0; i < MAX_NUM_EXPERTS; i++) {
    if (i < num_experts) {
      expert_offset[i] = tokens_cumulated;
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
  cudaMemcpyAsync(expert_offset_tensor.data<int>(),
                  expert_offset,
                  sizeof(int) * MAX_NUM_EXPERTS,
                  cudaMemcpyHostToDevice,
                  dev_ctx.stream());
  const int output_rows = tokens_cumulated;
  const int topk_calculated = expert_routemap_topk.dims()[1];
  X_unzipped->Resize({output_rows, cols});
  token_prob_unzipped->Resize({output_rows});
  if (XScale) {
    const int quanted_cols = XScale.get_ptr()->dims()[1];
    XScale_unzipped->Resize({output_rows, quanted_cols});
  }
  dev_ctx.template Alloc<float>(XScale_unzipped);
  dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<float>(token_prob_unzipped);
  auto X_unzipped_ptr = reinterpret_cast<void *>(X_unzipped->data<T>());

  for (int i = 0; i < num_experts; i++) {
    int next_expert_offset =
        i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
    int invalid_rows =
        next_expert_offset - expert_offset[i] - tokens_per_expert[i];
    int cur_expert_end = expert_offset[i] + tokens_per_expert[i];
    cudaMemsetAsync(X_unzipped_ptr + cur_expert_end * cols * sizeof(T),
                    0,
                    sizeof(T) * invalid_rows * cols,
                    dev_ctx.stream());
  }
  if (XScale) {
    auto XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<float>());
    for (int i = 0; i < num_experts; i++) {
      int next_expert_offset =
          i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
      int invalid_rows =
          next_expert_offset - expert_offset[i] - tokens_per_expert[i];
      int cur_expert_end = expert_offset[i] + tokens_per_expert[i];
      cudaMemsetAsync(
          XScale_unzipped_ptr + cur_expert_end * quanted_cols * sizeof(float),
          0,
          sizeof(float) * invalid_rows * quanted_cols,
          dev_ctx.stream());
    }
  }

  auto token_prob_unzipped_ptr =
      reinterpret_cast<void *>(token_prob_unzipped->data<float>());

  for (int i = 0; i < num_experts; i++) {
    int next_expert_offset =
        i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
    int invalid_rows =
        next_expert_offset - expert_offset[i] - tokens_per_expert[i];
    int cur_expert_end = expert_offset[i] + tokens_per_expert[i];
    cudaMemsetAsync(token_prob_unzipped_ptr + cur_expert_end * sizeof(float),
                    0,
                    sizeof(float) * invalid_rows,
                    dev_ctx.stream());
  }
  if (X.numel() == 0) return;
  const int cumsum_blocknum =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  DenseTensor global_expertwise_block_cumsum =
      phi::Full<int, Context>(dev_ctx,
                              phi::IntArray({cumsum_blocknum + 1, num_experts}),
                              CUMSUM_INVALID_TAG);
  dispatch_tokens_unzip_stable<T, Context>(dev_ctx,
                                           X,
                                           expert_routemap_topk,
                                           expert_prob_topk,
                                           XScale,
                                           expert_offset_tensor,
                                           X_unzipped,
                                           zipped_expertwise_rowmap,
                                           token_prob_unzipped,
                                           XScale_unzipped,
                                           &global_expertwise_block_cumsum,
                                           rows,
                                           cols,
                                           topk_calculated,
                                           num_experts,
                                           quanted_cols);
}
#undef CUMSUM_BLOCK_SIZE
#undef CUMSUM_INVALID_TAG
#undef MAX_NUM_EXPERTS
}  // namespace phi

PD_REGISTER_KERNEL(
    moe_permute, GPU, ALL_LAYOUT, phi::MoePermuteKernel, phi::dtype::bfloat16) {
}
