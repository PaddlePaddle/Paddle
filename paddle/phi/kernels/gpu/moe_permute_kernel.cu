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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/moe_permute_utils.h"
#include "paddle/utils/optional.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;

namespace cg = cooperative_groups;

// __device__ void compute(int* global_out, int const* shared_in){
//   global_out[0] = shared_in[0] + 1;
// }
// __global__ void with_single_stage(int* global_out, int const* global_in,
// size_t size, size_t batch_sz) {
//     auto grid = cooperative_groups::this_grid();
//     auto block = cooperative_groups::this_thread_block();
//     assert(size == batch_sz * grid.size()); // Assume input size fits
//     batch_sz * grid_size

//     constexpr size_t stages_count = 1; // Pipeline with one stage
//     // One batch must fit in shared memory:
//     extern __shared__ int shared[];  // block.size() * sizeof(int) bytes

//     // Allocate shared storage for a single stage cuda::pipeline:
//     __shared__ cuda::pipeline_shared_state<
//         cuda::thread_scope::thread_scope_block,
//         stages_count
//     > shared_state;
//     auto pipeline = cuda::make_pipeline(block, &shared_state);

//     // Each thread processes `batch_sz` elements.
//     // Compute offset of the batch `batch` of this thread block in global
//     memory: auto block_batch = [&](size_t batch) -> int {
//       return block.group_index().x * block.size() + grid.size() * batch;
//     };

//     for (size_t batch = 0; batch < batch_sz; ++batch) {
//         size_t global_idx = block_batch(batch);

//         // Collectively acquire the pipeline head stage from all producer
//         threads: pipeline.producer_acquire();

//         // Submit async copies to the pipeline's head stage to be
//         // computed in the next loop iteration
//         cuda::memcpy_async(block, shared, global_in + global_idx, sizeof(int)
//         * block.size(), pipeline);
//         // Collectively commit (advance) the pipeline's head stage
//         pipeline.producer_commit();

//         // Collectively wait for the operations committed to the
//         // previous `compute` stage to complete:
//         pipeline.consumer_wait();

//         // Computation overlapped with the memcpy_async of the "copy" stage:
//         compute(global_out + global_idx, shared);

//         // Collectively release the stage resources
//         pipeline.consumer_release();
//     }
// }

// static constexpr size_t buf_len = 1024;
// __global__ void add_one_kernel(int* data, size_t offset) {
//   // Shared memory 数组。数组整体 size 要对齐 16字节
//   __shared__ alignas(16) int smem_data[buf_len];

//   // 1. a) 用0号线程初始化 barrier，与上面的代码示例类似。
//   //    b)
//   插入一个fence。表示后续执行异步拷贝操作，需要在这个fence之后才执行。
//   #pragma nv_diag_suppress static_var_with_dynamic_init
//   __shared__ barrier bar;
//   if (threadIdx.x == 0) {
//     init(&bar, blockDim.x);                                    // a)
//     cuda::device::experimental::fence_proxy_async_shared_cta();// b)
//   }
//   __syncthreads();

//   // 2. 发起 TMA 异步拷贝。注意：TMA 操作是用单线程发起。
//   if (threadIdx.x == 0) {
//     // 3a. 发起异步拷贝
//     cuda::memcpy_async(
//         smem_data,
//         data + offset,
//         cuda::aligned_size_t<16>(sizeof(smem_data)),
//         bar
//     );
//   }
//   // 3b. 所有线程到达该标记点，barrier内部的计数器会加 1。
//   barrier::arrival_token token = bar.arrive();

//   //
//   3c.等待barrier内部的计数器等于期望数值，即所有线程到达3b点时，当前线程的wait会返回，结束等待。
//   bar.wait(std::move(token));

//   // 4. 在 Shared Memory 上写数据。
//   for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
//     smem_data[i] += 1;
//   }

//   // 5. 插入fence，保证后续的异步拷贝操作在Shared Memory写数据结束后再启动。
//   cuda::device::experimental::fence_proxy_async_shared_cta();   // b)
//   __syncthreads();
//   // 6. 发起从 Shared Memory 到 Global Memory 的异步拷贝操作。
//   if (threadIdx.x == 0) {
//     cuda::device::experimental::cp_async_bulk_shared_to_global(
//         data + offset, smem_data, sizeof(smem_data));
//     // 7. 一种同步方式，创建一个 bulk async-group，异步拷贝在这个 group
//     中运行，当异步拷贝结束后，
//     // group 内部标记为已完成。
//     cuda::device::experimental::cp_async_bulk_commit_group();
//     // 等待 group 完成。模版参数 0 表示要等待小于等于 0 个 bulk async-group
//     完成才结束等待。
//     cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
//   }
// }

namespace phi {

#define CUMSUM_BLOCK_SIZE 48
#define CUMSUM_INVALID_TAG -1
#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 16
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
          bool do_gather>
__global__ __launch_bounds__(512) void tokens_unzip_stable_kernel(
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
  // printf("token_length %d scale_length %d\n", token_length,scale_length);
  using expert_infos_t = expert_infos<probs_T>;
  int local_cumsum = 0;
  int local_expert_offsets;
  int local_expert_end_offsets;
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset = (blockIdx.x != 0) * CUMSUM_INVALID_TAG;
  __shared__ expert_infos_t
      shared_expert_infos[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS];

  // Init shared memory
  for (int i = threadIdx.x; i < CUMSUM_BLOCK_SIZE * MAX_NUM_EXPERTS;
       i += blockDim.x) {
    shared_expert_infos[i / MAX_NUM_EXPERTS][i % MAX_NUM_EXPERTS] =
        expert_infos_t();
  }
  __syncthreads();

  // ---------------Expertwise deterministic job scheduling ---------------
  if (threadIdx.x < num_experts) {
    const int expert_id = threadIdx.x;
    local_expert_offsets = expert_base_offset[expert_id];
    local_expert_end_offsets = expert_base_offset_end[expert_id];
    // expert_infos_t local_expert_infos[CUMSUM_BLOCK_SIZE];

    // From the block with a smaller idx to the block with a larger idx
    const int mid = gridDim.x / 2;
    if (blockIdx.x < mid) {
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
                    &global_expertwise_block_cumsum[anticipate_signal_idx],
                    0)) == CUMSUM_INVALID_TAG) {
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
        // shared_expert_infos[i][threadIdx.x] =
        // shared_expert_infos[i][expert_id];
      }
    } else {  // From the block with a larger idx to the block with a smaller
              // idx
      int local_suffixsum = 0;
      for (int row = block_row_base + CUMSUM_BLOCK_SIZE - 1;
           row >= block_row_base;
           --row) {
        if (row >= total_zipped_tokens_num) continue;
        const int internal_row = row - block_row_base;
#pragma unroll
        for (int k = 0; k < topk; k++) {
          expert_infos_t proposed = {routemap_topk[row * topk + k],
                                     probs_topk[row * topk + k]};
          if (proposed.expert_row_idx == -1) continue;
          if (threadIdx.x == proposed.expert_row_idx) {
            shared_expert_infos[internal_row][expert_id] = {
                local_suffixsum, proposed.expert_probs};
            local_suffixsum += 1;
          }
        }
      }
      // Inter-block communication
      const int anticipate_signal_idx =
          (blockIdx.x + 1) * num_experts + threadIdx.x;
      const int push_signal_idx = blockIdx.x * num_experts + threadIdx.x;
      int suffixsum_offset = 0;
      if (blockIdx.x != gridDim.x - 1) {
        // signal receive from previous block, using light-weight
        // atomicAdd(check) this will not change any data, only do fetch in
        // low-cost
        while ((suffixsum_offset = atomicAdd(
                    &global_expertwise_block_cumsum[anticipate_signal_idx],
                    0)) == CUMSUM_INVALID_TAG) {
        }
      }
      // signal send for next block, with current cumsum
      const int proposed_offset = suffixsum_offset + local_suffixsum;
      global_expertwise_block_cumsum[push_signal_idx] = proposed_offset;
// Intra-block communication;
#pragma unroll
      for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
        shared_expert_infos[i][expert_id].expert_row_idx =
            (shared_expert_infos[i][expert_id].expert_row_idx == -1)
                ? -1
                : local_expert_end_offsets -
                      (shared_expert_infos[i][expert_id].expert_row_idx +
                       suffixsum_offset);
      }
    }
  }

  extern __shared__ float smem_fp32[];
  X_T *smem = reinterpret_cast<X_T *>(smem_fp32);
  X_T *A0 = smem + 0 * token_length;
  X_T *A1 = smem + 1 * token_length;
  cg::thread_block block = cg::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int stages = 2;
// Suppress NVCC warning about dynamic initialization -
// cuda::pipeline_shared_state is trivially initializable and designed for use
// with __shared__ memory
#pragma nv_diag_suppress 20054
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
#pragma nv_diag_default 20054
  auto pipe = cuda::make_pipeline(block, &pstate);
  // Prime stage 0.
  pipe.producer_acquire();
  cuda::memcpy_async(block,
                     A0,
                     X + block_row_base * token_length,
                     cuda::aligned_size_t<32>(token_length * sizeof(X_T)),
                     pipe);
  pipe.producer_commit();

  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();
  const int block_row_end =
      std::min(block_row_base + CUMSUM_BLOCK_SIZE, total_zipped_tokens_num);
  for (int row = block_row_base; row < block_row_end; row++) {
    // OOB check
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
    X_T *a_stage = (internal_row % 2 == 0) ? A0 : A1;
    X_T *a_next = (internal_row % 2 == 0) ? A1 : A0;

    // wait current using stage
    pipe.consumer_wait();
    block.sync();  // ensure shared memory is ready

    // start next stage
    if (row + 1 < block_row_end) {
      pipe.producer_acquire();
      cuda::memcpy_async(block,
                         a_next,
                         X + (row + 1) * token_length,
                         cuda::aligned_size_t<32>(token_length * sizeof(X_T)),
                         pipe);

      pipe.producer_commit();
    }

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
        // vectorized_memcpy(
        //     &X[(int64_t)row * (int64_t)token_length],
        //     &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
        //     token_length);
        vectorized_memcpy(
            a_stage,
            &X_unzipped[(int64_t)proposed_row_idx * (int64_t)token_length],
            token_length);
      }
    }
    pipe.consumer_release();
  }
}

template <typename T, typename Context>
void dispatch_tokens_unzip_stable(const Context &dev_ctx,
                                  const DenseTensor &X,
                                  const DenseTensor &expert_routemap_topk,
                                  const DenseTensor &expert_prob_topk,
                                  const paddle::optional<DenseTensor> &XScale,
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
  block.x = 256;
  int smem = 2 * token_length * sizeof(phi::bfloat16);
#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()
#define GET_PTR_DATA(tensor, type) tensor->data<type>()
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, SCALE_T, HAS_SCALE, DO_GATHER) \
  auto kernel = tokens_unzip_stable_kernel<TOKEN_T,                          \
                                           INT_T,                            \
                                           PROB_T,                           \
                                           SCALE_T,                          \
                                           HAS_SCALE,                        \
                                           DO_GATHER>;                       \
  kernel<<<grid, block, smem, dev_ctx.stream()>>>(                           \
      GET_DATA(X, TOKEN_T),                                                  \
      GET_DATA(expert_routemap_topk, INT_T),                                 \
      GET_DATA(expert_prob_topk, PROB_T),                                    \
      XScale ? GET_PTR_DATA(XScale.get_ptr(), SCALE_T) : nullptr,            \
      GET_DATA(expert_offsets, int),                                         \
      GET_DATA(expert_offset_end, int),                                      \
      GET_PTR_DATA(X_unzipped, TOKEN_T),                                     \
      GET_PTR_DATA(zipped_expertwise_rowmap, INT_T),                         \
      GET_PTR_DATA(token_prob_unzipped, PROB_T),                             \
      GET_PTR_DATA(XScale_unzipped, SCALE_T),                                \
      global_expertwise_block_cumsum->data<int>(),                           \
      total_zipped_tokens_num,                                               \
      token_length,                                                          \
      scale_length,                                                          \
      num_experts,                                                           \
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
                      const paddle::optional<DenseTensor> &XScale,
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
  // // -------- Memset all padding area to zero, with regard to do_gather
  // auto memset_invalid_rows =
  //     [&](void *ptr, int64_t element_size, int64_t stride) {
  //       for (int i = 0; i < num_experts; i++) {
  //         int64_t next_expert_offset =
  //             i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
  //         int64_t invalid_rows =
  //             next_expert_offset - expert_offset[i] - tokens_per_expert[i];
  //         int64_t cur_expert_end = expert_offset[i] + tokens_per_expert[i];
  //         PADDLE_ENFORCE_GPU_SUCCESS(
  //             cudaMemsetAsync(ptr + cur_expert_end * stride * element_size,
  //                             0,
  //                             element_size * invalid_rows * stride,
  //                             dev_ctx.stream()));
  //       }
  //     };
  // if (do_gather) {  // no gather, no memset
  //   memset_invalid_rows(X_unzipped_ptr, sizeof(T), cols);
  //   if (XScale) {
  //     memset_invalid_rows(XScale_unzipped_ptr,
  //                         using_ue8m0_scale ? sizeof(int32_t) :
  //                         sizeof(float), quanted_cols);
  //   }
  // }
  // Probs will be memset to zero whatsoever
  // memset_invalid_rows(token_prob_unzipped_ptr, sizeof(float), 1);

  // memset all
  // auto memset_all = [&](void *ptr, int64_t element_size, int64_t total_size)
  // {
  //   PADDLE_ENFORCE_GPU_SUCCESS(
  //       cudaMemsetAsync(ptr, 0, total_size * element_size,
  //       dev_ctx.stream()));
  // };

  // // if (do_gather) {  // no gather, no memset
  // //   memset_all(X_unzipped_ptr, sizeof(T), output_rows * cols);
  // //   if (XScale) {
  // //     memset_all(XScale_unzipped_ptr,
  // //                         using_ue8m0_scale ? sizeof(int32_t) :
  // //                         sizeof(float), output_rows * quanted_cols);
  // //   }
  // // }
  // // Probs will be memset to zero whatsoever
  // memset_all(token_prob_unzipped_ptr, sizeof(float), output_rows);

  // -------- Initialize semaphore for cumsum ---------------
  const int cumsum_blocknum =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  // DenseTensor global_expertwise_block_cumsum =
  //     phi::Full<int, Context>(dev_ctx,
  //                             phi::IntArray({cumsum_blocknum + 1,
  //                             num_experts}), CUMSUM_INVALID_TAG);

  DenseTensor global_expertwise_block_cumsum;
  global_expertwise_block_cumsum.Resize(
      {static_cast<int64_t>(cumsum_blocknum + 1),
       static_cast<int64_t>(num_experts)});
  dev_ctx.template Alloc<int>(&global_expertwise_block_cumsum);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(global_expertwise_block_cumsum.data<int>(),
                      -1,
                      global_expertwise_block_cumsum.numel() * sizeof(int),
                      dev_ctx.stream()));

  dispatch_tokens_unzip_stable<T, Context>(dev_ctx,
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
