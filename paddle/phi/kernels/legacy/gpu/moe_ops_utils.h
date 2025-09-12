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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/legacy/gpu/moe_fuse_op.h"

namespace phi {

#define CUDACHECK(cmd)                          \
  do {                                          \
    cudaError_t e = cmd;                        \
    if (e != cudaSuccess) {                     \
      printf("Failed: Cuda error %s:%d '%s'\n", \
             __FILE__,                          \
             __LINE__,                          \
             cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                       \
    }                                           \
  } while (0)

namespace details {
// --------      getWorkspaceSize      -------- //
template <typename KeyT>
size_t getWorkspaceSize(const int num_rows,
                        const int hidden_size,
                        const int inter_size,
                        const int num_experts,
                        const int k,
                        phi::CubKeyValueSorter &sorter  // NOLINT
) {
  const int num_moe_inputs = AlignTo16(k * num_rows);
  int num_softmax_outs = 0;

  // softmax output, permuted_rows and permuted_experts have moved to outside of
  // moe kernel, allocate them in Encoder or Decoder before invoking FfnLayer
  // forward.
  size_t total_ws_bytes =
      4 * num_moe_inputs *
      sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_

  const int sorter_ws_size_bytes =
      AlignTo16(sorter.getWorkspaceSize(k * num_rows));

  // 用所有 bit 做排序,会降低些许性能,但是防止越界
  total_ws_bytes += sorter_ws_size_bytes;  // intermediate (fc1) output + cub
                                           // sorting workspace
  return total_ws_bytes;
}
}  // namespace details

template <typename T, typename Context>
void topk_gating(const Context &dev_ctx,
                 const T *x,
                 const float *gate_logits,
                 const float *corr_bias,
                 int **permuted_rows,
                 int **permuted_experts,
                 int64_t num_rows,
                 int64_t num_experts,
                 int64_t hidden_size,
                 int64_t capacity,
                 int64_t k,
                 float *combine_weights,
                 int *scatter_index,
                 int64_t *expert_offset,
                 int *expert_id,
                 bool use_pad,
                 cudaStream_t stream) {
  phi::CubKeyValueSorter sorter(stream);

  DenseTensor xpanded_source_row_to_expanded_dest_row_tensor =
      phi::Empty<int, Context>(dev_ctx, IntArray({num_rows, k}));

  DenseTensor active_cnt_tensor =
      phi::Empty<int, Context>(dev_ctx, IntArray({1}));

  int64_t bytes =
      phi::details::getWorkspaceSize<T>(num_rows,
                                        hidden_size,  // hidden-size=0
                                        0,            // inter-size=0
                                        num_experts,
                                        k,
                                        sorter);

  DenseTensor ws_ptr_tensor =
      phi::Empty<int8_t, Context>(dev_ctx, IntArray({bytes}));
  int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

  // Pointers
  int *source_rows_;
  int *permuted_rows_;
  int *permuted_experts_;
  int *expert_id_;

  float *softmax_out_;
  T *fc1_result_;

  const int sorter_ws_size_bytes =
      AlignTo16(sorter.getWorkspaceSize(k * num_rows));

  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(k * num_rows);

  source_rows_ = reinterpret_cast<int *>(ws_ptr);
  permuted_rows_ = source_rows_ + num_moe_inputs;
  permuted_experts_ = permuted_rows_ + num_moe_inputs;
  expert_id_ = permuted_experts_ + num_moe_inputs;

  fc1_result_ = reinterpret_cast<T *>(expert_id_ + num_moe_inputs);
  softmax_out_ = nullptr;

  topk_gating_softmax_kernelLauncher<float>(gate_logits,
                                            corr_bias,
                                            combine_weights,  // output
                                            softmax_out_,     // no use
                                            expert_id,        // output
                                            source_rows_,     // output
                                            num_rows,
                                            num_experts,
                                            k,
                                            stream);

  // modify expert-id according to k
  if (use_pad)  // 为了区分 k=1 选择和 k=2 选择，修改 expert-id
    modify_expert_id_launcher(
        expert_id, expert_id_, k, num_rows, num_experts, stream);

  sorter.run(
      fc1_result_,
      sorter_ws_size_bytes,
      use_pad ? expert_id_ : expert_id,  // key in
      permuted_experts_,                 // key out // [num_row, k]: expert-id
      source_rows_,                      // value in
      permuted_rows_,  // value out //[num_row, k]: id在原 activation 中的位置
      k * num_rows,  // num_rows
      false,
      stream);

  if (use_pad)
    unmodify_expert_id_launcher(
        permuted_experts_, permuted_experts_, k, num_rows, num_experts, stream);

  compute_total_rows_before_expert(
      permuted_experts_, k * num_rows, num_experts, expert_offset, stream);

  *permuted_rows = permuted_rows_;
  *permuted_experts = permuted_experts_;
}

}  // namespace phi
