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

#include "paddle/phi/kernels/cal_aux_loss_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {

template <typename T>
__global__ void cal_aux_loss_kernel(
    const T* gate_prob,           /*[s, e]*/
    const int64_t row_gate_prob,  /*seq_len*/
    const int64_t col_gate_prob,  /*expert_num*/
    const int64_t* dispatch_mask, /*[s, e] or [e]*/
    const int64_t row_dispatch_mask,
    const int64_t col_dispatch_mask,
    const T* tokens_mask, /*[s]*/
    const bool* dispatch_tokens_mask,
    const int64_t dispatch_tokens_mask_len, /*global_seq_len*/
    const int64_t num_experts,
    const bool use_group,
    const int64_t moe_k,
    const float clip_min,
    T* l_aux_loss, /*output*/
    T* seqlen_float,
    T* ce) {
  extern __shared__ int64_t aux_loss_shared[];
  static __shared__ float shared_float[1];

  float scale_val = 1.f;

  // 算seqlen_float
  float seqlen_float_f = 0.f;
  if (dispatch_tokens_mask) {
    float local_seqlen_float_f = 0.f;
    int64_t num_k = (dispatch_tokens_mask_len + blockDim.x - 1) / blockDim.x;
    for (int64_t k = 0; k < num_k; ++k) {
      if (k * blockDim.x + threadIdx.x >= dispatch_tokens_mask_len) continue;
      bool mask = dispatch_tokens_mask[k * blockDim.x + threadIdx.x];
      local_seqlen_float_f += static_cast<float>(mask);
    }
    seqlen_float_f =
        phi::funcs::BlockReduceSum<float>(local_seqlen_float_f, 0xFFFFFFFF);

    // 算scale_val
    if (tokens_mask && row_gate_prob != dispatch_tokens_mask_len) {
      float sum_tokens_mask = 0.f;
      float local_sum_tokens_mask = 0.f;
      int64_t num_k = (row_gate_prob + blockDim.x - 1) / blockDim.x;
      for (int64_t k = 0; k < num_k; ++k) {
        if (k * blockDim.x + threadIdx.x >= row_gate_prob) continue;
        T mask = tokens_mask[k * blockDim.x + threadIdx.x];
        local_sum_tokens_mask += static_cast<float>(mask);
      }
      sum_tokens_mask =
          phi::funcs::BlockReduceSum<float>(local_sum_tokens_mask, 0xFFFFFFFF);
      if (threadIdx.x == 0) {
        shared_float[0] = seqlen_float_f / max(sum_tokens_mask, clip_min);
      }
      __syncthreads();
      scale_val = shared_float[0];
    }

  } else if (tokens_mask) {
    float local_seqlen_float_f = 0.f;
    int64_t num_k = (row_gate_prob + blockDim.x - 1) / blockDim.x;
    for (int64_t k = 0; k < num_k; ++k) {
      if (k * blockDim.x + threadIdx.x >= row_gate_prob) continue;
      T mask = tokens_mask[k * blockDim.x + threadIdx.x];
      local_seqlen_float_f += static_cast<float>(mask);
    }
    seqlen_float_f =
        phi::funcs::BlockReduceSum<float>(local_seqlen_float_f, 0xFFFFFFFF);
  } else {
    seqlen_float_f = static_cast<float>(row_gate_prob) /
                     static_cast<float>(num_experts) *
                     static_cast<float>(col_gate_prob);
  }

  if (threadIdx.x == 0) {
    shared_float[0] = max(seqlen_float_f, clip_min);
  }
  __syncthreads();
  seqlen_float_f = shared_float[0];

  __syncthreads();
  // 处理dispatch_mask
  if (col_dispatch_mask > 1) {
    int64_t num_k = (row_dispatch_mask + blockDim.x - 1) / blockDim.x;

    for (int64_t e = 0; e < col_dispatch_mask; e++) {
      int64_t local_sum_val = 0.f;
      for (int64_t k = 0; k < num_k; ++k) {
        int64_t mask_val = 0;
        if (k * blockDim.x + threadIdx.x < row_dispatch_mask) {
          mask_val = static_cast<int64_t>(
              dispatch_mask[(k * blockDim.x + threadIdx.x) * col_dispatch_mask +
                            e]);
        }
        local_sum_val += mask_val;
      }
      int64_t sum_val =
          phi::funcs::BlockReduceSum<int64_t>(local_sum_val, 0xFFFFFFFF);
      if (threadIdx.x == 0) {
        aux_loss_shared[e] = sum_val;
      }
    }
  } else {
    if (threadIdx.x < row_dispatch_mask) {
      aux_loss_shared[threadIdx.x] =
          static_cast<int64_t>(dispatch_mask[threadIdx.x]);
    }
  }

  // 算me和l_aux
  float l_aux = 0.f;
  int64_t num_k = (row_gate_prob + blockDim.x - 1) / blockDim.x;
  for (int64_t e = 0; e < col_gate_prob; e++) {
    float local_sum_val = 0.f;
    for (int64_t k = 0; k < num_k; ++k) {
      float gate_prob_val = 0.f;
      if (k * blockDim.x + threadIdx.x < row_gate_prob) {
        gate_prob_val = static_cast<float>(
            gate_prob[(k * blockDim.x + threadIdx.x) * col_gate_prob + e]);
      }
      local_sum_val += gate_prob_val;
    }
    float sum_val =
        phi::funcs::BlockReduceSum<float>(local_sum_val, 0xFFFFFFFF);
    if (threadIdx.x == 0) {
      float ce_val = static_cast<float>(aux_loss_shared[e]) / seqlen_float_f;
      float me_val = sum_val / seqlen_float_f;
      l_aux += ce_val * me_val * static_cast<float>(num_experts);
      ce[e] = static_cast<T>(ce_val);
    }
  }

  if (threadIdx.x == 0) {
    if (use_group) {
      l_aux /= static_cast<float>(moe_k);
    }
    l_aux = l_aux * scale_val;
    *l_aux_loss = static_cast<T>(l_aux);
    *seqlen_float = static_cast<T>(seqlen_float_f);
  }
}

template <typename T>
void cal_aux_loss(const T* gate_prob,
                  const int64_t row_gate_prob, /*seq_len*/
                  const int64_t col_gate_prob, /*expert_num*/
                  const int64_t* dispatch_mask,
                  const int64_t row_dispatch_mask,
                  const int64_t col_dispatch_mask,
                  const T* tokens_mask,
                  const bool* dispatch_tokens_mask,
                  const int64_t dispatch_tokens_mask_len, /*global_seq_len*/
                  const int64_t num_experts,              /*global_num_experts*/
                  const bool use_group,
                  const int64_t moe_k,
                  const float clip_min,
                  T* l_aux_loss, /*output*/
                  T* seqlen_float,
                  T* ce,
                  cudaStream_t stream) {
  int64_t threads = 1024;
  threads = std::min(row_gate_prob, threads);
  cal_aux_loss_kernel<T>
      <<<1, threads, col_gate_prob * sizeof(int64_t), stream>>>(
          gate_prob,
          row_gate_prob,
          col_gate_prob,
          dispatch_mask,
          row_dispatch_mask,
          col_dispatch_mask,
          tokens_mask,
          dispatch_tokens_mask,
          dispatch_tokens_mask_len,
          num_experts,
          use_group,
          moe_k,
          clip_min,
          l_aux_loss,
          seqlen_float,
          ce);
}

template <typename T, typename Context>
void CalAuxLossKernel(const Context& dev_ctx,
                      const DenseTensor& gate_prob,
                      const DenseTensor& dispatch_mask,
                      const paddle::optional<DenseTensor>& tokens_mask,
                      const paddle::optional<DenseTensor>& dispatch_tokens_mask,
                      int64_t num_experts,
                      bool use_group,
                      int64_t moe_k,
                      float clip_min,
                      DenseTensor* l_aux_loss,
                      DenseTensor* seqlen_float,
                      DenseTensor* ce) {
  auto gate_prob_dims = gate_prob.dims();
  auto dispatch_mask_dims = dispatch_mask.dims();

  int64_t dispatch_tokens_mask_len = 0;
  auto dispatch_tokens_mask_ptr = dispatch_tokens_mask.get_ptr();
  if (dispatch_tokens_mask) {
    const auto mask_dims = dispatch_tokens_mask_ptr->dims();
    const auto dim_size = mask_dims.size();
    const bool is_not_zero_size = (dim_size > 0);
    if (is_not_zero_size) {
      dispatch_tokens_mask_len = dispatch_tokens_mask_ptr->dims()[0];
    } else {
      dispatch_tokens_mask_len = 0;
    }
  }

  /*
  T* l_aux_loss_data = dev_ctx.template Alloc<T>(l_aux_loss);
  T* seqlen_float_data = dev_ctx.template Alloc<T>(seqlen_float);
  T* ce_data = dev_ctx.template Alloc<T>(ce);
  */
  dev_ctx.template Alloc<T>(l_aux_loss);
  dev_ctx.template Alloc<T>(seqlen_float);
  dev_ctx.template Alloc<T>(ce);

  cal_aux_loss<T>(gate_prob.data<T>(),
                  gate_prob_dims[0],
                  gate_prob_dims[1],
                  dispatch_mask.data<int64_t>(),
                  dispatch_mask_dims[0],
                  dispatch_mask_dims.size() > 1 ? dispatch_mask_dims[1]
                                                : static_cast<int64_t>(1),
                  tokens_mask ? tokens_mask.get_ptr()->data<T>() : nullptr,
                  dispatch_tokens_mask
                      ? dispatch_tokens_mask.get_ptr()->data<bool>()
                      : nullptr,
                  dispatch_tokens_mask_len,
                  num_experts,
                  use_group,
                  moe_k,
                  clip_min,
                  l_aux_loss->data<T>(),
                  seqlen_float->data<T>(),
                  ce->data<T>(),
                  dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(cal_aux_loss,
                   GPU,
                   ALL_LAYOUT,
                   phi::CalAuxLossKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
