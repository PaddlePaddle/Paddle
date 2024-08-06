// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fused_softmax_mask_upper_triangle_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/fused_softmax_mask_upper_triangle_utils.h"

namespace phi {
namespace fusion {

template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseUpperTriangleGradGPUKernel(const T* grad_input,
                                                          T* grad_output,
                                                          const T* softmax_rst,
                                                          int64_t batch_count,
                                                          int64_t key_seq_len) {
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;
  int64_t key_seq_len_pow_2 = key_seq_len * key_seq_len;

  int64_t first_idx =
      (static_cast<int64_t>(blockDim.y) * blockIdx.y + threadIdx.y) *
          gridDim.x * kLocalBatchSize +
      blockIdx.x;
  int64_t local_block_idx = blockIdx.x + 1;

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int64_t local_batches = batch_count - first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int64_t local_idx = threadIdx.x;

  // the first element to process by the current thread
  int64_t offset = first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  grad_input += offset;
  grad_output += offset;
  softmax_rst += offset;

  // load data from global memory
  float grad_input_reg[kLocalBatchSize][kLocalIterations]{0.0f};
  float softmax_rst_reg[kLocalBatchSize][kLocalIterations]{0.0f};
  T temp_grad_input[kOneLoadingCounts];
  T temp_softmax_rst[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    auto batch_total_number = (i >= local_batches) ? 0 : local_block_idx;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      auto element_index = kOneLoadingCounts * local_idx + ii * warp_size;
      if (element_index < batch_total_number) {
        load_data_upper_tri(
            temp_grad_input,
            grad_input + i * key_seq_len_pow_2 + ii * warp_size);
        load_data_upper_tri(
            temp_softmax_rst,
            softmax_rst + i * key_seq_len_pow_2 + ii * warp_size);

#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          if (element_index + counter < batch_total_number) {
            softmax_rst_reg[i][ii + counter] =
                static_cast<float>(temp_softmax_rst[counter]);
          }
        }
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          if (element_index + counter < batch_total_number) {
            grad_input_reg[i][ii + counter] =
                static_cast<float>(temp_grad_input[counter]) *
                softmax_rst_reg[i][ii + counter];
          }
        }
      }
    }
  }

  float sum[kLocalBatchSize];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    sum[i] = grad_input_reg[i][0];
#pragma unroll
    for (int ii = 1; ii < kLocalIterations; ++ii) {
      sum[i] += grad_input_reg[i][ii];
    }
  }
  warp_reduce_upper_tri<float, kLocalBatchSize, warp_size, AddOP_upper_tri>(
      sum);

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      auto element_index = kOneLoadingCounts * local_idx + ii * warp_size;
      if (element_index < key_seq_len) {
        // compute gradients
        T samples_out[kOneLoadingCounts];
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          samples_out[counter] = grad_input_reg[i][ii + counter] -
                                 softmax_rst_reg[i][ii + counter] * sum[i];
        }
        load_data_upper_tri(
            grad_output + i * key_seq_len_pow_2 + ii * warp_size, samples_out);
      }
    }
  }
}

template <typename T, typename Context>
void FusedSoftmaxMaskFuseUpperTriangleGradKernel(const Context& dev_ctx,
                                                 const DenseTensor& out,
                                                 const DenseTensor& out_grad,
                                                 DenseTensor* x_grad) {
  auto* grad_y = &out_grad;
  auto* softmax_rst = &out;

  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  auto* grad_y_data = grad_y->data<T>();
  auto* softmax_rst_data = softmax_rst->data<T>();

  auto y_dim = grad_y->dims();
  auto batches = y_dim[0];
  auto attn_heads = y_dim[1];
  auto attn_mul_batch = batches * attn_heads;
  auto query_seq_len = y_dim[2];
  auto key_seq_len = y_dim[3];

  auto stream = dev_ctx.stream();

  int pow2_index = get_pow2_index_value(key_seq_len);
  const int next_pow2 = 1 << pow2_index;
  int64_t batch_count = attn_mul_batch * query_seq_len;
  int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
  // use 128 threads per block to maximum gpu utilization
  constexpr int threads_per_block = 128;

  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  dim3 blocks(query_seq_len,
              (attn_mul_batch + batches_per_block) / batches_per_block,
              1);
  dim3 threads(warp_size, warps_per_block, 1);

  switch (pow2_index) {
    case 5:  // 32
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 5>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 6:  // 64
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 6>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 7:  // 128
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 7>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 8:  // 256
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 8>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 9:  // 512
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 9>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 10:  // 1024
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 10>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 11:  // 2048
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 11>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 12:  // 4096
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 12>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 13:  // 8192
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 13>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    case 14:
      SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 14>
          <<<blocks, threads, 0, stream>>>(grad_y_data,
                                           x_grad_data,
                                           softmax_rst_data,
                                           batch_count,
                                           key_seq_len);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented("Too large sequence length."));
      break;
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_softmax_mask_upper_triangle_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskFuseUpperTriangleGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
