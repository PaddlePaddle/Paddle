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

#include <algorithm>

#include "paddle/phi/kernels/fusion/fused_softmax_mask_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/fused_softmax_mask_utils.h"

namespace phi {
namespace fusion {

template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseGradGPUKernel(const T* grad_input,
                                             T* grad_output,
                                             const T* softmax_rst,
                                             int batch_count,
                                             int key_seq_len) {
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;

  int data_first_idx =
      (blockDim.y * blockIdx.x + threadIdx.y) * kLocalBatchSize;

  // batch_count might not be a multiple of kLocalBatchSize. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_count - data_first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  // might be many batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  // the first element to process by the current thread
  int offset = data_first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  grad_input += offset;
  grad_output += offset;
  softmax_rst += offset;

  // using float for all inter compute
  float grad_input_reg[kLocalBatchSize][kLocalIterations]{0.0f};
  float softmax_rst_reg[kLocalBatchSize][kLocalIterations]{0.0f};
  T temp_grad_input[kOneLoadingCounts];
  T temp_softmax_rst[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    int batch_data = (i >= local_batches) ? 0 : key_seq_len;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int data_index = kOneLoadingCounts * local_idx + ii * WARP_SIZE;
      if (data_index < batch_data) {
        load_data(temp_grad_input,
                  grad_input + i * key_seq_len + ii * warp_size);
        load_data(temp_softmax_rst,
                  softmax_rst + i * key_seq_len + ii * warp_size);

#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          softmax_rst_reg[i][ii + counter] =
              static_cast<float>(temp_softmax_rst[counter]);
        }
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          grad_input_reg[i][ii + counter] =
              static_cast<float>(temp_grad_input[counter]) *
              softmax_rst_reg[i][ii + counter];
        }
      }
    }
  }

  float samples_sum[kLocalBatchSize];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    samples_sum[i] = grad_input_reg[i][0];
#pragma unroll
    for (int ii = 1; ii < kLocalIterations; ++ii) {
      samples_sum[i] += grad_input_reg[i][ii];
    }
  }
  warp_reduce<float, kLocalBatchSize, warp_size, AddOP>(samples_sum);

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int data_index = kOneLoadingCounts * local_idx + ii * warp_size;
      if (data_index < key_seq_len) {
        // compute gradients
        T samples_out[kOneLoadingCounts];
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          samples_out[counter] =
              grad_input_reg[i][ii + counter] -
              softmax_rst_reg[i][ii + counter] * samples_sum[i];
        }
        load_data(grad_output + i * key_seq_len + ii * warp_size, samples_out);
      }
    }
  }
}

template <typename T, typename Context>
void FusedSoftmaxMaskGradKernel(const Context& dev_ctx,
                                const DenseTensor& out,
                                const DenseTensor& out_grad,
                                DenseTensor* x_grad) {
  auto* grad_x_data = dev_ctx.template Alloc<T>(x_grad);
  auto* grad_y_data = out_grad.data<T>();
  auto* softmax_rst_data = out.data<T>();

  auto y_dim = out_grad.dims();
  auto batches = y_dim[0];
  auto attn_heads = y_dim[1];
  auto query_seq_len = y_dim[2];
  auto key_seq_len = y_dim[3];

  auto stream = dev_ctx.stream();

  int pow2_index = get_pow2(key_seq_len);
  const int next_pow2 = 1 << pow2_index;
  int batch_count = batches * attn_heads * query_seq_len;
  int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
  // use 128 threads per block to maximum gpu utilization
  constexpr int threads_per_block = 128;

  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = batch_count / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

  // launch the kernel based on the pow2_index
  switch (pow2_index) {
    case 5:  // 32
      SoftmaxMaskFuseGradGPUKernel<T, 5><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 6:  // 64
      SoftmaxMaskFuseGradGPUKernel<T, 6><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 7:  // 128
      SoftmaxMaskFuseGradGPUKernel<T, 7><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 8:  // 256
      SoftmaxMaskFuseGradGPUKernel<T, 8><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 9:  // 512
      SoftmaxMaskFuseGradGPUKernel<T, 9><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 10:  // 1024
      SoftmaxMaskFuseGradGPUKernel<T, 10><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 11:  // 2048
      SoftmaxMaskFuseGradGPUKernel<T, 11><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 12:  // 4096
      SoftmaxMaskFuseGradGPUKernel<T, 12><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    case 13:  // 8192
      SoftmaxMaskFuseGradGPUKernel<T, 13><<<blocks, threads, 0, stream>>>(
          grad_y_data, grad_x_data, softmax_rst_data, batch_count, key_seq_len);
      break;
    default:
      break;
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_softmax_mask_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskGradKernel,
                   float,
                   phi::dtype::float16) {}
