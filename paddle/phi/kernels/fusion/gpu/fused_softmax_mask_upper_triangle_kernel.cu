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
__global__ void SoftmaxMaskFuseUpperTriangleGPUKernel(const T* src,
                                                      T* dst,
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
  int64_t warp_iter_upper_bound =
      (local_block_idx + kOneLoadingCounts * warp_size - 1) / warp_size;

  int64_t local_batches = batch_count - first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  int64_t local_idx = threadIdx.x;

  src += first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  dst += first_idx * key_seq_len + kOneLoadingCounts * local_idx;

  float data[kLocalBatchSize][kLocalIterations];
  T temp_in[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    auto batch_total_number = (i >= local_batches) ? 0 : local_block_idx;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      auto element_index = kOneLoadingCounts * local_idx + ii * warp_size;

      if (element_index < batch_total_number) {
        load_data_upper_tri(temp_in,
                            src + i * key_seq_len_pow_2 + ii * warp_size);

#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          if ((element_index + counter) < batch_total_number) {
            data[i][ii + counter] = static_cast<float>(temp_in[counter]);
          } else {
            data[i][ii + counter] = -std::numeric_limits<float>::infinity();
          }
        }
      } else {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          data[i][ii + counter] = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  float max_value[kLocalBatchSize];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    max_value[i] = data[i][0];
#pragma unroll
    for (int ii = 1; ii < kLocalIterations; ++ii) {
      max_value[i] = (max_value[i] > data[i][ii]) ? max_value[i] : data[i][ii];
    }
  }
  warp_reduce_upper_tri<float, kLocalBatchSize, warp_size, MaxOP_upper_tri>(
      max_value);

  float sum[kLocalBatchSize]{0.0f};
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ++ii) {
      if (ii < warp_iter_upper_bound) {
        data[i][ii] = std::exp((data[i][ii] - max_value[i]));
        sum[i] += data[i][ii];
      }
    }
  }
  warp_reduce_upper_tri<float, kLocalBatchSize, warp_size, AddOP_upper_tri>(
      sum);

  T out[kOneLoadingCounts];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      auto element_index = kOneLoadingCounts * local_idx + ii * warp_size;

      if (element_index < local_block_idx) {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          if (element_index + counter < local_block_idx) {
            out[counter] = data[i][ii + counter] / sum[i];
          } else {
            out[counter] = 0;
          }
        }
        load_data_upper_tri(dst + i * key_seq_len_pow_2 + ii * warp_size, out);
      } else if (element_index < key_seq_len) {
        load_zero_vector_upper_tri(dst + i * key_seq_len_pow_2 +
                                   ii * warp_size);
      } else {
        break;
      }
    }
  }
}

template <typename T, typename Context>
void FusedSoftmaxMaskFuseUpperTriangleKernel(const Context& dev_ctx,
                                             const DenseTensor& x,
                                             DenseTensor* out) {
  auto* x_ptr = &x;

  auto* x_data = x_ptr->data<T>();
  auto* y_data = dev_ctx.template Alloc<T>(out);

  auto x_dim = x_ptr->dims();
  auto batches = x_dim[0];
  auto attn_heads = x_dim[1];
  auto attn_mul_batch = batches * attn_heads;
  auto query_seq_len = x_dim[2];
  auto key_seq_len = x_dim[3];

  PADDLE_ENFORCE_EQ(key_seq_len,
                    query_seq_len,
                    common::errors::InvalidArgument(
                        "Key seq len must be equal with query seq len "
                        "received key len: %d, query len: %d",
                        key_seq_len,
                        query_seq_len));

  PADDLE_ENFORCE_EQ(key_seq_len >= 32 && key_seq_len <= 16384,
                    true,
                    common::errors::InvalidArgument(
                        "Input x's last dim must be between [32, 16384] "
                        "received the last dimension of x is %d",
                        key_seq_len));

  auto stream = dev_ctx.stream();

  int pow2_index = get_pow2_index_value(key_seq_len);
  const int next_pow2 = 1 << pow2_index;
  int64_t batch_count = attn_mul_batch * query_seq_len;
  int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;

  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  PADDLE_ENFORCE_EQ(
      query_seq_len % batches_per_block,
      0,
      common::errors::InvalidArgument(
          "The query seq len (third dim of input X) must can divide the "
          "number of batches per block. The query seq len is %d, while "
          "the number of batches per block is %d.",
          query_seq_len,
          batches_per_block));
  dim3 blocks(query_seq_len,
              (attn_mul_batch + batches_per_block) / batches_per_block,
              1);
  dim3 threads(warp_size, warps_per_block, 1);

  switch (pow2_index) {
    case 5:  // 32
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 5>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 6:  // 64
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 6>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 7:  // 128
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 7>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 8:  // 256
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 8>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 9:  // 512
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 9>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 10:  // 1024
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 10>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 11:  // 2048
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 11>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 12:  // 4096
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 12>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 13:  // 8192
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 13>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    case 14:  // 16384
      SoftmaxMaskFuseUpperTriangleGPUKernel<T, 14>
          <<<blocks, threads, 0, stream>>>(
              x_data, y_data, batch_count, key_seq_len);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented("Too large sequence length."));
      break;
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_softmax_mask_upper_triangle,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskFuseUpperTriangleKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
