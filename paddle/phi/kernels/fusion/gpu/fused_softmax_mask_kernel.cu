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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/fused_softmax_mask_utils.h"

namespace phi {
namespace fusion {

#define LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, MT, pow2_index) \
  SoftmaxMaskFuseV2GPUKernel<T, MT, pow2_index>                   \
      <<<blocks, threads, 0, stream>>>(x_data,                    \
                                       mask_data,                 \
                                       y_data,                    \
                                       batch_count,               \
                                       attn_heads,                \
                                       query_seqs,                \
                                       key_seq_len);

#define LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, MT, pow2_index) \
  SoftmaxMaskFuseV1GPUKernel<T, MT, pow2_index>                   \
      <<<blocks, threads, 0, stream>>>(                           \
          x_data, mask_data, y_data, batch_count, key_seq_len);

// T == fp16
// SoftmaxMaskFuseV1GPUKernel is only suitable for small-scale data, with
// limited block partitioning and a relatively small memory index. But it has
// good performance
template <typename T, typename MT, int pow2_index>
__global__ void SoftmaxMaskFuseV1GPUKernel(const T* x_data,
                                           const MT* mask_data,
                                           T* y_data,
                                           int batch_count,
                                           int key_seq_len) {
  // the forward gpu kernel
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;

  int data_first_idx =
      (blockDim.y *
           (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) +
       threadIdx.y) *
      kLocalBatchSize;

  int mask_fist_idx =
      (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) *
      kLocalBatchSize;

  // batch_count might not be a multiple of kLocalBatchSize. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_count - data_first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  // might be many batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  int x_offset = data_first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  int mask_offset = mask_fist_idx * key_seq_len + kOneLoadingCounts * local_idx;
  x_data += x_offset;
  mask_data += mask_offset;
  y_data += x_offset;

  // using float for all inter compute
  float data[kLocalBatchSize][kLocalIterations];
  T temp_data[kOneLoadingCounts];
  MT temp_mask[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    int batch_data = (i >= local_batches) ? 0 : key_seq_len;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int data_index = kOneLoadingCounts * local_idx + ii * warp_size;

      if (data_index < batch_data) {
        int itr_idx = i * key_seq_len + ii * warp_size;

        // efficiently load data from global memory
        load_data(temp_data, x_data + itr_idx);
        load_data(temp_mask, mask_data + itr_idx);

#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          data[i][ii + counter] = static_cast<float>(temp_data[counter]) +
                                  static_cast<float>(temp_mask[counter]);
        }
      } else {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          data[i][ii + counter] = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  // compute max_value
  // max value for each batch for current warp
  float samples_max_value[kLocalBatchSize];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    samples_max_value[i] = data[i][0];
#pragma unroll
    for (int ii = 1; ii < kLocalIterations; ++ii) {
      samples_max_value[i] = (samples_max_value[i] > data[i][ii])
                                 ? samples_max_value[i]
                                 : data[i][ii];
    }
  }
  // max value for each batch for all warp
  warp_reduce<float, kLocalBatchSize, warp_size, MaxOP>(samples_max_value);

  // compute the sum for each batch for current warp
  float samples_sum[kLocalBatchSize]{0.0f};
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ++ii) {
      data[i][ii] = std::exp((data[i][ii] - samples_max_value[i]));
      samples_sum[i] += data[i][ii];
    }
  }
  // samples_sum for each batch for all warp
  warp_reduce<float, kLocalBatchSize, warp_size, AddOP>(samples_sum);

  // load the result from device back to host
  T samples_out[kOneLoadingCounts];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int idx = kOneLoadingCounts * local_idx + ii * warp_size;
      if (idx < key_seq_len) {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          samples_out[counter] = data[i][ii + counter] / samples_sum[i];
        }
        load_data(y_data + i * key_seq_len + ii * warp_size, samples_out);
      } else {
        break;
      }
    }
  }
}

// T == fp16
// SoftmaxMaskFuseV2GPUKernel is used for larger scale data and has stronger
// universality, but its performance is slightly lower than
// SoftmaxMaskFuseV1GPUKernel

template <typename T, typename MT, int pow2_index>
__global__ void SoftmaxMaskFuseV2GPUKernel(const T* x_data,
                                           const MT* mask_data,
                                           T* y_data,
                                           int64_t batch_count,
                                           uint32_t attn_heads,
                                           uint32_t query_seqs,
                                           int key_seq_len) {
  // the forward gpu kernel
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;

  uint32_t blockInGrid = blockIdx.x;

  uint32_t indexInMaskDim0 = blockInGrid / (attn_heads * query_seqs);
  uint32_t indexInMaskDim2 = blockInGrid % (query_seqs);

  int64_t data_first_idx =
      (blockDim.y * static_cast<int64_t>(blockInGrid) + threadIdx.y) *
      kLocalBatchSize;

  // The original implementation was like this
  // int64_t mask_fist_idx =
  //     (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) *
  //     kLocalBatchSize;
  // The mapping relationship is as follows：
  // query_seqs <-> gridDim.x
  // attn_heads <-> gridDim.y
  // indexInMaskDim0 <-> blockIdx.z
  // indexInMaskDim2 <-> blockIdx.x

  int64_t mask_fist_idx =
      (blockDim.y * (indexInMaskDim2 +
                     static_cast<int64_t>(query_seqs) * indexInMaskDim0) +
       threadIdx.y) *
      kLocalBatchSize;

  // batch_count might not be a multiple of kLocalBatchSize. Check how
  // many batches have to computed within this WARP.
  int64_t local_batches = batch_count - data_first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  // might be many batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  int64_t x_offset =
      data_first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  int64_t mask_offset =
      mask_fist_idx * key_seq_len + kOneLoadingCounts * local_idx;
  x_data += x_offset;
  mask_data += mask_offset;
  y_data += x_offset;

  // using float for all inter compute
  float data[kLocalBatchSize][kLocalIterations];
  T temp_data[kOneLoadingCounts];
  MT temp_mask[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    int batch_data = (i >= local_batches) ? 0 : key_seq_len;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int data_index = kOneLoadingCounts * local_idx + ii * warp_size;

      if (data_index < batch_data) {
        int itr_idx = i * key_seq_len + ii * warp_size;

        // efficiently load data from global memory
        load_data(temp_data, x_data + itr_idx);
        load_data(temp_mask, mask_data + itr_idx);

#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          data[i][ii + counter] = static_cast<float>(temp_data[counter]) +
                                  static_cast<float>(temp_mask[counter]);
        }
      } else {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          data[i][ii + counter] = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  // compute max_value
  // max value for each batch for current warp
  float samples_max_value[kLocalBatchSize];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    samples_max_value[i] = data[i][0];
#pragma unroll
    for (int ii = 1; ii < kLocalIterations; ++ii) {
      samples_max_value[i] = (samples_max_value[i] > data[i][ii])
                                 ? samples_max_value[i]
                                 : data[i][ii];
    }
  }
  // max value for each batch for all warp
  warp_reduce<float, kLocalBatchSize, warp_size, MaxOP>(samples_max_value);

  // compute the sum for each batch for current warp
  float samples_sum[kLocalBatchSize]{0.0f};
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ++ii) {
      data[i][ii] = std::exp((data[i][ii] - samples_max_value[i]));
      samples_sum[i] += data[i][ii];
    }
  }
  // samples_sum for each batch for all warp
  warp_reduce<float, kLocalBatchSize, warp_size, AddOP>(samples_sum);

  // load the result from device back to host
  T samples_out[kOneLoadingCounts];
#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int idx = kOneLoadingCounts * local_idx + ii * warp_size;
      if (idx < key_seq_len) {
#pragma unroll
        for (int counter = 0; counter < kOneLoadingCounts; ++counter) {
          samples_out[counter] = data[i][ii + counter] / samples_sum[i];
        }
        load_data(y_data + i * key_seq_len + ii * warp_size, samples_out);
      } else {
        break;
      }
    }
  }
}
template <typename T, typename Context>
static void CallSoftmaxMaskGPUKernelV1(const Context& dev_ctx,
                                       const DenseTensor& x,
                                       const DenseTensor& mask,
                                       DenseTensor* out,
                                       int batch_count,
                                       int key_seq_len,
                                       int pow2_index,
                                       const dim3& blocks,
                                       const dim3& threads) {
  auto* x_data = x.data<T>();
  auto* y_data = out->data<T>();
  auto stream = dev_ctx.stream();
  if (mask.dtype() == x.dtype()) {
    auto* mask_data = mask.data<T>();
    switch (pow2_index) {
      case 5:  // 32
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 5)
        break;
      case 6:  // 64
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 6)
        break;
      case 7:  // 128
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 7)
        break;
      case 8:  // 256
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 8)
        break;
      case 9:  // 512
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 9)
        break;
      case 10:  // 1024
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 10)
        break;
      case 11:  // 2048
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 11)
        break;
      case 12:  // 4096
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 12)
        break;
      case 13:  // 8192
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, T, 13)
        break;
      default:
        break;
    }
  } else if (mask.dtype() == phi::DataType::FLOAT32) {
    auto* mask_data = mask.data<float>();
    switch (pow2_index) {
      case 5:  // 32
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 5)
        break;
      case 6:  // 64
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 6)
        break;
      case 7:  // 128
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 7)
        break;
      case 8:  // 256
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 8)
        break;
      case 9:  // 512
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 9)
        break;
      case 10:  // 1024
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 10)
        break;
      case 11:  // 2048
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 11)
        break;
      case 12:  // 4096
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 12)
        break;
      case 13:  // 8192
        LAUNCH_SOFTMAX_MASK_FUSE_V1_GPU_KERNEL(T, float, 13)
        break;
      default:
        break;
    }
  }
}

template <typename T, typename Context>
static void CallSoftmaxMaskGPUKernelV2(const Context& dev_ctx,
                                       const DenseTensor& x,
                                       const DenseTensor& mask,
                                       DenseTensor* out,
                                       int64_t batch_count,
                                       int64_t attn_heads,
                                       int64_t query_seqs,
                                       int key_seq_len,
                                       int pow2_index,
                                       const dim3& blocks,
                                       const dim3& threads) {
  auto* x_data = x.data<T>();
  auto* y_data = out->data<T>();
  auto stream = dev_ctx.stream();
  if (mask.dtype() == x.dtype()) {
    auto* mask_data = mask.data<T>();
    switch (pow2_index) {
      case 5:  // 32
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 5)
        break;
      case 6:  // 64
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 6)
        break;
      case 7:  // 128
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 7)
        break;
      case 8:  // 256
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 8)
        break;
      case 9:  // 512
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 9)
        break;
      case 10:  // 1024
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 10)
        break;
      case 11:  // 2048
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 11)
        break;
      case 12:  // 4096
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 12)
        break;
      case 13:  // 8192
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, T, 13)
        break;
      default:
        break;
    }
  } else if (mask.dtype() == phi::DataType::FLOAT32) {
    auto* mask_data = mask.data<float>();
    switch (pow2_index) {
      case 5:  // 32
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 5)
        break;
      case 6:  // 64
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 6)
        break;
      case 7:  // 128
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 7)
        break;
      case 8:  // 256
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 8)
        break;
      case 9:  // 512
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 9)
        break;
      case 10:  // 1024
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 10)
        break;
      case 11:  // 2048
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 11)
        break;
      case 12:  // 4096
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 12)
        break;
      case 13:  // 8192
        LAUNCH_SOFTMAX_MASK_FUSE_V2_GPU_KERNEL(T, float, 13)
        break;
      default:
        break;
    }
  }
}
// T only supports fp16
// leave as template only for future update
template <typename T, typename Context>
void FusedSoftmaxMaskKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* y_data = dev_ctx.template Alloc<T>(out);

  auto x_dim = x.dims();
  auto mask_dim = mask.dims();
  auto batches = x_dim[0];
  auto attn_heads = x_dim[1];
  auto query_seq_len = x_dim[2];
  auto key_seq_len = x_dim[3];

  PADDLE_ENFORCE_GT(query_seq_len,
                    1,
                    common::errors::InvalidArgument(
                        "Input x's second last dim must be large than 1 but "
                        "received the second last dimension of x is %d",
                        query_seq_len));

  PADDLE_ENFORCE_EQ(key_seq_len >= 32 && key_seq_len < 8192,
                    true,
                    common::errors::InvalidArgument(
                        "Input x's last dim must be between [32, 8192) "
                        "received the last dimension of x is %d",
                        key_seq_len));

  PADDLE_ENFORCE_EQ(mask_dim[1],
                    1,
                    common::errors::InvalidArgument(
                        "Input mask's second dim must be 1 "
                        "received the second dimension of mask is %d",
                        mask_dim[1]));

  // dim of x and mask must be equal
  for (size_t idx = 0; idx < 4; ++idx) {
    if (idx == 1) continue;
    PADDLE_ENFORCE_EQ(
        x_dim[idx],
        mask_dim[idx],
        common::errors::InvalidArgument(
            "Input x's %dth dim should be equal with input mask's %dth dim "
            "but "
            "received the %dth dimension of x and mask are not equal "
            "the %dth dim of x is %d, while the %dth dim of mask is %d.",
            idx,
            idx,
            idx,
            idx,
            x_dim[idx],
            idx,
            mask_dim[idx]));
  }

  int pow2_index = get_pow2(key_seq_len);
  const int next_pow2 = 1 << pow2_index;
  int64_t batch_count = batches * attn_heads * query_seq_len;
  int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
  // use 128 threads per block to maximum gpu utilization
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

  // The original implementation was like this:
  // dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
  // If attn_heads or batches beyond 65535, it will cause CUDA error 9
  int64_t total_elements = batch_count * key_seq_len;

  dim3 threads(warp_size, warps_per_block, 1);
  if (total_elements > std::numeric_limits<int>::max() ||
      attn_heads > dev_ctx.GetCUDAMaxGridDimSize()[1] ||
      batches > dev_ctx.GetCUDAMaxGridDimSize()[2]) {
    int64_t total_blocks = batch_count / batches_per_block;
    dim3 blocks(total_blocks);
    int64_t query_seqs = query_seq_len / batches_per_block;
    CallSoftmaxMaskGPUKernelV2<T, Context>(dev_ctx,
                                           x,
                                           mask,
                                           out,
                                           batch_count,
                                           attn_heads,
                                           query_seqs,
                                           static_cast<int>(key_seq_len),
                                           pow2_index,
                                           blocks,
                                           threads);
  } else {
    dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
    CallSoftmaxMaskGPUKernelV1<T, Context>(dev_ctx,
                                           x,
                                           mask,
                                           out,
                                           static_cast<int>(batch_count),
                                           static_cast<int>(key_seq_len),
                                           pow2_index,
                                           blocks,
                                           threads);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_softmax_mask,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskKernel,
                   float,
                   phi::dtype::float16) {}
