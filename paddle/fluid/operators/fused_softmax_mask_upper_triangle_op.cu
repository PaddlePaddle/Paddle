/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// this file is inspired by:
// https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/fused_kernels/scaled_upper_triang_masked_softmax.h
/* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#endif
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <string>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fused_softmax_mask_upper_triangle_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define MASK 0xffffffff

namespace plat = paddle::platform;

__device__ __inline__ void load_data_upper_tri(plat::float16* dst,
                                               const plat::float16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data_upper_tri(float* dst, const float* src) {
  *(reinterpret_cast<float4*>(dst)) = *(reinterpret_cast<const float4*>(src));
}

__device__ __inline__ void load_zero_vector_upper_tri(plat::float16* dst) {
  *(reinterpret_cast<float2*>(dst)) = make_float2(0.0f, 0.0f);
}

__device__ __inline__ void load_zero_vector_upper_tri(float* dst) {
  *(reinterpret_cast<float4*>(dst)) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

int get_pow2_index_value(int value) {
  int pow2_index = 0;
  while ((1 << pow2_index) < value) {
    ++pow2_index;
  }
  return pow2_index;
}

template <typename T>
struct AddOP_upper_tri {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct MaxOP_upper_tri {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T warp_shfl_xor_upper_tri(T value,
                                                     int laneMask,
                                                     int width,
                                                     unsigned int mask = MASK) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, int batch, int width, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce_upper_tri(T* sum) {
  ReduceOp<T> r;
#pragma unroll
  for (int offset = width / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      T b = warp_shfl_xor_upper_tri(sum[i], offset, width);
      sum[i] = r(sum[i], b);
    }
  }
}

template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseUpperTriangleGPUKernel(const T* src,
                                                      T* dst,
                                                      int batch_count,
                                                      int key_seq_len) {
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;
  int key_seq_len_pow_2 = key_seq_len * key_seq_len;

  int first_idx =
      (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * kLocalBatchSize +
      blockIdx.x;
  int local_block_idx = blockIdx.x + 1;
  int warp_iter_upper_bound =
      (local_block_idx + kOneLoadingCounts * warp_size - 1) / warp_size;

  int local_batches = batch_count - first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  int local_idx = threadIdx.x;

  src += first_idx * key_seq_len + kOneLoadingCounts * local_idx;
  dst += first_idx * key_seq_len + kOneLoadingCounts * local_idx;

  float data[kLocalBatchSize][kLocalIterations];
  T temp_in[kOneLoadingCounts];

#pragma unroll
  for (int i = 0; i < kLocalBatchSize; ++i) {
    int batch_total_number = (i >= local_batches) ? 0 : local_block_idx;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int element_index = kOneLoadingCounts * local_idx + ii * warp_size;

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
      int element_index = kOneLoadingCounts * local_idx + ii * warp_size;

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

template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseUpperTriangleGradGPUKernel(const T* grad_input,
                                                          T* grad_output,
                                                          const T* softmax_rst,
                                                          int batch_count,
                                                          int key_seq_len) {
  constexpr int next_pow2 = 1 << pow2_index;
  constexpr int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
  constexpr int kLocalIterations = std::max(next_pow2 / warp_size, 4);
  constexpr int kLocalBatchSize = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kOneLoadingCounts = 4;
  int key_seq_len_pow_2 = key_seq_len * key_seq_len;

  int first_idx =
      (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * kLocalBatchSize +
      blockIdx.x;
  int local_block_idx = blockIdx.x + 1;

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_count - first_idx;
  if (local_batches > kLocalBatchSize) local_batches = kLocalBatchSize;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  // the first element to process by the current thread
  int offset = first_idx * key_seq_len + kOneLoadingCounts * local_idx;
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
    int batch_total_number = (i >= local_batches) ? 0 : local_block_idx;

#pragma unroll
    for (int ii = 0; ii < kLocalIterations; ii += kOneLoadingCounts) {
      int element_index = kOneLoadingCounts * local_idx + ii * warp_size;
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
      int element_index = kOneLoadingCounts * local_idx + ii * warp_size;
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

template <typename Place, typename T>
class SoftmaxMaskFuseUpperTriangleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* y = context.Output<phi::DenseTensor>("Out");

    auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());

    auto x_dim = x->dims();
    auto batches = x_dim[0];
    auto attn_heads = x_dim[1];
    auto attn_mul_batch = batches * attn_heads;
    auto query_seq_len = x_dim[2];
    auto key_seq_len = x_dim[3];

    PADDLE_ENFORCE_EQ(key_seq_len,
                      query_seq_len,
                      platform::errors::InvalidArgument(
                          "Key seq len must be equal with query seq len "
                          "received key len: %d, query len: %d",
                          key_seq_len,
                          query_seq_len));

    PADDLE_ENFORCE_EQ(key_seq_len >= 32 && key_seq_len < 8192,
                      true,
                      platform::errors::InvalidArgument(
                          "Input x's last dim must be between [32, 8192) "
                          "received the last dimension of x is %d",
                          key_seq_len));

    auto& place = *context.template device_context<Place>().eigen_device();
    auto stream = context.cuda_device_context().stream();

    int pow2_index = get_pow2_index_value(key_seq_len);
    const int next_pow2 = 1 << pow2_index;
    int batch_count = attn_mul_batch * query_seq_len;
    int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
    int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    PADDLE_ENFORCE_EQ(
        query_seq_len % batches_per_block,
        0,
        platform::errors::InvalidArgument(
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
      default:
        break;
    }
  }
};

template <typename Place, typename T>
class SoftmaxMaskFuseUpperTriangleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* grad_y =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* softmax_rst = context.Input<phi::DenseTensor>("Softmax");

    auto* grad_x_data = grad_x->mutable_data<T>(context.GetPlace());
    auto* grad_y_data = grad_y->data<T>();
    auto* softmax_rst_data = softmax_rst->data<T>();

    auto y_dim = grad_y->dims();
    auto batches = y_dim[0];
    auto attn_heads = y_dim[1];
    auto attn_mul_batch = batches * attn_heads;
    auto query_seq_len = y_dim[2];
    auto key_seq_len = y_dim[3];

    auto& place = *context.template device_context<Place>().eigen_device();
    auto stream = context.cuda_device_context().stream();

    int pow2_index = get_pow2_index_value(key_seq_len);
    const int next_pow2 = 1 << pow2_index;
    int batch_count = attn_mul_batch * query_seq_len;
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
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 6:  // 64
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 6>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 7:  // 128
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 7>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 8:  // 256
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 8>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 9:  // 512
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 9>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 10:  // 1024
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 10>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 11:  // 2048
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 11>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 12:  // 4096
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 12>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      case 13:  // 8192
        SoftmaxMaskFuseUpperTriangleGradGPUKernel<T, 13>
            <<<blocks, threads, 0, stream>>>(grad_y_data,
                                             grad_x_data,
                                             softmax_rst_data,
                                             batch_count,
                                             key_seq_len);
        break;
      default:
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    fused_softmax_mask_upper_triangle,
    ops::SoftmaxMaskFuseUpperTriangleKernel<phi::GPUContext, plat::float16>,
    ops::SoftmaxMaskFuseUpperTriangleKernel<phi::GPUContext, float>);
REGISTER_OP_CUDA_KERNEL(
    fused_softmax_mask_upper_triangle_grad,
    ops::SoftmaxMaskFuseUpperTriangleGradKernel<phi::GPUContext, plat::float16>,
    ops::SoftmaxMaskFuseUpperTriangleGradKernel<phi::GPUContext, float>);
