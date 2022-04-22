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
// https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/fused_kernels/scaled_masked_softmax.h
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
#include "paddle/fluid/operators/fused_softmax_mask_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using framework::Tensor;

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define MASK 0xffffffff

namespace plat = paddle::platform;

__device__ __inline__ void load_data(plat::float16* dst,
                                     const plat::float16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data(float* dst, const float* src) {
  *(reinterpret_cast<float4*>(dst)) = *(reinterpret_cast<const float4*>(src));
}

int get_pow2(int value) {
  // get next pow2 index
  int pow2_index = 0;
  while ((1 << pow2_index) < value) {
    ++pow2_index;
  }
  return pow2_index;
}

template <typename T>
struct AddOP {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct MaxOP {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T warp_shfl_xor(T value, int laneMask, int width,
                                           unsigned int mask = MASK) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, int batch, int width, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(T* sum) {
  ReduceOp<T> r;
#pragma unroll
  for (int offset = width / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      T b = warp_shfl_xor(sum[i], offset, width);
      sum[i] = r(sum[i], b);
    }
  }
}

// T == fp16
template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseGPUKernel(const T* x_data, const T* mask_data,
                                         T* y_data, int batch_count,
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
  T temp_mask[kOneLoadingCounts];

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

template <typename T, int pow2_index>
__global__ void SoftmaxMaskFuseGradGPUKernel(const T* grad_input,
                                             T* grad_output,
                                             const T* softmax_rst,
                                             int batch_count, int key_seq_len) {
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

// T only supports fp16
// leave as template only for future update
template <typename Place, typename T>
class SoftmaxMaskFuseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* mask = context.Input<Tensor>("Mask");
    auto* y = context.Output<Tensor>("Out");

    auto* x_data = x->data<T>();
    auto* mask_data = mask->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());

    auto x_dim = x->dims();
    auto mask_dim = mask->dims();
    auto batches = x_dim[0];
    auto attn_heads = x_dim[1];
    auto query_seq_len = x_dim[2];
    auto key_seq_len = x_dim[3];

    PADDLE_ENFORCE_GT(query_seq_len, 1,
                      platform::errors::InvalidArgument(
                          "Input x's second last dim must be large than 1 but "
                          "received the second last dimension of x is %d",
                          query_seq_len));

    PADDLE_ENFORCE_EQ(key_seq_len >= 32 && key_seq_len < 8192, true,
                      platform::errors::InvalidArgument(
                          "Input x's last dim must be between [32, 8192) "
                          "received the last dimension of x is %d",
                          key_seq_len));

    PADDLE_ENFORCE_EQ(mask_dim[1], 1,
                      platform::errors::InvalidArgument(
                          "Input mask's second dim must be 1 "
                          "received the second dimension of mask is %d",
                          mask_dim[1]));

    // dim of x and mask must be equal
    for (size_t idx = 0; idx < 4; ++idx) {
      if (idx == 1) continue;
      PADDLE_ENFORCE_EQ(
          x_dim[idx], mask_dim[idx],
          platform::errors::InvalidArgument(
              "Input x's %dth dim should be equal with input mask's %dth dim "
              "but "
              "received the %dth dimension of x and mask are not equal "
              "the %dth dim of x is %d, while the %dth dim of mask is %d.",
              idx, idx, idx, idx, x_dim[idx], idx, mask_dim[idx]));
    }

    auto& place = *context.template device_context<Place>().eigen_device();
    auto stream = context.cuda_device_context().stream();

    int pow2_index = get_pow2(key_seq_len);
    const int next_pow2 = 1 << pow2_index;
    int batch_count = batches * attn_heads * query_seq_len;
    int warp_size = (next_pow2 < WARP_SIZE) ? next_pow2 : WARP_SIZE;
    int batches_per_warp = (next_pow2 <= 128) ? 2 : 1;
    // use 128 threads per block to maximum gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    PADDLE_ENFORCE_EQ(
        query_seq_len % batches_per_block, 0,
        platform::errors::InvalidArgument(
            "The query seq len (third dim of input X) must can divide the "
            "number of batches per block. The query seq len is %d, while "
            "the number of batches per block is %d.",
            query_seq_len, batches_per_block));
    dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
    dim3 threads(warp_size, warps_per_block, 1);

    // launch the kernel based on the pow2_index
    switch (pow2_index) {
      case 5:  // 32
        SoftmaxMaskFuseGPUKernel<T, 5><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 6:  // 64
        SoftmaxMaskFuseGPUKernel<T, 6><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 7:  // 128
        SoftmaxMaskFuseGPUKernel<T, 7><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 8:  // 256
        SoftmaxMaskFuseGPUKernel<T, 8><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 9:  // 512
        SoftmaxMaskFuseGPUKernel<T, 9><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 10:  // 1024
        SoftmaxMaskFuseGPUKernel<T, 10><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 11:  // 2048
        SoftmaxMaskFuseGPUKernel<T, 11><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 12:  // 4096
        SoftmaxMaskFuseGPUKernel<T, 12><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      case 13:  // 8192
        SoftmaxMaskFuseGPUKernel<T, 13><<<blocks, threads, 0, stream>>>(
            x_data, mask_data, y_data, batch_count, key_seq_len);
        break;
      default:
        break;
    }
  }
};

template <typename Place, typename T>
class SoftmaxMaskFuseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* softmax_rst = context.Input<Tensor>("Softmax");

    auto* grad_x_data = grad_x->mutable_data<T>(context.GetPlace());
    auto* grad_y_data = grad_y->data<T>();
    auto* softmax_rst_data = softmax_rst->data<T>();

    auto y_dim = grad_y->dims();
    auto batches = y_dim[0];
    auto attn_heads = y_dim[1];
    auto query_seq_len = y_dim[2];
    auto key_seq_len = y_dim[3];

    auto& place = *context.template device_context<Place>().eigen_device();
    auto stream = context.cuda_device_context().stream();

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
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 6:  // 64
        SoftmaxMaskFuseGradGPUKernel<T, 6><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 7:  // 128
        SoftmaxMaskFuseGradGPUKernel<T, 7><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 8:  // 256
        SoftmaxMaskFuseGradGPUKernel<T, 8><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 9:  // 512
        SoftmaxMaskFuseGradGPUKernel<T, 9><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 10:  // 1024
        SoftmaxMaskFuseGradGPUKernel<T, 10><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 11:  // 2048
        SoftmaxMaskFuseGradGPUKernel<T, 11><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 12:  // 4096
        SoftmaxMaskFuseGradGPUKernel<T, 12><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
            key_seq_len);
        break;
      case 13:  // 8192
        SoftmaxMaskFuseGradGPUKernel<T, 13><<<blocks, threads, 0, stream>>>(
            grad_y_data, grad_x_data, softmax_rst_data, batch_count,
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
    fused_softmax_mask,
    ops::SoftmaxMaskFuseKernel<plat::CUDADeviceContext, plat::float16>,
    ops::SoftmaxMaskFuseKernel<plat::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    fused_softmax_mask_grad,
    ops::SoftmaxMaskFuseGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::SoftmaxMaskFuseGradKernel<plat::CUDADeviceContext, float>);
