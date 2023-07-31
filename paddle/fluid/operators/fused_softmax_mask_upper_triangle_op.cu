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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fused_softmax_mask_upper_triangle_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/generator.h"

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

__device__ __inline__ void load_data_upper_tri(plat::bfloat16* dst,
                                               const plat::bfloat16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data_upper_tri(float* dst, const float* src) {
  *(reinterpret_cast<float4*>(dst)) = *(reinterpret_cast<const float4*>(src));
}

__device__ __inline__ void load_zero_vector_upper_tri(plat::float16* dst) {
  *(reinterpret_cast<float2*>(dst)) = make_float2(0.0f, 0.0f);
}

__device__ __inline__ void load_zero_vector_upper_tri(plat::bfloat16* dst) {
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

template <typename T, typename DeviceContext>
class SoftmaxMaskFuseUpperTriangleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(fused_softmax_mask_upper_triangle_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::SoftmaxMaskFuseUpperTriangleGradKernel,
                          float,
                          plat::float16,
                          plat::bfloat16) {}
